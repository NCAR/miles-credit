# This file includes code adapted from NVIDIA's PhysicsNemo project:
# https://github.com/NVIDIA/physicsnemo

from typing import Any, List, Optional

import dgl
import torch
import torch.distributed as dist
from dgl import DGLGraph
from torch import Tensor

from .distributed_graph import DistributedGraph, partition_graph_by_coordinate_bbox


class CuGraphCSC:
    """Constructs a CuGraphCSC object which is a generic graph object wrapping
    typical fields of the CSC representation. It is intended for easy handling
    of the dedicated graph structures required to call into the optimized cugraph-ops
    routines and is a convenience wrapper around a partioned graph in a distributed
    setting. In the latter case, a conversion to DGL compatible structures is possible.

    Parameters
    ----------
    offsets : Tensor
        The offsets tensor.
    indices : Tensor
        The indices tensor.
    num_src_nodes : int
        The number of source nodes.
    num_dst_nodes : int
        The number of destination nodes.
    ef_indices : Optional[Tensor], optional
        The edge feature indices tensor, by default None.
        These can be used if you want to keep edge-input originally
        indexed over COO-indices instead of permuting it such that they
        can be indexed by CSC-indices.
    reverse_graph_bwd : bool, optional
        Whether to reverse the graph for the backward pass, by default True
    cache_graph : bool, optional
        Whether to cache graph structures when wrapping offsets and indices
        to the corresponding cugraph-ops graph types. If graph change in each
        iteration, set to False, by default True.
    partition_size : int, default=1
        Number of process groups across which graph is distributed. If equal to 1,
        the model is run in a normal Single-GPU congiguration. For details on how
        the graph is partitioned, see ``DistributedGraph``.
    partition_group_name : str, default=None
        Name of process group across which graph is distributed. If partition_size
        is set to 1, the model is run in a normal Single-GPU configuration and the
        specification of a process group is not necessary. If partitition_size > 1,
        passing no process group name leads to a parallelism across the default
        process group. Otherwise, the group size of a process group is expected
        to match partition_size.
    """

    def __init__(
        self,
        offsets: Tensor,
        indices: Tensor,
        num_src_nodes: int,
        num_dst_nodes: int,
        ef_indices: Optional[Tensor] = None,
        reverse_graph_bwd: bool = True,
        cache_graph: bool = True,
        partition_size: Optional[int] = -1,
        partition_group_name: Optional[str] = None,
        graph_partition=None,
    ) -> None:
        self.offsets = offsets
        self.indices = indices
        self.num_src_nodes = num_src_nodes
        self.num_dst_nodes = num_dst_nodes
        self.ef_indices = ef_indices
        self.reverse_graph_bwd = reverse_graph_bwd
        self.cache_graph = cache_graph

        # cugraph-ops structures
        self.bipartite_csc = None
        self.static_csc = None
        # dgl graph
        self.dgl_graph = None

        self.is_distributed = False
        self.dist_csc = None

        if partition_size <= 1:
            self.is_distributed = False
            return

        if self.ef_indices is not None:
            raise AssertionError(
                "DistributedGraph does not support mapping CSC-indices to COO-indices."
            )

        self.dist_graph = DistributedGraph(
            # self.offsets,
            # self.indices,
            partition_size,
            partition_group_name,
            graph_partition=graph_partition,
        )

        # overwrite graph information with local graph after distribution
        self.offsets = self.dist_graph.graph_partition.local_offsets
        self.indices = self.dist_graph.graph_partition.local_indices
        self.num_src_nodes = self.dist_graph.graph_partition.num_local_src_nodes
        self.num_dst_nodes = self.dist_graph.graph_partition.num_local_dst_nodes
        self.is_distributed = True

    @staticmethod
    def from_dgl(
        graph: DGLGraph,
        partition_size: int = 1,
        partition_group_name: Optional[str] = None,
        partition_by_bbox: bool = False,
        src_coordinates: Optional[torch.Tensor] = None,
        dst_coordinates: Optional[torch.Tensor] = None,
        coordinate_separators_min: Optional[List[List[Optional[float]]]] = None,
        coordinate_separators_max: Optional[List[List[Optional[float]]]] = None,
    ):  # pragma: no cover
        # DGL changed their APIs w.r.t. how sparse formats can be accessed
        # this here is done to support both versions
        if hasattr(graph, "adj_tensors"):
            offsets, indices, edge_perm = graph.adj_tensors("csc")
        elif hasattr(graph, "adj_sparse"):
            offsets, indices, edge_perm = graph.adj_sparse("csc")
        else:
            raise ValueError("Passed graph object doesn't support conversion to CSC.")

        n_src_nodes, n_dst_nodes = (graph.num_src_nodes(), graph.num_dst_nodes())

        graph_partition = None

        if partition_by_bbox and partition_size > 1:
            partition_rank = dist.get_rank()
            device = torch.device(
                f"cuda:{partition_rank % torch.cuda.device_count()}"
                if torch.cuda.is_available()
                else "cpu"
            )
            graph_partition = partition_graph_by_coordinate_bbox(
                offsets.to(dtype=torch.int64),
                indices.to(dtype=torch.int64),
                src_coordinates=src_coordinates,
                dst_coordinates=dst_coordinates,
                coordinate_separators_min=coordinate_separators_min,
                coordinate_separators_max=coordinate_separators_max,
                partition_size=partition_size,
                partition_rank=partition_rank,
                device=device,
            )

        graph_csc = CuGraphCSC(
            offsets.to(dtype=torch.int64),
            indices.to(dtype=torch.int64),
            n_src_nodes,
            n_dst_nodes,
            partition_size=partition_size,
            partition_group_name=partition_group_name,
            graph_partition=graph_partition,
        )

        return graph_csc, edge_perm

    def get_src_node_features_in_partition(
        self, global_src_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Get local chunk of global source node features for each rank corresponding
        to its rank in the process group across which the graph is partitioned.
        """
        if self.is_distributed:  # pragma: no cover
            return self.dist_graph.get_src_node_features_in_partition(global_src_feat)
        return global_src_feat

    def get_src_node_features_in_local_graph(
        self, local_src_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Get all source node features on all ranks from all other ranks which are requires
        for the neighborhood definition in the local graph. ``local_src_feat`` here
        corresponds to the local chunk of the global source node features on each rank
        corresponding to its rank in the process group across which the graph is partitioned.
        After this primitive, any message passing routine should have all necessary tensors
        to work on the corresponding local graph according to the partition rank.
        """
        if self.is_distributed:  # pragma: no cover
            return self.dist_graph.get_src_node_features_in_local_graph(local_src_feat)
        return local_src_feat

    def get_dst_node_features_in_partition(
        self, global_dst_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Get local chunk of global destination node features for each rank corresponding
        to its rank in the process group across which the graph is partitioned.
        """
        if self.is_distributed:  # pragma: no cover
            return self.dist_graph.get_dst_node_features_in_partition(global_dst_feat)
        return global_dst_feat

    def get_edge_features_in_partition(
        self, global_efeat: torch.Tensor
    ) -> torch.Tensor:
        """
        Get local chunk of global edge features for each rank corresponding
        to its rank in the process group across which the graph is partitioned.
        """
        if self.is_distributed:  # pragma: no cover
            return self.dist_graph.get_edge_features_in_partition(global_efeat)
        return global_efeat

    def get_global_src_node_features(self, local_nfeat: torch.Tensor) -> torch.Tensor:
        """
        Based on local source node features on each rank corresponding
        to its rank in the process group across which the graph is partitioned,
        get the global node features either on all group ranks or on group rank 0.
        """
        if self.is_distributed:  # pragma: no cover
            return self.dist_graph.get_global_src_node_features(local_nfeat)
        return local_nfeat

    def get_global_dst_node_features(self, local_nfeat: torch.Tensor) -> torch.Tensor:
        """
        Based on local destination node features on each rank corresponding
        to its rank in the process group across which the graph is partitioned,
        get the global node features either on all group ranks or on group rank 0.
        """
        if self.is_distributed:  # pragma: no cover
            return self.dist_graph.get_global_dst_node_features(local_nfeat)
        return local_nfeat

    def get_global_edge_features(self, local_efeat: torch.Tensor) -> torch.Tensor:
        """
        Based on local edge features on each rank corresponding
        to its rank in the process group across which the graph is partitioned,
        get the global edge features either on all group ranks or on group rank 0.
        """
        if self.is_distributed:  # pragma: no cover
            return self.dist_graph.get_global_edge_features(local_efeat)
        return local_efeat

    def to_dgl_graph(self) -> DGLGraph:  # pragma: no cover
        """Converts the graph to a DGLGraph.
        This can be useful if e.g. one wants to operate on a distributed
        graph in Modulus which assumes a simple CSC structure, but
        has only implemented GNN primitives in a DGL backend.

        Returns
        -------
        DGLGraph
            The DGLGraph created from the given object in CSC format.
        """

        if self.dgl_graph is None or not self.cache_graph:
            if self.ef_indices is not None:
                raise AssertionError("ef_indices is not supported.")
            graph_offsets = self.offsets
            dst_degree = graph_offsets[1:] - graph_offsets[:-1]
            src_indices = self.indices
            dst_indices = torch.arange(
                0,
                graph_offsets.size(0) - 1,
                dtype=graph_offsets.dtype,
                device=graph_offsets.device,
            )
            dst_indices = torch.repeat_interleave(dst_indices, dst_degree, dim=0)

            # labels not important here
            self.dgl_graph = dgl.heterograph(
                {("src", "src2dst", "dst"): ("coo", (src_indices, dst_indices))},
                idtype=torch.int32,
            )

            if not self.is_distributed:
                self.dgl_graph = self.dgl_graph.to(
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )

        return self.dgl_graph
