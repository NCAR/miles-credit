# This file includes code adapted from NVIDIA's PhysicsNemo project:
# https://github.com/NVIDIA/physicsnemo


from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist

from .autograd import (
    all_gather_v,
    indexed_all_to_all_v,
)


def partition_graph_by_coordinate_bbox(  # Assigns each node (scr or dst) to a rank
    global_offsets,
    global_indices,
    src_coordinates,
    dst_coordinates,
    coordinate_separators_min,
    coordinate_separators_max,
    partition_size,
    partition_rank,
    device,
):
    """
    Parameters
    ----------
    global_offsets : torch.Tensor
        CSC offsets, can live on the CPU
    global_indices : torch.Tensor
        CSC indices, can live on the CPU
    src_coordinates : torch.Tensor
        coordinates of each source node
    dst_coordinates : torch.Tensor
        coordinates of each destination node
    partition_size : int
        number of process groups across which graph is partitioned,
        i.e. the number of graph partitions
    partition_rank : int
        rank within process group managing the distributed graph, i.e.
        the rank determining which partition the corresponding local rank
        will manage
    device : torch.device
        device connected to the passed partition rank, i.e. the device
        on which the local graph and related buffers will live on
    """

    dim = src_coordinates.size(-1)
    if dst_coordinates.size(-1) != dim:
        raise ValueError()
    if len(coordinate_separators_min) != partition_size:
        a, b = len(coordinate_separators_min), partition_size
        error_msg = "Expected len(coordinate_separators_min) == partition_size"
        error_msg += f", but got {a} and {b} respectively"
        raise ValueError(error_msg)
    if len(coordinate_separators_max) != partition_size:
        a, b = len(coordinate_separators_max), partition_size
        error_msg = "Expected len(coordinate_separators_max) == partition_size"
        error_msg += f", but got {a} and {b} respectively"
        raise ValueError(error_msg)

    for rank in range(partition_size):
        if len(coordinate_separators_min[rank]) != dim:
            a, b = len(coordinate_separators_min[rank]), dim
            error_msg = f"Expected len(coordinate_separators_min[{rank}]) == dim"
            error_msg += f", but got {a} and {b} respectively"
        if len(coordinate_separators_max[rank]) != dim:
            a, b = len(coordinate_separators_max[rank]), dim
            error_msg = f"Expected len(coordinate_separators_max[{rank}]) == dim"
            error_msg += f", but got {a} and {b} respectively"

    num_global_src_nodes = global_indices.max().item() + 1
    num_global_dst_nodes = global_offsets.size(0) - 1

    mapping_dst_ids_to_ranks = torch.zeros(  # Tells where dst is placed
        num_global_dst_nodes, dtype=global_offsets.dtype, device=global_offsets.device
    )
    mapping_src_ids_to_ranks = torch.zeros(  # Tells where src is placed
        num_global_src_nodes,
        dtype=global_offsets.dtype,
        device=global_offsets.device,
    )

    def _assign_ranks(mapping, coordinates):
        for p in range(partition_size):
            mask = torch.ones_like(mapping).to(dtype=torch.bool)
            for d in range(dim):
                min_val, max_val = (
                    coordinate_separators_min[p][d],
                    coordinate_separators_max[p][d],
                )
                if min_val is not None:
                    mask = mask & (coordinates[:, d] >= min_val)
                if max_val is not None:
                    mask = mask & (coordinates[:, d] < max_val)
            mapping[mask] = p  # Maps every dst and src node to its rank

    _assign_ranks(mapping_src_ids_to_ranks, src_coordinates)
    _assign_ranks(mapping_dst_ids_to_ranks, dst_coordinates)

    return partition_graph_with_id_mapping(
        global_offsets,
        global_indices,
        mapping_src_ids_to_ranks,
        mapping_dst_ids_to_ranks,
        partition_size,
        partition_rank,
        device,
    )


def partition_graph_with_id_mapping(
    global_offsets,
    global_indices,
    mapping_src_ids_to_ranks,
    mapping_dst_ids_to_ranks,
    partition_size,
    partition_rank,
    device,
):
    """
    Utility function which partitions a global graph given as CSC structure.
    It partitions both the global ID spaces for source nodes and destination nodes
    based on the corresponding mappings as well as the graph structure and edge IDs.
    For more details on partitioning in general see `GraphPartition`.
    The function performs the partitioning based on a global graph in CPU
    memory for each rank independently. It could be rewritten to e.g. only
    do it one rank and exchange the partitions or to an algorithm that also
    assumes an already distributed global graph, however, we expect global
    graphs to fit in CPU memory. After the partitioning, we can get rid off
    the larger one in CPU memory, only keep the local graphs on each GPU, and
    avoid tedious gather/scatter routines for exchanging partitions in the process.
    Note: It is up to the user to ensure that the provided mapping is valid. In particular,
    we expect each rank to receive a non-empty partition of node IDs.

    Parameters
    ----------
    global_offsets : torch.Tensor
        CSC offsets, can live on the CPU
    global_indices : torch.Tensor
        CSC indices, can live on the CPU
    mapping_src_ids_to_ranks: torch.Tensor
        maps each global ID from every source node to its partition rank
    mapping_dst_ids_to_ranks: torch.Tensor
        maps each global ID from every destination node to its partition rank
    partition_size : int
        number of process groups across which graph is partitioned,
        i.e. the number of graph partitions
    partition_rank : int
        rank within process group managing the distributed graph, i.e.
        the rank determining which partition the corresponding local rank
        will manage
    device : torch.device
        device connected to the passed partition rank, i.e. the device
        on which the local graph and related buffers will live on
    """

    # initialize graph partition
    graph_partition = GraphPartition(
        partition_size=partition_size, partition_rank=partition_rank, device=device
    )

    # --------------------------------------------------------------
    # initialize temporary variables used in computing the partition

    # global IDs of in each partition
    dst_nodes_in_each_partition = [None] * partition_size
    src_nodes_in_each_partition = [None] * partition_size
    num_dst_nodes_in_each_partition = [None] * partition_size
    num_src_nodes_in_each_partition = [None] * partition_size

    dtype = global_indices.dtype
    input_device = global_indices.device

    graph_partition.map_concatenated_local_src_ids_to_global = torch.empty_like(
        mapping_src_ids_to_ranks
    )
    graph_partition.map_concatenated_local_dst_ids_to_global = torch.empty_like(
        mapping_dst_ids_to_ranks
    )
    graph_partition.map_concatenated_local_edge_ids_to_global = torch.empty_like(
        global_indices
    )
    graph_partition.map_global_src_ids_to_concatenated_local = torch.empty_like(
        mapping_src_ids_to_ranks
    )
    graph_partition.map_global_dst_ids_to_concatenated_local = torch.empty_like(
        mapping_dst_ids_to_ranks
    )
    graph_partition.map_global_edge_ids_to_concatenated_local = torch.empty_like(
        global_indices
    )
    _map_global_src_ids_to_local = torch.empty_like(mapping_src_ids_to_ranks)

    # temporarily track cum-sum of nodes per partition for "concatenated_local_ids"
    _src_id_offset = 0
    _dst_id_offset = 0
    _edge_id_offset = 0

    # Find dst and scr in every rank and assign them local IDS.
    for rank in range(partition_size):
        dst_nodes_in_each_partition[rank] = torch.nonzero(
            mapping_dst_ids_to_ranks == rank
        ).view(-1)
        src_nodes_in_each_partition[rank] = torch.nonzero(
            mapping_src_ids_to_ranks == rank
        ).view(-1)
        num_nodes = dst_nodes_in_each_partition[rank].numel()
        if num_nodes == 0:
            raise RuntimeError(
                f"Aborting partitioning, rank {rank} has 0 destination nodes to work on."
            )
        num_dst_nodes_in_each_partition[rank] = num_nodes

        num_nodes = src_nodes_in_each_partition[rank].numel()
        num_src_nodes_in_each_partition[rank] = num_nodes
        if num_nodes == 0:
            raise RuntimeError(
                f"Aborting partitioning, rank {rank} has 0 source nodes to work on."
            )

        # create mapping of global node IDs to/from "concatenated local" IDs
        ids = src_nodes_in_each_partition[rank]
        mapped_ids = torch.arange(
            start=_src_id_offset,
            end=_src_id_offset + ids.numel(),
            dtype=dtype,
            device=input_device,
        )
        _map_global_src_ids_to_local[ids] = (
            mapped_ids - _src_id_offset
        )  # Assign local ids to src
        graph_partition.map_global_src_ids_to_concatenated_local[ids] = (
            mapped_ids  # Re-arranges/re-assigns global nodes such that lower ranks/partitions have smaller indices (in this case ids in the same ranks are consecutive)
        )
        graph_partition.map_concatenated_local_src_ids_to_global[mapped_ids] = ids
        _src_id_offset += ids.numel()

        ids = dst_nodes_in_each_partition[rank]
        mapped_ids = torch.arange(
            start=_dst_id_offset,
            end=_dst_id_offset + ids.numel(),
            dtype=dtype,
            device=input_device,
        )
        graph_partition.map_global_dst_ids_to_concatenated_local[ids] = mapped_ids
        graph_partition.map_concatenated_local_dst_ids_to_global[mapped_ids] = ids
        _dst_id_offset += ids.numel()

    graph_partition.num_src_nodes_in_each_partition = num_src_nodes_in_each_partition
    graph_partition.num_dst_nodes_in_each_partition = num_dst_nodes_in_each_partition

    # create local graph structures # Start looking into CSC graph for partitioning
    for rank in range(partition_size):
        offset_start = global_offsets[dst_nodes_in_each_partition[rank]].view(-1)
        offset_end = global_offsets[dst_nodes_in_each_partition[rank] + 1].view(-1)
        degree = offset_end - offset_start
        local_offsets = degree.view(-1).cumsum(
            dim=0
        )  # Create local offset by looking into cumulative degrees of dst
        local_offsets = torch.cat(
            [
                torch.Tensor([0]).to(  # Add 0 at the beginning of the offset array
                    dtype=dtype,
                    device=input_device,
                ),
                local_offsets,
            ]
        )

        partitioned_edge_ids = torch.cat(
            [
                torch.arange(
                    start=offset_start[i],
                    end=offset_end[i],
                    dtype=dtype,
                    device=input_device,
                )
                for i in range(len(offset_start))
            ]
        )

        ids = partitioned_edge_ids
        mapped_ids = torch.arange(
            _edge_id_offset,
            _edge_id_offset + ids.numel(),
            device=ids.device,
            dtype=ids.dtype,
        )
        graph_partition.map_global_edge_ids_to_concatenated_local[ids] = (
            mapped_ids  # same as with nodes
        )
        graph_partition.map_concatenated_local_edge_ids_to_global[mapped_ids] = ids
        _edge_id_offset += ids.numel()

        partitioned_src_ids = torch.cat(  # Retrieve all the src nodes based on the indices so that each dst node has its full neighborhood
            [
                global_indices[offset_start[i] : offset_end[i]].clone()
                for i in range(len(offset_start))
            ]
        )

        global_src_ids_on_rank, inverse_mapping = partitioned_src_ids.unique(
            sorted=True, return_inverse=True
        )
        remote_local_src_ids_on_rank = _map_global_src_ids_to_local[
            global_src_ids_on_rank
        ]

        _map_global_src_ids_to_local_graph = torch.zeros_like(mapping_src_ids_to_ranks)
        _num_local_indices = 0
        for rank_offset in range(partition_size):
            mask = mapping_src_ids_to_ranks[global_src_ids_on_rank] == rank_offset
            if (
                partition_rank == rank_offset
            ):  # If the rank of src node in retrieved neighborhood is the same as the current rank (partition rank)
                # indices to send to this rank from this rank
                graph_partition.scatter_indices[
                    rank
                ] = (  # Let partition_rank know that it should be sending these ids to rank
                    remote_local_src_ids_on_rank[mask]
                    .detach()
                    .clone()
                    .to(dtype=torch.int64)
                )
            numel_mask = mask.sum().item()
            graph_partition.sizes[rank_offset][rank] = (
                numel_mask  # Number of nodes that partition_rank sends to other ranks
            )
            tmp_ids = torch.arange(
                _num_local_indices,
                _num_local_indices + numel_mask,
                device=input_device,
                dtype=dtype,
            )
            _num_local_indices += numel_mask
            tmp_map = global_src_ids_on_rank[mask]
            _map_global_src_ids_to_local_graph[tmp_map] = (
                tmp_ids  # This is where src indices (based on neighborhood) are re-assigned so that there is no conflict in numbering
            )

        local_indices = _map_global_src_ids_to_local_graph[partitioned_src_ids]
        graph_partition.num_indices_in_each_partition[rank] = local_indices.size(0)

        if rank == partition_rank:
            # local graph
            graph_partition.local_offsets = local_offsets
            graph_partition.local_indices = local_indices
            graph_partition.num_local_indices = graph_partition.local_indices.size(0)
            graph_partition.num_local_dst_nodes = num_dst_nodes_in_each_partition[rank]
            graph_partition.num_local_src_nodes = global_src_ids_on_rank.size(0)

            # partition-local mappings (local IDs to global)
            graph_partition.map_partitioned_src_ids_to_global = (
                src_nodes_in_each_partition[rank]
            )
            graph_partition.map_partitioned_dst_ids_to_global = (
                dst_nodes_in_each_partition[rank]
            )
            graph_partition.map_partitioned_edge_ids_to_global = partitioned_edge_ids

    for r in range(graph_partition.partition_size):
        err_msg = "error in graph partition: list containing sizes of exchanged indices does not match the tensor of indices to be exchanged"
        if (
            graph_partition.sizes[graph_partition.partition_rank][r]
            != graph_partition.scatter_indices[r].numel()
        ):
            raise AssertionError(err_msg)

    graph_partition = graph_partition.to(device=device)

    return graph_partition


@dataclass
class GraphPartition:
    """
    Class acting as an "utility" structure to hold all relevant buffers and variables
    to define a graph partition and faciliate exchange of necessary buffers for
    message passing on a distributed graph.

    A global graph is assumed to be defined through a global CSC structure
    defining edges between source nodes and destination nodes which are assumed
    to be numbered indexed by contiguous IDs. Hence, features associated to both
    nodes and edges can be represented through dense feature tables globally.
    When partitioning graph and features, we distribute destination nodes and all
    their incoming edges on all ranks within the partition group based on a specified
    mapping. Based on this scheme, there will a be a difference between
    partitioned source nodes (partitioned features) and local source node
    IDs which refer to the node IDs within the local graph defined by the
    destination nodes on each rank. To allow message passing, communication
    primitives have to ensure to gather all corresponding features for all
    local source nodes based on the applied partitioning scheme. This also
    leads to the distinction of local source node IDs and remote source node
    IDs on each rank where the latter simply refers to the local row ID within
    the dense partitioning of node features and the former indicates the source
    of a message for each edge within each local graph.

    Parameters
    ----------
    partition_size : int
        size of partition
    partition_rank : int
        local rank of this partition w.r.t. group of partitions
    device : torch.device
        device handle for buffers within this partition rank
    """

    partition_size: int
    partition_rank: int
    device: torch.device
    # flag to indicate using adj matrix 1-D row-decomp
    # matrix_decomp: bool = False

    # data structures defining partition
    # set in after initialization or during execution
    # of desired partition scheme

    # local CSR offsets defining local graph on each `partition_rank`
    local_offsets: Optional[torch.Tensor] = None
    # local CSR indices defining local graph on each `partition_rank`
    local_indices: Optional[torch.Tensor] = None
    # number of source nodes in local graph on each `partition_rank`
    num_local_src_nodes: int = -1
    # number of destination nodes in local graph on each `partition_rank`
    num_local_dst_nodes: int = -1
    # number of edges in local graph on each `partition_rank`
    num_local_indices: int = -1
    # mapping from global to local ID space (source node IDs)
    map_partitioned_src_ids_to_global: Optional[torch.Tensor] = None
    map_concatenated_local_src_ids_to_global: Optional[torch.Tensor] = None
    # mapping from local to global ID space (destination node IDs)
    map_partitioned_dst_ids_to_global: Optional[torch.Tensor] = None
    map_concatenated_local_dst_ids_to_global: Optional[torch.Tensor] = None
    # mapping from local to global ID space (edge IDs)
    map_partitioned_edge_ids_to_global: Optional[torch.Tensor] = None
    map_concatenated_local_edge_ids_to_global: Optional[torch.Tensor] = None
    # reverse mappings
    map_global_src_ids_to_concatenated_local: Optional[torch.Tensor] = None
    map_global_dst_ids_to_concatenated_local: Optional[torch.Tensor] = None
    map_global_edge_ids_to_concatenated_local: Optional[torch.Tensor] = None

    # utility lists and sizes required for exchange of messages
    # between graph partitions through distributed communication primitives

    # number of IDs each rank potentially sends to all other ranks
    sizes: Optional[List[List[int]]] = None
    # local indices of IDs current rank sends to all other ranks
    scatter_indices: Optional[List[torch.Tensor]] = None
    # number of global source nodes for each `partition_rank`
    num_src_nodes_in_each_partition: Optional[List[int]] = None
    # number of global destination nodes for each `partition_rank`
    num_dst_nodes_in_each_partition: Optional[List[int]] = None
    # number of global indices for each `partition_rank`
    num_indices_in_each_partition: Optional[List[int]] = None

    def __post_init__(self):
        # after partition_size has been set in __init__
        if self.partition_size <= 0:
            raise ValueError(f"Expected partition_size > 0, got {self.partition_size}")
        if not (0 <= self.partition_rank < self.partition_size):
            raise ValueError(
                f"Expected 0 <= partition_rank < {self.partition_size}, got {self.partiton_rank}"
            )

        if self.sizes is None:
            self.sizes = [
                [None for _ in range(self.partition_size)]
                for _ in range(self.partition_size)
            ]

        if self.scatter_indices is None:
            self.scatter_indices = [None] * self.partition_size
        if self.num_src_nodes_in_each_partition is None:
            self.num_src_nodes_in_each_partition = [None] * self.partition_size
        if self.num_dst_nodes_in_each_partition is None:
            self.num_dst_nodes_in_each_partition = [None] * self.partition_size
        if self.num_indices_in_each_partition is None:
            self.num_indices_in_each_partition = [None] * self.partition_size

    def to(self, *args, **kwargs):
        # move all tensors
        for attr in dir(self):
            attr_val = getattr(self, attr)
            if isinstance(attr_val, torch.Tensor):
                setattr(self, attr, attr_val.to(*args, **kwargs))

        # handle scatter_indices separately
        self.scatter_indices = [idx.to(*args, **kwargs) for idx in self.scatter_indices]

        return self


class DistributedGraph:
    def __init__(
        self,
        # global_offsets: torch.Tensor,
        # global_indices: torch.Tensor,
        partition_size: int,
        graph_partition_group_name: str = None,
        graph_partition: Optional[GraphPartition] = None,
    ):  # pragma: no cover
        """
        Utility Class representing a distributed graph based on a given
        partitioning of a CSC graph structure. By default, a naive node-wise
        partitioning scheme is applied, see ``partition_graph_nodewise`` for
        details on that. This class then wraps necessary communication primitives
        to access all relevant feature buffers related to the graph.

        Parameters
        ----------
        global_offsets : torch.Tensor
            CSC offsets, can live on the CPU
        global_indices : torch.Tensor
            CSC indices, can live on the CPU
        partition_size : int
            Number of process groups across which graphs are distributed, expected to
            be larger than 1, i.e. an actual partition distributed among multiple ranks.
        partition_group_name : str, default=None
            Name of process group across which graphs are distributed. Passing no process
            group name leads to a parallelism across the default process group.
            Otherwise, the group size of a process group is expected to match partition_size.
        graph_partition : GraphPartition, optional
            Optional graph_partition, if passed as None, the naive
            node-wise partitioning scheme will be applied to global_offsets and global_indices,
            otherwise, these will be ignored and the passed partition will be used instead.
        """

        self.partition_rank = dist.get_rank()
        self.partition_size = partition_size
        self.device = torch.device(
            f"cuda:{self.partition_rank % torch.cuda.device_count()}"
            if torch.cuda.is_available()
            else "cpu"
        )

        assert graph_partition is not None, "graph_partition should be provided"
        error_msg = f"Passed graph_partition.partition_size does not correspond to size of process_group, got {graph_partition.partition_size} and {self.partition_size} respectively."
        if graph_partition.partition_size != self.partition_size:
            raise AssertionError(error_msg)
        error_msg = f"Passed graph_partition.device does not correspond to device of this rank, got {graph_partition.device} and {self.device} respectively."
        if graph_partition.device != self.device:
            raise AssertionError(error_msg)
        self.graph_partition = graph_partition
        self.total_num_src_nodes_in_whole_graph = sum(
            graph_partition.num_src_nodes_in_each_partition
        )
        self.total_num_dst_nodes_in_whole_graph = sum(
            graph_partition.num_dst_nodes_in_each_partition
        )
        self.num_edges_in_whole_graph = sum(
            graph_partition.num_indices_in_each_partition
        )
        self.num_src_nodes_in_this_partition = (
            self.graph_partition.num_src_nodes_in_each_partition[
                self.graph_partition.partition_rank
            ]
        )
        self.num_dst_nodes_in_this_partition = (
            self.graph_partition.num_dst_nodes_in_each_partition[
                self.graph_partition.partition_rank
            ]
        )
        self.num_edges_in_this_partition = (
            self.graph_partition.num_indices_in_each_partition[
                self.graph_partition.partition_rank
            ]
        )

        send_sizes = self.graph_partition.sizes[self.graph_partition.partition_rank]
        recv_sizes = [
            p[self.graph_partition.partition_rank] for p in self.graph_partition.sizes
        ]
        msg = f"GraphPartition(rank={self.graph_partition.partition_rank}, "
        msg += f"num_local_src_nodes={self.graph_partition.num_local_src_nodes}, "
        msg += f"num_local_dst_nodes={self.graph_partition.num_local_dst_nodes}, "
        msg += f"num_partitioned_src_nodes={self.graph_partition.num_src_nodes_in_each_partition[self.graph_partition.partition_rank]}, "
        msg += f"num_partitioned_dst_nodes={self.graph_partition.num_dst_nodes_in_each_partition[self.graph_partition.partition_rank]}, "
        msg += f"num_partitioned_edges={self.graph_partition.num_indices_in_each_partition[self.graph_partition.partition_rank]}, "
        msg += f"send_sizes={send_sizes}, recv_sizes={recv_sizes})"
        print(msg)

        # dist.barrier(self.process_group)
        dist.barrier()

    def get_src_node_features_in_partition(
        self, global_node_features: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover
        assert len(global_node_features) == self.total_num_src_nodes_in_whole_graph
        return global_node_features.to(device=self.device)[
            self.graph_partition.map_partitioned_src_ids_to_global, :
        ]

    def get_src_node_features_in_local_graph(
        self, partitioned_src_node_features: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover
        # main primitive to gather all necessary src features
        # which are required for a csc-based message passing step
        assert (
            len(partitioned_src_node_features) == self.num_src_nodes_in_this_partition
        )
        return indexed_all_to_all_v(
            partitioned_src_node_features,
            indices=self.graph_partition.scatter_indices,
            sizes=self.graph_partition.sizes,
            use_fp32=True,
            dim=0,
        )

    def get_dst_node_features_in_partition(
        self, global_node_features: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover
        assert len(global_node_features) == self.total_num_dst_nodes_in_whole_graph
        return global_node_features.to(device=self.device)[
            self.graph_partition.map_partitioned_dst_ids_to_global, :
        ]

    def get_dst_node_features_in_local_graph(
        self,
        partitioned_dst_node_features: torch.Tensor,
    ) -> torch.Tensor:  # pragma: no cover
        # current partitioning scheme assumes that
        # local graph is built from partitioned IDs
        assert (
            len(partitioned_dst_node_features) == self.num_dst_nodes_in_this_partition
        )
        return partitioned_dst_node_features

    def get_edge_features_in_partition(
        self, global_edge_features: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover
        assert len(global_edge_features) == self.num_edges_in_whole_graph
        return global_edge_features.to(device=self.device)[
            self.graph_partition.map_partitioned_edge_ids_to_global, :
        ]

    def get_edge_features_in_local_graph(
        self, partitioned_edge_features: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover
        # current partitioning scheme assumes that
        # local graph is built from partitioned IDs
        assert len(partitioned_edge_features) == self.num_edges_in_this_partition
        return partitioned_edge_features

    def get_global_src_node_features(
        self, partitioned_node_features: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover
        assert len(partitioned_node_features) == self.num_src_nodes_in_this_partition
        global_node_feat = all_gather_v(
            partitioned_node_features,
            self.graph_partition.num_src_nodes_in_each_partition,
            dim=0,
            use_fp32=True,
        )
        global_node_feat = global_node_feat[
            self.graph_partition.map_global_src_ids_to_concatenated_local
        ]
        return global_node_feat

    def get_global_dst_node_features(
        self, partitioned_node_features: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover
        assert len(partitioned_node_features) == self.num_dst_nodes_in_this_partition
        global_node_feat = all_gather_v(
            partitioned_node_features,
            self.graph_partition.num_dst_nodes_in_each_partition,
            dim=0,
            use_fp32=True,
        )
        global_node_feat = global_node_feat[
            self.graph_partition.map_global_dst_ids_to_concatenated_local
        ]
        return global_node_feat

    def get_global_edge_features(
        self, partitioned_edge_features: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover
        assert len(partitioned_edge_features) == self.num_edges_in_this_partition
        global_edge_feat = all_gather_v(
            partitioned_edge_features,
            self.graph_partition.num_indices_in_each_partition,
            dim=0,
            use_fp32=True,
        )
        global_edge_feat = global_edge_feat[
            self.graph_partition.map_global_edge_ids_to_concatenated_local
        ]
        return global_edge_feat
