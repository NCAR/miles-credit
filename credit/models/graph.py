import os

os.environ["DGLBACKEND"] = "pytorch"
import logging
import math

import dgl
import dgl.function as fn
import torch
import torch.distributed as dist
import xarray as xr
from dgl.nn.functional import edge_softmax
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.utils import parametrizations
from torch.utils.checkpoint import checkpoint

from credit.models.base_model import BaseModel

from .graph_partition import CuGraphCSC, get_lat_lon_partition_separators

logger = logging.getLogger(__name__)


def apply_spectral_norm(model):
    for module in model.modules():
        if isinstance(module, (nn.Linear)):
            parametrizations.spectral_norm(module)


def load_graph(edge_path, hierarchy, partition_size, min_seps, max_seps, reverse):
    """Loads and processes a edge data from an xarray NetCDF file,
    creating a DGL graph with edge and node spatial features.

    Parameters
    ----------
    edge_path : str or pathlike object
        Path to a NetCDF file containing graph edge data in xarray format.
    hierarchy : bool
        If True, expects 'source' and 'target prefixes on every node feature.
        E.g., 'source_latitude' instead 'latitude';
    partition_size : int
        Number of partitions to divide the graph into.
    min_seps : int
        Minimum number of spatial separators (bounding boxes) for partitioning.
    max_seps : int
        Maximum number of spatial separators (bounding boxes) for partitioning.
    reverse : bool
        If True, reverses edge direction and adjusts angular differences accordingly.

    Returns
    -------
    dict
        A dictionary containing:
        - 'graph' : dgl.DGLGraph
            The constructed partitioned DGL graph.
        - 'edata' : torch.Tensor
            Edge features consisting of directional cos/sin differences and RBF-weighted distance.
        - 'src_lat_lon' : torch.Tensor
            Processed source node angular embedding (sin/cos of latitude and longitude).
        - 'dst_lat_lon' : torch.Tensor
            Processed destination node angular embedding (sin/cos of latitude and longitude).
    """
    xr_dataset = xr.open_dataset(edge_path)
    edge_index = torch.from_numpy(xr_dataset.edges.values.T)
    if not hierarchy:
        mask = ~(edge_index[0] == edge_index[1])
        edge_index = edge_index[:, mask]  # Remove self loops
    edata = None

    src_lat = (
        torch.from_numpy(xr_dataset.source_latitude.values).float()
        if hierarchy
        else torch.from_numpy(xr_dataset.latitude.values).float()
    )
    src_lon = (
        torch.from_numpy(xr_dataset.source_longitude.values).float()
        if hierarchy
        else torch.from_numpy(xr_dataset.longitude.values).float()
    )
    dst_lat = (
        torch.from_numpy(xr_dataset.target_latitude.values).float()
        if hierarchy
        else src_lat
    )
    dst_lon = (
        torch.from_numpy(xr_dataset.target_longitude.values).float()
        if hierarchy
        else src_lon
    )

    edge_dist = torch.from_numpy(xr_dataset.distances.values).unsqueeze(1).float()

    if not hierarchy:
        edge_dist = edge_dist[mask]  # Removing self loops

    edge_dist = edge_dist / edge_dist.max()

    src_lat_lon = torch.stack([src_lat, src_lon], dim=-1)
    dst_lat_lon = torch.stack([dst_lat, dst_lon], dim=-1)

    rad_lat_lon_diff = torch.deg2rad(
        src_lat_lon[edge_index[0]] - dst_lat_lon[edge_index[1]]
    )

    assert edge_index.shape[1] == edge_dist.shape[0] == rad_lat_lon_diff.shape[0], (
        f"{edge_index.shape=:} {edge_dist.shape=:} {rad_lat_lon_diff.shape=:}"
    )

    if reverse:
        edge_index = edge_index[[1, 0]]
        rad_lat_lon_diff = -rad_lat_lon_diff
        src_lat, dst_lat = dst_lat, src_lat
        src_lon, dst_lon = dst_lon, src_lon
        src_lat_lon, dst_lat_lon = dst_lat_lon, src_lat_lon

    edata = torch.cat(
        [
            torch.cos(rad_lat_lon_diff),
            torch.sin(rad_lat_lon_diff),
            torch.exp(-((edge_dist) ** 2)),
        ],
        dim=-1,
    )
    assert edata.shape[1] == 5

    xr_dataset.close()

    if hierarchy:
        graph = dgl.heterograph(
            {("src", "src2dst", "dst"): tuple(item.long() for item in edge_index)},
            num_nodes_dict={"src": len(src_lat), "dst": len(dst_lat)},
        )
    else:
        graph = dgl.graph(
            tuple(item.long() for item in edge_index), num_nodes=len(src_lat)
        )

    src_data = torch.deg2rad(src_lat_lon)
    dst_data = torch.deg2rad(dst_lat_lon)
    src_data = torch.cat([torch.sin(src_data), torch.cos(src_data)], dim=-1)
    dst_data = torch.cat([torch.sin(dst_data), torch.cos(dst_data)], dim=-1)
    kwargs = {
        "src_coordinates": src_lat_lon,
        "dst_coordinates": dst_lat_lon,
        "coordinate_separators_min": min_seps,
        "coordinate_separators_max": max_seps,
    }
    graph, edge_perm = CuGraphCSC.from_dgl(
        graph=graph,
        partition_size=partition_size,
        partition_by_bbox=True,
        **kwargs,
    )
    edata = edata[edge_perm]
    edata = graph.get_edge_features_in_partition(edata)
    src_data = graph.get_src_node_features_in_partition(src_data)
    dst_data = graph.get_dst_node_features_in_partition(dst_data)

    return {
        "graph": graph,
        "edata": edata,
        "src_lat_lon": src_data,
        "dst_lat_lon": dst_data,
    }


class GraphGRUNet(BaseModel):
    """
    This model constructs a hierarchical bidirectional graph architecture using DGL, enabling fine-to-coarse
    and coarse-to-fine message passing. It incorporates configurable downsampling blocks and processor layers.
    It enables distributed training/inference by paritioning the graph representing the globe.

    Parameters
    ----------
    n_variables : int
        Number of upper-air atmospheric variables.
    n_surface_variables : int
        Number of surface-level variables.
    n_static_forcing_variables : int
        Number of static variables.
    levels : int
        Number of vertical levels.
    history_len : int
        Number of historical timesteps used for temporal context.
    down_hid_dims : list of int
        Hidden dimensions for each downsampling stage in the graph hierarchy.
    down_dim_head : list of int
        Attention head dimensions for each processor block. Should have the same size as `down_hid_dims`
    down_block_depth : list of int
        Number of layers per processor block.
    dropout : float
        Dropout probability used throughout the network.
    down_graph_path : list of str or None
        Paths to NetCDF graph edges.
    use_spectral_norm : bool
        Whether to apply spectral normalization to model weights.
    use_checkpoint : bool
        If True, uses checkpointing to save memory during training.
    world_size : int
        Number of partitions to divide the graph representing the globe.
    use_coords : bool
        Whether to include latitude-longitude embeddings in graph message passing.
    **kwargs : dict
        Additional arguments passed to the base class.
    """

    def __init__(
        self,
        n_variables=4,
        n_surface_variables=7,
        n_static_forcing_variables=3,
        levels=16,
        history_len=2,
        down_hid_dims=[128, 256, 512],
        down_dim_head=[32, 64, 128],
        down_block_depth=[2, 2, 2],
        dropout=0,
        down_graph_path=[None] * 6,
        use_spectral_norm=True,
        use_checkpoint=False,
        world_size=1,
        use_coords=True,
        **kwargs,
    ):
        super().__init__()
        assert len(down_hid_dims) == len(down_dim_head) == len(down_block_depth)
        assert len(down_block_depth) * 2 == len(down_graph_path)
        self.partition_size = world_size

        min_seps, max_seps = None, None
        if self.partition_size > 1:
            min_seps, max_seps = get_lat_lon_partition_separators(self.partition_size)
        down_graph_chain = [
            load_graph(
                edge_path,
                partition_size=self.partition_size,
                min_seps=min_seps,
                max_seps=max_seps,
                hierarchy=(i % 2 == 0),
                reverse=False,
            )
            for i, edge_path in enumerate(down_graph_path)
        ]
        up_graph_path = down_graph_path[::-1][
            1:
        ]  # Reverse the down chain and skip the last block from down chain
        up_graph_chain = down_graph_chain[::-1][1:]
        self.all_graph_chain = down_graph_chain + [
            load_graph(
                up_graph_path[i],
                partition_size=self.partition_size,
                min_seps=min_seps,
                max_seps=max_seps,
                hierarchy=True,
                reverse=True,
            )
            if (i % 2 == 0)
            else up_graph_chain[i]
            for i in range(len(up_graph_path))
        ]

        assert len(self.all_graph_chain) + 1 == len(down_graph_path) * 2

        self.dropout = dropout
        self.n_variables = n_variables
        self.n_static_forcing_variables = n_static_forcing_variables
        self.n_surface_variables = n_surface_variables
        self.histroy_len = history_len
        self.n_levels = levels
        self.state_vars = self.n_variables * self.n_levels + self.n_surface_variables
        self.total_n_vars = (
            self.state_vars + self.n_static_forcing_variables
        ) * self.histroy_len

        all_dims = down_hid_dims + down_hid_dims[::-1][1:]
        all_block_depths = down_block_depth + down_block_depth[::-1][1:]
        all_head_dims = down_dim_head + down_dim_head[::-1][1:]

        self.first_graph = self.all_graph_chain[0]["graph"]
        self.last_graph = self.all_graph_chain[-1]["graph"]

        self.init_norm = GlobalLayerNorm(self.total_n_vars)
        self.lin_final = nn.Sequential(
            nn.Linear(self.state_vars, self.state_vars * 2),
            GlobalLayerNorm(self.state_vars * 2),
            nn.ELU(),
            nn.Linear(self.state_vars * 2, self.state_vars),
        )
        max_dim = max(down_hid_dims)
        self.lin_mean = nn.Sequential(
            nn.Linear(self.total_n_vars, max_dim, bias=False),
            nn.ELU(),
            nn.Linear(max_dim, max_dim, bias=False),
            nn.ELU(),
            nn.Linear(max_dim, self.state_vars, bias=False),
            nn.Tanh(),
        )
        self.lin_std = nn.Sequential(
            nn.Linear(self.total_n_vars, max_dim, bias=False),
            nn.ELU(),
            nn.Linear(max_dim, max_dim, bias=False),
            nn.ELU(),
            nn.Linear(max_dim, self.state_vars, bias=False),
            nn.Sigmoid(),
        )
        self.block_list = nn.ModuleList()
        for i, graph_dict in enumerate(self.all_graph_chain):
            idx = int(i // 2)
            prev_idx = idx - 1
            prev_dim = all_dims[prev_idx] if (prev_idx >= 0) else self.total_n_vars
            curr_dim = all_dims[idx] if (idx < len(all_dims)) else self.state_vars

            if i % 2 == 0:
                self.block_list.append(
                    GridToGridConv(
                        prev_dim,
                        curr_dim,
                        graph_dict["graph"],
                        graph_dict["edata"],
                        graph_dict["src_lat_lon"] if use_coords else None,
                        graph_dict["dst_lat_lon"] if use_coords else None,
                        dropout=self.dropout,
                    )
                )
            else:
                self.block_list.append(
                    ProcessorBlock(
                        curr_dim,
                        depth=all_block_depths[idx],
                        dim_head=all_head_dims[idx],
                        dropout=self.dropout,
                        graph=graph_dict["graph"],
                        edata=graph_dict["edata"],
                        ndata=graph_dict["dst_lat_lon"] if use_coords else None,
                    )
                )

        self.use_spectral_norm = use_spectral_norm
        self.use_checkpoint = use_checkpoint
        if self.use_spectral_norm:
            apply_spectral_norm(self)

    def forward(self, x):
        lat_lon_shape = x.shape[-2:]
        state = x[:, : self.state_vars, -1:]
        x = x.view(-1, self.total_n_vars, lat_lon_shape[0] * lat_lon_shape[1]).permute(
            2, 0, 1
        )
        mean_x = x.mean(0, keepdim=True) * 10  # Get mean and std before partitioning
        std_x = x.std(0, keepdim=True, unbiased=False) / 10
        x = self.init_norm(x)
        x = self.first_graph.get_src_node_features_in_partition(
            x
        )  # the first node features are considered sources (for downscaling)

        res_list = []
        for i in range(len(self.block_list) // 2 + 1):
            layer_i = self.block_list[i]
            if self.use_checkpoint:
                x = checkpoint(layer_i, x, use_reentrant=False)
            else:
                x = layer_i(x)
            # x = F.dropout(x, p=0.4, training=self.training)
            if i % 2 == 1:
                res_list.append(x)

        res_list = res_list[:-1]
        for i in range(len(self.block_list) // 2 + 1, len(self.block_list)):
            layer_i = self.block_list[i]
            if self.use_checkpoint:
                x = checkpoint(layer_i, x, use_reentrant=False)
            else:
                x = layer_i(x)
            # x = F.dropout(x, p=0.4, training=self.training)
            if i % 2 == 1:
                x = x + res_list[-1]
                res_list = res_list[:-1]

        x = self.last_graph.get_global_dst_node_features(x)
        x = self.lin_final(x)
        x = x * self.lin_std(std_x) + self.lin_mean(mean_x)
        x = x.permute(1, 2, 0)
        x = x.view(-1, self.state_vars, 1, *lat_lon_shape)

        assert x.shape == state.shape, f"{x.shape=:} {state.shape=:}"

        return x + state


class GridToGridConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        graph,
        edata,
        src_data,
        dst_data,
        dropout,
    ):
        super().__init__()

        self.graph = graph
        self.edata = edata
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.src_data = src_data
        self.dst_data = dst_data
        self.dropout = dropout

        if self.src_data is not None:
            self.lin_src = nn.Linear(self.src_data.shape[-1], self.in_feats)
        else:
            self.register_parameter("lin_src", None)

        self.lin_node = MultiLayer(self.in_feats, self.in_feats, self.in_feats, 2)
        self.lin_edge = MultiLayer(self.edata.shape[1], self.in_feats, 1, 2)
        self.lin_output = nn.Sequential(
            LayerNorm(self.in_feats), nn.ELU(), nn.Linear(self.in_feats, self.out_feats)
        )

    def forward(self, x):
        device = x.device
        src_feats = self.lin_node(x)
        if self.lin_src is not None:
            src_feats = src_feats + self.lin_src(self.src_data.to(device)).unsqueeze(1)
        dgl_graph = self.graph.to_dgl_graph()
        src_feats = self.graph.get_src_node_features_in_local_graph(src_feats)
        edata = self.lin_edge(self.edata.to(device))

        with dgl_graph.local_scope():
            attn = edge_softmax(dgl_graph, edata)
            dgl_graph.edata["attn"] = attn.unsqueeze(1)
            dgl_graph.srcdata["src_feats"] = src_feats
            dgl_graph.update_all(
                fn.u_mul_e("src_feats", "attn", "a_u"), fn.sum("a_u", "new_feats")
            )
            new_feats = dgl_graph.dstdata["new_feats"]

        new_feats = self.lin_output(new_feats)
        return new_feats


class ProcessorBlock(nn.Module):
    def __init__(
        self,
        hid_dim,
        depth,
        dim_head=32,
        dropout=0,
        graph=None,
        edata=None,
        ndata=None,
    ):
        super().__init__()
        self.hid_dim = hid_dim
        self.dim_head = dim_head
        assert hid_dim % dim_head == 0
        self.heads = hid_dim // dim_head
        self.dropout = dropout
        self.depth = depth
        self.edata = edata
        self.graph = graph

        self.ndata = ndata
        if ndata is not None:
            self.lin_ndata = nn.Linear(self.ndata.shape[-1], hid_dim)
        else:
            self.register_parameter("lin_ndata", None)

        self.graph_layers = nn.ModuleList(
            TransformerConv(
                self.hid_dim,
                self.hid_dim // self.heads,
                heads=self.heads,
                dropout=self.dropout,
                edge_dim=self.edata.shape[1] if self.edata is not None else None,
            )
            for _ in range(self.depth)
        )
        self.gated_unit = GateCell(self.hid_dim)

        self.lin_out = MultiLayer(self.hid_dim, self.hid_dim, self.hid_dim, 1)

        self.graph_norm_layers = nn.ModuleList(
            nn.Sequential(
                LayerNorm(self.hid_dim),
                nn.ELU(),
                nn.Linear(self.hid_dim, self.hid_dim),
            )
            for _ in range(self.depth - 1)
        )

    def forward(self, x):
        device = x.device
        x_skip = x
        h = None
        if self.lin_ndata is not None:
            x = x + self.lin_ndata(self.ndata.to(device)).unsqueeze(1)
        for i, graph_transf in enumerate(self.graph_layers):
            x = graph_transf(x, self.graph, edata=self.edata.to(device))
            h = self.gated_unit(x, h)
            # x = F.dropout(x, p=0.4, training=self.training)
            if i < len(self.graph_norm_layers):
                x = self.graph_norm_layers[i](x)

        x = h
        x = self.lin_out(x)
        x = x + x_skip
        x = F.elu(x)

        return x


class GateCell(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.z_x_ln = nn.Linear(self.hidden_size, self.hidden_size)
        self.z_h_ln = nn.Linear(self.hidden_size, self.hidden_size)
        self.r_x_ln = nn.Linear(self.hidden_size, self.hidden_size)
        self.r_h_ln = nn.Linear(self.hidden_size, self.hidden_size)
        self.h_x_ln = nn.Linear(self.hidden_size, self.hidden_size)
        self.h_h_ln = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x, h):
        z = self.z_x_ln(x)
        r = self.r_x_ln(x)
        if h is not None:
            z = z + self.z_h_ln(h)
            r = r + self.r_h_ln(h)

        z = F.sigmoid(z)
        r = F.sigmoid(r)

        h_hat = self.h_x_ln(x)

        if h is not None:
            h_hat = self.h_h_ln(r * h)

        h_hat = torch.tanh(h_hat)

        h = h_hat if h is None else (1 - z) * h + z * h_hat

        return h


class TransformerConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout=0.0, edge_dim=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_root = nn.Linear(in_channels, heads * out_channels)

    def forward(self, x, graph, edata):
        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x = (x, x)

        batch_size = x[0].shape[1]
        src_feats, dst_feats = x
        dgl_graph = graph.to_dgl_graph()
        src_feats = graph.get_src_node_features_in_local_graph(src_feats)
        query = self.lin_query(dst_feats).view(-1, batch_size, H, C)
        key = self.lin_key(src_feats).view(-1, batch_size, H, C)
        value = self.lin_value(src_feats).view(-1, batch_size, H, C)
        edata = self.lin_edge(edata).view(-1, H, C)
        edata = edata.unsqueeze(1)

        with dgl_graph.local_scope():
            dgl_graph.srcdata["key"] = key
            dgl_graph.dstdata["query"] = query
            dgl_graph.srcdata["value"] = value
            dgl_graph.apply_edges(fn.u_mul_v("key", "query", "attn"))
            dgl_graph.edata["attn"] = (
                dgl_graph.edata["attn"] / math.sqrt(C) + edata
            ).sum(-1, keepdim=True)
            dgl_graph.edata["attn"] = edge_softmax(dgl_graph, dgl_graph.edata["attn"])
            dgl_graph.edata["attn"] = F.dropout(
                dgl_graph.edata["attn"], p=self.dropout, training=self.training
            )
            dgl_graph.update_all(
                fn.u_mul_e("value", "attn", "att_v"), fn.sum("att_v", "new_feats")
            )
            out = dgl_graph.dstdata["new_feats"]

        out = out.view(-1, batch_size, H * C)
        x_res = self.lin_root(dst_feats)

        out = x_res + out
        return out


class MultiLayer(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, depth):
        super().__init__()

        layers = []
        for i in range(depth):
            in_dim = in_feats if i == 0 else hidden_feats
            out_dim = hidden_feats if (i + 1) < depth else out_feats
            layers.append(nn.Linear(in_dim, out_dim))
            if (i + 1) < depth:
                layers.append(LayerNorm(out_dim))
                layers.append(nn.ELU())

        self.all_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.all_layers(x)


class LayerNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(in_channels))
        self.bias = nn.Parameter(torch.zeros(in_channels))

        self.register_buffer(
            "norm_scaler", torch.tensor(-1)
        )  # using -1 as place holder

    def forward(self, x):
        """Approximate variance with mean of sum of sq minus sq of mean"""
        s = x.sum(0, keepdim=True)
        s_sq = (x**2).sum(0, keepdim=True)
        n = torch.ones(1, dtype=x.dtype, device=x.device) * x.shape[0]
        if dist.is_initialized() and dist.get_world_size() > 1:
            all_tensors = torch.stack([s, s_sq], dim=0)
            all_tensors = dist.nn.all_reduce(all_tensors, op=dist.ReduceOp.SUM)
            dist.all_reduce(n, op=dist.ReduceOp.SUM)
            if self.training and (self.norm_scaler == -1):
                self.norm_scaler = torch.tensor(dist.get_world_size()).to(
                    self.weight.device
                )
            s, s_sq = all_tensors

        if self.norm_scaler != -1:
            s = s / self.norm_scaler  # Without division sometimes loss = nan
            s_sq = s_sq / self.norm_scaler  # Without division sometimes loss = nan

        mean = s / n
        var = (s_sq / n) - mean**2
        var[var < 0] = 0
        std = torch.sqrt(var)
        x = x - mean.to(x.device)
        out = x / (std.to(x.device) + self.eps)
        out = out * self.weight + self.bias
        return out


class GlobalLayerNorm(nn.Module):
    def __init__(self, in_channels, eps=1e-5):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(in_channels))
        self.bias = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x):
        x = x - x.mean(0, keepdim=True)
        out = x / (x.std(0, keepdim=True, unbiased=False) + self.eps)

        out = out * self.weight + self.bias
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(weight={tuple(self.weight.shape)}, bias={tuple(self.bias.shape)})"
