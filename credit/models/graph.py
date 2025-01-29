import math
import torch
from torch import nn, Tensor
import xarray as xr
from torch.nn import functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import ones, zeros
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax

from typing import Optional, Tuple, Union
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
)
from torch_geometric.utils import to_undirected
from credit.models.base_model import BaseModel


def apply_spectral_norm(model):
    for module in model.modules():
        if isinstance(module, (nn.Linear)):
            nn.utils.spectral_norm(module)


class GraphResTransfGRU(BaseModel):
    def __init__(
        self,
        n_variables=4,
        n_surface_variables=7,
        n_static_variables=3,
        levels=15,
        hidden_size=128,
        dim_head=32,
        dropout=0,
        n_blocks=3,
        history_len=2,
        edge_path="/glade/derecho/scratch/dgagne/credit_scalers/grid_edge_pairs_125.nc",
        use_spectral_norm=True,
        use_edge_attr=True,
    ):
        super().__init__()
        self.n_variables = n_variables
        self.n_static_variables = n_static_variables
        self.n_surface_variables = n_surface_variables
        self.histroy_len = history_len
        self.n_levels = levels
        self.state_vars = self.n_variables * self.n_levels + self.n_surface_variables
        self.total_n_vars = (
            self.state_vars + self.n_static_variables
        ) * self.histroy_len
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        self.dim_head = dim_head
        assert hidden_size % dim_head == 0
        self.heads = hidden_size // dim_head
        self.dropout = dropout
        self.edge_path = edge_path
        self.use_spectral_norm = use_spectral_norm
        self.use_edge_attr = use_edge_attr

        self.load_graph()

        # Try to see if you can load the edges here.

        self.encoder = nn.Linear(self.total_n_vars, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.state_vars)

        if self.use_edge_attr:
            self.edge_ln = nn.Linear(3, self.hidden_size)
            self.edge_layer_norm = LayerNorm(self.hidden_size)
        else:
            self.edge_ln = self.register_parameter("edge_ln", None)
            self.edge_layer_norm = self.register_parameter("edge_layer_norm", None)

        self.graph_blocks = nn.ModuleList(
            GraphResidualBlock(
                self.hidden_size,
                self.hidden_size,
                self.hidden_size,
                self.heads,
                self.dropout,
                self.use_spectral_norm,
                self.use_edge_attr,
            )
            for i in range(self.n_blocks)
        )
        self.gated_unit = GateCell(self.hidden_size)

        self.graph_norm_layers = nn.ModuleList(
            LayerNorm(self.hidden_size) for _ in range(self.n_blocks)
        )
        self.enc_norm = LayerNorm(self.hidden_size)
        self.dec_norm = LayerNorm(self.hidden_size)

        if self.use_spectral_norm:
            apply_spectral_norm(self)

    def forward(self, x):
        edge_index = self.edge_index

        edge_attr = self.edge_attr

        if self.use_edge_attr:
            edge_attr = self.edge_ln(edge_attr)
            edge_attr = self.edge_layer_norm(edge_attr)

        lat_lon_shape = x.shape[-2:]

        state = x[:, : self.state_vars, 1:]
        x = x.view(-1, self.total_n_vars, lat_lon_shape[0] * lat_lon_shape[1]).permute(
            2, 0, 1
        )
        x = self.encoder(x)
        x = self.enc_norm(x)  # Seems to be the most useful

        # self.gated_unit.h = None
        h = None
        for graph_transf, norm_layer in zip(self.graph_blocks, self.graph_norm_layers):
            x = graph_transf(x, edge_index, edge_attr=edge_attr)
            x = norm_layer(x)
            h = self.gated_unit(x, h)
            x = h
        # x = self.dec_norm(x)

        # x = h
        x = self.decoder(x).permute(1, 2, 0)
        x = x.view(-1, self.state_vars, 1, *lat_lon_shape)

        return x + state

    def load_graph(self):
        xr_dataset = xr.open_dataset(self.edge_path)
        self.edge_index = torch.from_numpy(xr_dataset.edges.values.T)
        self.edge_attr = None

        if self.use_edge_attr:
            self.edge_dist = (
                torch.from_numpy(xr_dataset.distances.values).unsqueeze(1).float()
            )
            self.edge_dist = (self.edge_dist - self.edge_dist.mean()) / (
                self.edge_dist.std() + 1e-5
            )  # Normalization
            # self.edge_dist = torch.exp(-(self.edge_dist) ** 2)

            lat = (
                torch.from_numpy(xr_dataset.latitude.values).float() + 180.0
            )  # convert to non-negative values
            lon = torch.from_numpy(xr_dataset.longitude.values).float()

            lat = (lat - lat.mean()) / (lat.std() + 1e-5)
            lon = (lon - lon.mean()) / (lon.std() + 1e-5)

            self.lat_lon = torch.stack([lat, lon], dim=-1)
            self.edge_attr = torch.cat(
                [
                    self.lat_lon[self.edge_index[1]] - self.lat_lon[self.edge_index[0]],
                    self.edge_dist,
                ],
                dim=-1,
            )
            self.edge_attr = torch.exp(-((self.edge_attr) ** 2))
            assert self.edge_attr.shape[1] == 3
            self.edge_index, self.edge_attr = to_undirected(
                self.edge_index, self.edge_attr
            )

        else:
            self.edge_index = to_undirected(self.edge_index)

    def to(self, *args, **kwargs):
        self.edge_index = self.edge_index.to(*args, **kwargs)
        if self.use_edge_attr:
            # self.edge_dist = self.edge_dist.to(*args, **kwargs)
            self.edge_attr = self.edge_attr.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class GraphResidualBlock(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        out_size,
        heads=1,
        dropout=0,
        use_spectral_norm=True,
        use_edge_attr=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.heads = heads
        self.dropout = dropout
        self.use_spectral_norm = use_spectral_norm
        self.use_edge_attr = use_edge_attr
        # self.transformer = TransformerConv(self.hidden_size, self.hidden_size,
        self.transformer = TransformerConv(
            self.input_size,
            self.hidden_size // self.heads,
            heads=self.heads,
            dropout=self.dropout,
            concat=True,
            edge_dim=self.hidden_size if self.use_edge_attr else None,
        )
        self.physics_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.merge_linear = nn.Linear(self.hidden_size, self.out_size)

        if self.use_spectral_norm:
            apply_spectral_norm(self.transformer)
            # apply_spectral_norm(self.physics_linear)
            # apply_spectral_norm(self.merge_linear)

        self.norm_layer = LayerNorm(hidden_size)

    def forward(self, x, edge_index, edge_attr):
        state = x
        # x_table = reshape(x, (-1, x.shape[1] * x.shape[2], x.shape[3]))
        # print(x.shape, 'input size of transformconv is', self.input_size, self.hidden_size, self.out_size)
        x_t = self.transformer(x, edge_index, edge_attr=edge_attr)
        x_t = F.relu(x_t)
        x_t = self.physics_linear(x_t)
        x_t = self.norm_layer(x_t)
        x_t = F.relu(x_t)
        x_t = self.merge_linear(x_t)
        # tendency = reshape(x_t, (-1, x.shape[1], x.shape[2], x.shape[3]))
        tendency = x_t
        tendency += state
        return tendency


class TransformerConv(MessagePassing):
    r"""
    Adapted from
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/transformer_conv.html#TransformerConv
    To added additional `batch` dimension to the input since the graph doesn't change instead of using PyG's graph_batch which will duplicate the same graph.
    """

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter("lin_edge", None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tuple[Tensor, Tensor]],
        Tuple[Tensor, SparseTensor],
    ]:
        H, C = self.heads, self.out_channels
        batch_size = x.shape[1]
        if isinstance(x, Tensor):
            x = (x, x)

        query = self.lin_query(x[1]).view(-1, batch_size, H, C)
        key = self.lin_key(x[0]).view(-1, batch_size, H, C)
        value = self.lin_value(x[0]).view(-1, batch_size, H, C)

        out = self.propagate(
            edge_index, query=query, key=key, value=value, edge_attr=edge_attr
        )

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, batch_size, self.heads * self.out_channels)
        else:
            out = out.mean(dim=2)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j = key_j + edge_attr.unsqueeze(1)

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr.unsqueeze(1)

        out = out * alpha.view(-1, query_i.shape[1], self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )


class LayerNorm(nn.Module):
    def __init__(self, in_channels, eps=1e-5):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(in_channels))
        self.bias = nn.Parameter(torch.empty(in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        ones(self.weight)
        zeros(self.bias)

    def forward(self, x):
        x = x - x.mean(0, keepdim=True)
        out = x / (x.std(0, keepdim=True, unbiased=False) + self.eps)

        return out


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


if __name__ == "__main__":
    n_variables = 4
    n_surface_variables = 7
    n_static_variables = 3
    levels = 15
    hidden_size = 256
    dim_head = 64
    dropout = 0
    n_blocks = 4
    history_len = 2
    use_spectral_norm = True
    use_edge_attr = False

    image_height = 192  # 640, 192
    image_width = 288  # 1280, 288

    # edge_path = "/glade/derecho/scratch/dgagne/credit_scalers/grid_edge_pairs_125.nc"
    edge_path = (
        "/glade/derecho/scratch/dgagne/credit_scalers/grid_edge_pairs_125_onedeg.nc"
    )

    # edge_index = torch.randint(image_height * image_width, size=(2, image_width * image_width * 4))
    input_tensor = torch.randn(
        4,
        n_variables * levels + n_surface_variables + n_static_variables,
        history_len,
        image_height,
        image_width,
    ).to("cpu")
    print("Loading the model input size", input_tensor.shape)
    model_class = GraphResTransfGRU
    print(f"Using the class {model_class} with edges={use_edge_attr}")
    model = model_class(
        n_variables=n_variables,
        n_surface_variables=n_surface_variables,
        n_static_variables=n_static_variables,
        levels=levels,
        hidden_size=hidden_size,
        dim_head=dim_head,
        dropout=0,
        n_blocks=n_blocks,
        history_len=history_len,
        edge_path=edge_path,
        use_spectral_norm=use_spectral_norm,
        use_edge_attr=use_edge_attr,
    ).to("cpu")

    model.train()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    # y_pred = model(input_tensor.to("cuda"), edge_index.cuda())
    y_pred = model(input_tensor.to("cpu"))
    print("Predicted shape:", y_pred.shape)

    # print(model.rk4(input_tensor.to("cpu")).shape)
