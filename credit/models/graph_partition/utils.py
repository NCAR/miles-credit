# This file includes code adapted from NVIDIA's PhysicsNemo project:
# https://github.com/NVIDIA/physicsnemo


# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.distributed as dist


def get_memory_format(tensor):
    """Gets format for tensor"""
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    else:
        return torch.contiguous_format


def all_gather_v_wrapper(
    tensor,
    sizes=None,
    dim=0,
    group=None,
):  # pragma: no cover
    """
    Implements a distributed AllGatherV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive gathers all local tensors from each rank into the
    full global tensor onto each rank.

    Parameters
    ----------
    tensor : "torch.Tensor"
        local tensor on each rank
    sizes : List[int], optional
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank, by default None
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        full global tensor, valid on each rank
    """

    comm_size = dist.get_world_size(group=group)

    if (sizes is not None) and (len(sizes) != comm_size):
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()

    if comm_size == 1:
        return tensor

    tensor_shape = list(tensor.shape)
    tensor_format = get_memory_format(tensor)

    if sizes is not None:
        tensor_list = [None] * comm_size

        for src in range(comm_size):
            tensor_shape[dim] = sizes[src]
            tensor_list[src] = torch.empty(
                tensor_shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )
    else:
        # assume equal shape on all ranks
        tensor_list = [torch.empty_like(tensor) for _ in range(comm_size)]

    dist.all_gather(tensor_list, tensor, group=group)

    output = torch.cat(tensor_list, dim=dim).contiguous(memory_format=tensor_format)

    return output


def all_gather_v_bwd_wrapper(
    tensor,
    sizes,
    dim=0,
    use_fp32=True,
    group=None,
):  # pragma: no cover
    """
    Implements a distributed AllReduceV primitive. It is based
    on the idea of a single global tensor which which can be distributed
    along a specified dimension into chunks of variable size.
    This primitive assumes different global tensors of the same shape on each
    rank. It then re-distributes chunks of all these tensors such that each rank
    receives all corresponding parts of a global tensor. Each rank then sums up
    the chunks after receiving it. By design, this primitive thus implements the
    backward pass of the "all_gather_v" primitive. In this case, the result would
    be a single global gradient tensor distributed onto different ranks.

    Parameters
    ----------
    tensor : torch.Tensor
        global tensor on each rank (different one on each rank)
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    use_fp32 : bool, optional
        flag to specify reduction taking place at least in FP32 precision, by default True
        only acts on floating point inputs in lower precision
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        local tensor, i.e. result of reduction of all corresponding chunks
        from all global tensors for each rank separately
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()

    tensor_shape = list(tensor.shape)
    tensor_shape[dim] = sizes[rank]
    tmp = [
        torch.empty(
            tensor_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        for _ in range(comm_size)
    ]
    scatter_list = list(torch.split(tensor, sizes, dim=dim))
    scatter_list = [t.contiguous() for t in scatter_list]

    dist.all_to_all(tmp, scatter_list, group=group)
    stack_dim = tensor.dim()
    tmp = torch.stack(tmp, dim=stack_dim)

    if use_fp32 and (tmp.dtype.itemsize < 4) and tmp.dtype.is_floating_point:
        # cast to float before sum and return float, then cast back
        output = tmp.sum(dim=stack_dim, dtype=torch.float32)
        output = output.to(dtype=tensor.dtype)
    else:
        # else: just do sum in native dtype
        output = tmp.sum(dim=stack_dim)

    return output


def indexed_all_to_all_v_wrapper(
    tensor,
    indices,
    sizes,
    dim=0,
    group=None,
):  # pragma: no cover
    """
    Implements an indexed version of a distributed AllToAllV
    primitive. It is based on the idea of a single global tensor which
    is distributed along a specified dimension into chunks of variable size.
    This primitive assumes a set of indices into this dimension which indicate
    the corresponding slices sent to each other rank forming an indexed version
    of an AllToAllV primitive.

    Parameters
    ----------
    tensor : torch.Tensor
        local part of global tensor on each rank
    indices : List[torch.Tensor]
        list of indices on each rank of slices being sent to
        each other rank from this rank
    sizes : List[List[int]]
        number of indices each rank sends to each other rank,
        valid and set on each rank, e.g. sizes[0][3] corresponds
        to the number of slices rank 0 sends to rank 3
    dim : int
        dimension along which global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        local result of primitive corresponding to indexed global tensor
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if len(sizes[rank]) != comm_size:
        raise ValueError()
    if len(indices) != comm_size:
        raise ValueError()

    x_send = [tensor[idx] for idx in indices]
    x_recv = [None] * comm_size
    tensor_shape = list(tensor.shape)
    for r in range(comm_size):
        tensor_shape[dim] = sizes[r][rank]
        x_recv[r] = torch.empty(
            tensor_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    dist.all_to_all(x_recv, x_send, group=group)

    tensor_to_recv = torch.cat(x_recv, dim=dim)

    return tensor_to_recv


def indexed_all_to_all_v_wrapper_bwd(
    tensor,
    indices,
    sizes,
    tensor_size_along_dim,
    use_fp32=True,
    dim=0,
    group=None,
):  # pragma: no cover
    """
    Implements the backward pass to the indexed version of a distributed
    AllToAllV primitive.

    Parameters
    ----------
    tensor : torch.Tensor
        local tensor, i.e. gradient on resulting tensor from forward pass
    indices : List[torch.Tensor]
        list of indices on each rank of slices being sent to
        each other rank from this rank
    sizes : List[List[int]]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    tensor_size_along_dim : int
        size of original local tensor along specified dimension,
        i.e. from the corresponding forward pass
    use_fp32 : bool, optional
        flag to specify reduction taking place at least in FP32 precision, by default True
        only acts on floating point inputs in lower precision
    dim : int, optional
        dimension along with global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        result of primitive corresponding to indexed global tensor
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if len(sizes[rank]) != comm_size:
        raise ValueError()
    if len(indices) != comm_size:
        raise ValueError()

    tensor_shape = list(tensor.shape)

    # scatter gradients, roles reversed compared to forward pass
    # recv_sizes in forward pass
    recv_sizes = [sizes[i][rank] for i in range(comm_size)]
    # send_sizes in forward pass
    send_sizes = [sizes[rank][i] for i in range(comm_size)]

    x_send = torch.split(tensor, recv_sizes, dim=dim)
    x_send = [t.contiguous() for t in x_send]
    x_recv = [None] * comm_size
    for r in range(comm_size):
        tensor_shape[dim] = send_sizes[r]
        x_recv[r] = torch.empty(
            tensor_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    dist.all_to_all(x_recv, x_send, group=group)

    tensor_to_recv = torch.cat(x_recv, dim=dim)

    # sum up gathered gradients and taking
    # care of precision handling as specified
    # by boolean flag
    indices = torch.cat(indices, dim=0)
    tensor_shape[dim] = tensor_size_along_dim
    if use_fp32 and (tensor.dtype.itemsize < 4) and tensor.dtype.is_floating_point:
        out = torch.zeros(
            tensor_shape,
            dtype=torch.float32,
            device=tensor.device,
        )
        tensor_to_recv = tensor_to_recv.to(dtype=torch.float32)
    else:
        out = torch.zeros(
            tensor_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    out.index_add_(source=tensor_to_recv, index=indices, dim=dim)

    if out.dtype != tensor.dtype:
        out = out.to(tensor.dtype)

    return out


def get_lat_lon_partition_separators(partition_size: int):
    """Utility Function to get separation intervals for lat-lon
    grid for partition_sizes of interest.

    Parameters
    ----------
    partition_size : int
        size of graph partition
    """

    def _divide(num_lat_chunks: int, num_lon_chunks: int):
        # divide lat-lon grid into equally-sizes chunks along both latitude and longitude
        if (num_lon_chunks * num_lat_chunks) != partition_size:
            raise ValueError(
                "Can't divide lat-lon grid into grid {num_lat_chunks} x {num_lon_chunks} chunks for partition_size={partition_size}."
            )
        # divide latitutude into num_lat_chunks of size 180 / num_lat_chunks
        # divide longitude into chunks of size 360 / (partition_size / num_lat_chunks)
        lat_bin_width = 180.0 / num_lat_chunks
        lon_bin_width = 360.0 / num_lon_chunks

        lat_ranges = []
        lon_ranges = []

        for p_lat in range(num_lat_chunks):
            for p_lon in range(num_lon_chunks):
                lat_ranges += [
                    (lat_bin_width * p_lat - 90.0, lat_bin_width * (p_lat + 1) - 90.0)
                ]
                lon_ranges += [
                    (lon_bin_width * p_lon - 180.0, lon_bin_width * (p_lon + 1) - 180.0)
                ]

        lat_ranges[-1] = (lat_ranges[-1][0], None)
        lon_ranges[-1] = (lon_ranges[-1][0], None)

        return lat_ranges, lon_ranges

    # use two closest factors of partition_size
    lat_chunks, lon_chunks, i = 1, partition_size, 0
    while lat_chunks < lon_chunks:
        i += 1
        if partition_size % i == 0:
            lat_chunks = i
            lon_chunks = partition_size // lat_chunks

    lat_ranges, lon_ranges = _divide(lat_chunks, lon_chunks)

    # mainly for debugging
    if (lat_ranges is None) or (lon_ranges is None):
        raise ValueError("unexpected error, abort")

    min_seps = []
    max_seps = []

    for i in range(partition_size):
        lat = lat_ranges[i]
        lon = lon_ranges[i]
        min_seps.append([lat[0], lon[0]])
        max_seps.append([lat[1], lon[1]])

    return min_seps, max_seps
