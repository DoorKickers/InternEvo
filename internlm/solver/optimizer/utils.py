#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.parallel.comm.utils import WrappedHandle
from internlm.utils.common import get_current_device, get_tensor_norm, move_norm_to_cuda
from internlm.utils.logger import get_logger
from internlm.utils.parallel import (
    is_replica_expert_data_parallel_parameter,
    is_replica_zero_parallel_parameter,
    is_tensor_expert_data_parallel_parameter,
    is_tensor_zero_parallel_parameter,
    is_using_isp,
    is_weight_expert_data_parallel_parameter,
    is_weight_zero_parallel_parameter,
)

logger = get_logger(__file__)

try:
    import amp_C
    from apex.multi_tensor_apply import multi_tensor_applier

    APEX_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    logger.warning("The torch implementation for cal_l2norm is slower than apex. Please note this!")
    APEX_AVAILABLE = False

inf = math.inf


def flatten(input_):
    return _flatten_dense_tensors(input_)


def unflatten(flat, tensors):
    return _unflatten_dense_tensors(flat, tensors)


def get_grad_accumulate_object(tensor):
    """
    Return the AccumulateGrad of the input tensor
    """

    # grad_fn reference:
    # https://discuss.pytorch.org/t/in-the-grad-fn-i-find-a-next-functions-but-i-dont-understand-the-meaning-of-the-attribute/24463
    # expand_as reference: https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand
    #
    # `next_functions` will return the backward graph where
    # the first element is the AccumulateGrad of the leaf nodes.
    # we want to get the AccumulateGrad of the input tensor instead of the leaf
    # node in the whole computation graph.
    # Therefore, we call expand_as to create a dummy graph
    # where tensor_tmp and tensor indeed point to the same object.
    # You can check this by print(tensor.data_ptr() == tensor_tmp.data_ptr())
    tensor_tmp = tensor.expand_as(tensor)
    grad_acc_obj = tensor_tmp.grad_fn.next_functions[0][0]
    return grad_acc_obj


def split_half_float_double(tensor_list):
    dtype_buckets = {}

    for t in tensor_list:
        dtype = t.type()
        if dtype not in dtype_buckets:
            dtype_buckets[dtype] = []

        dtype_buckets[dtype].append(t)

    buckets = [bucket for bucket in dtype_buckets.values() if bucket]
    return buckets


def reduce_tensor(
    tensor,
    dtype=None,
    dst_rank=None,
    parallel_mode=ParallelMode.DATA,
    op_type=torch.distributed.ReduceOp.AVG,
    async_op=True,
):
    """
    Reduce the tensor in the data parallel process group

    :param tensor: A tensor object to reduce/all-reduce
    :param dtype: The data type used in communication
    :param dst_rank: The source rank for reduce. If dst_rank is None,
    :param parallel_mode: Communication parallel mode
    all-reduce will be used instead of reduce. Default is None.

    :type tensor: torch.Tensor
    :type dtype: torch.dtype, optional
    :type dst_rank: int, optional
    :type parallel_mode: ParallelMode, optional
    """
    # use the original dtype
    # if dtype is None:
    assert dtype is None
    dtype = gpc.config.reduce_comm_dtype
    tensor_dtype = tensor.dtype

    # cast the data to specified dtype for reduce/all-reduce
    if tensor_dtype != dtype:
        tensor = tensor.to(dtype)

    # world_size = gpc.get_world_size(parallel_mode)
    # tensor.div_(world_size)
    group = gpc.get_group(parallel_mode)

    # if rank is None, all reduce will be used
    # else, reduce is used
    use_all_reduce = dst_rank is None

    if use_all_reduce:
        handle = dist.all_reduce(tensor=tensor, group=group, op=op_type, async_op=async_op)
    else:
        ranks_in_group = gpc.get_ranks_in_group(parallel_mode)
        global_rank = ranks_in_group[dst_rank]
        handle = dist.reduce(tensor=tensor, dst=global_rank, group=group, op=op_type, async_op=async_op)

    if tensor_dtype != dtype:
        if async_op:
            handle = WrappedHandle(handle=handle, output=tensor, dtype=tensor_dtype)
        else:
            tensor = tensor.to(tensor_dtype)
    return handle


def has_inf_or_nan(tensor):
    try:
        # if tensor is half, the .float() incurs an additional deep copy, but it's necessary if
        # Pytorch's .sum() creates a one-element tensor of the same type as tensor
        # (which is true for some recent version of pytorch).
        tensor_sum = float(tensor.float().sum())
        # More efficient version that can be used if .sum() returns a Python scalar
        # tensor_sum = float(tensor.sum())
    except RuntimeError as instance:
        # We want to check if inst is actually an overflow exception.
        # RuntimeError could come from a different error.
        # If so, we still want the exception to propagate.
        if "value cannot be converted" not in instance.args[0]:
            raise
        return True
    else:
        if tensor_sum == float("inf") or tensor_sum == -float("inf"):
            return True
        return False


def release_param_grad(tensor_list):
    for tensor in tensor_list:
        tensor.grad = None


def sync_param(flat_tensor, tensor_list):
    """
    Synchronize the flattened tensor and unflattened tensor list. When
    a list of tensor are flattened with `torch._utils._unflatten_dense_tensors`,
    a new tensor is created. Thus, the flat tensor and original tensor list do not
    share the same memory space. This function will update the tensor list so that
    they point to the same value.

    :param flat_tensor: A flat tensor obtained by calling `torch._utils._unflatten_dense_tensors` on a tensor lsit
    :param tensor_list: A list of tensors corresponding to the flattened tensor
    :type flat_tensor: torch.Tensor
    :type tensor_list: List[torch.Tensor]
    """
    updated_params = unflatten(flat_tensor, tensor_list)

    # update the tensor data
    for p, q in zip(tensor_list, updated_params):
        p.data = q.data


def multi_tensor_l2norm_torch(tensor_list, per_tensor):
    # Convert tensor_list elements to torch.float32
    tensor_list = [tensor.float() for tensor in tensor_list]
    norms_tensor = torch.stack([torch.norm(tensor, p=2) for tensor in tensor_list])
    l2_norm = torch.norm(norms_tensor, p=2).unsqueeze(0)

    if per_tensor:
        per_tensor_norm = norms_tensor
    else:
        per_tensor_norm = torch.Tensor([]).to(norms_tensor.device)

    return l2_norm, per_tensor_norm


def calc_l2_norm(grads):
    norm = 0.0
    if len(grads) > 0:
        if APEX_AVAILABLE:
            dummy_overflow_buf = torch.tensor([0], device=get_current_device(), dtype=torch.int32)
            norm, _ = multi_tensor_applier(
                amp_C.multi_tensor_l2norm,
                dummy_overflow_buf,
                [grads],
                False,  # no per-parameter norm
            )
        else:
            norm, _ = multi_tensor_l2norm_torch(grads, False)
    return norm


def calc_lp(grads, norm_type):
    norm = 0.0
    for grad in grads:
        grad_norm = torch.norm(grad, norm_type)
        norm += grad_norm**norm_type
    return norm


def get_norm(grads, norm_type, enable_cuda_kernels):
    if norm_type == inf:
        grad_norm = max(g.data.abs().max() for g in grads)
    elif norm_type == 2.0 and enable_cuda_kernels:
        grad_norm = calc_l2_norm(grads) ** norm_type
    else:
        grad_norm = calc_lp(grads, norm_type)
    return grad_norm


def reduce_grads(gradients, parameters, weight_parallel_mode):
    parallel_grads = []

    for g, p in zip(gradients, parameters):
        # TODO: consider the pipeline shared parameter
        if (
            gpc.is_initialized(ParallelMode.PIPELINE)
            and hasattr(p, "pipeline_shared_module_pg")
            and dist.get_rank(p.pipeline_shared_module_pg) == 0
        ):  # if shared between different pipe, only count o
            parallel_grads.append(g.data.float())
        elif (
            gpc.is_initialized(ParallelMode.PIPELINE)
            and hasattr(p, "pipeline_shared_module_pg")
            and dist.get_rank(p.pipeline_shared_module_pg) != 0
        ):
            continue
        elif (
            is_replica_zero_parallel_parameter(p) and gpc.get_local_rank(weight_parallel_mode) == 0
        ):  # if not used in each chunk, such as layernorm IS_REPLICA_ZERO_PARALLEL parameter group
            parallel_grads.append(g.data.float())
        elif is_tensor_zero_parallel_parameter(p):
            # process all ranks for IS_TENSOR_ZERO_PARALLEL parameter group
            parallel_grads.append(g.data.float())
        elif is_weight_zero_parallel_parameter(p):
            # process all ranks for IS_WEIGHT_ZERO_PARALLEL parameter group
            parallel_grads.append(g.data.float())
        elif is_tensor_expert_data_parallel_parameter(p):
            # process all ranks for IS_TENSOR_EXPERT_DATA_PARALLEL parameter group
            parallel_grads.append(g.data.float())
        elif is_weight_expert_data_parallel_parameter(p):
            # process all ranks for IS_WEIGHT_EXPERT_DATA_PARALLEL parameter group
            parallel_grads.append(g.data.float())
        elif is_replica_expert_data_parallel_parameter(p) and gpc.get_local_rank(weight_parallel_mode) == 0:
            # process all ranks for IS_REPLICA_EXPERT_DATA_PARALLEL parameter group
            parallel_grads.append(g.data.float())
        elif gpc.get_local_rank(weight_parallel_mode) != 0:
            continue
        else:
            raise RuntimeError("Should not arrive here")
    return parallel_grads


def compute_norm(gradients, parameters, norm_type=2, zero_mode=ParallelMode.ZERO1):
    """Get the norm
    Arguments:
        gradients (Iterable[Tensor]): The gradient value.
        parameters (Iterable[Tensor]): The parameter each gradient corresponds to.
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters, need total_norm**(1/norm) before using.
    """

    weight_parallel_mode = ParallelMode.WEIGHT if is_using_isp() else ParallelMode.TENSOR
    enable_cuda_kernels = gradients[0].device.type != "cpu"
    # Norm parameters.
    norm_type = float(norm_type)

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(g.data.abs().max() for g in gradients)
        total_norm_cuda = torch.tensor([float(total_norm)], device=gradients[0].device, dtype=torch.float32)

        # Take max across all model-parallel GPUs.
        if is_tensor_zero_parallel_parameter(parameters[0]):
            if gpc.is_using_parallel_mode(ParallelMode.TENSOR):
                dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.TENSOR))
        else:
            if gpc.is_using_parallel_mode(weight_parallel_mode):
                dist.all_reduce(
                    total_norm_cuda,
                    op=dist.ReduceOp.MAX,
                    group=gpc.get_group(weight_parallel_mode),
                )

        if gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
            dist.all_reduce(
                total_norm_cuda,
                op=dist.ReduceOp.MAX,
                group=gpc.get_group(ParallelMode.PIPELINE),
            )

        total_norm = total_norm_cuda[0].item()
    else:
        tensor_parallel_grads = reduce_grads(gradients, parameters, weight_parallel_mode)

        tensor_parallel_norm = get_norm(tensor_parallel_grads, norm_type, enable_cuda_kernels)

        # If norm is type of float, then we convert them into torch.Tensor.
        tensor_parallel_norm = get_tensor_norm(tensor_parallel_norm, enable_cuda_kernels)
        # If grads are on CPU, the norms is also on CPU. Cast them to CUDA tensors
        if not enable_cuda_kernels:
            tensor_parallel_norm = move_norm_to_cuda(tensor_parallel_norm)

        total_norm = tensor_parallel_norm

        """
        Sum across all model-parallel GPUs.
        1. For the IS_REPLICA_ZERO_PARALLEL parameter group, gradients from rank 0 in the tp/wp process group and
            gradients along the pp+zero dimensions from all ranks should be aggregated.
        2. For the IS_TENSOR_ZERO_PARALLEL parameter group, gradients along the tp+pp+zero dimensions
            from all ranks should be aggregated.
        3. For the IS_WEIGHT_ZERO_PARALLEL parameter group, gradients along the wp+pp+zero dimensions
            from all ranks should be aggregated.
        """
        if is_tensor_zero_parallel_parameter(parameters[0]):
            if gpc.is_using_parallel_mode(ParallelMode.TENSOR):
                dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.TENSOR))
        else:
            if gpc.is_using_parallel_mode(weight_parallel_mode):
                dist.all_reduce(
                    total_norm,
                    op=dist.ReduceOp.SUM,
                    group=gpc.get_group(weight_parallel_mode),
                )

        if gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
            dist.all_reduce(
                total_norm,
                op=dist.ReduceOp.SUM,
                group=gpc.get_group(ParallelMode.PIPELINE),
            )

        # This is because we use zero1, so we need to use this reduction.
        if gpc.is_using_parallel_mode(zero_mode):
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=gpc.get_group(zero_mode))

        if torch.is_tensor(total_norm):
            total_norm = total_norm.item()

    # Need to allreduce(avg) the norms across different ranks because moe params will not be synced during allreduce
    # model and zero have been reduced!!!
    if zero_mode == ParallelMode.EXPERT_DATA:
        pg = gpc.get_group(ParallelMode.EXPERT)
        scaled_norm = total_norm * 1.0 / float(gpc.get_world_size(ParallelMode.EXPERT))
        scaled_norm_tensor = torch.tensor(scaled_norm, device=get_current_device(), dtype=torch.float)
        dist.all_reduce(scaled_norm_tensor, group=pg)
        total_norm = scaled_norm_tensor.item()

    # Scale.
    if total_norm == float("inf") or total_norm == -float("inf"):
        total_norm = -1

    if math.isnan(total_norm):
        total_norm = -2

    return total_norm


class BaseGradScaler(ABC):
    """A base class for the gradient scaler.

    Args:
        initial_scale (float): the initial loss scale
    """

    def __init__(self, initial_scale: float):
        assert initial_scale > 0
        self._scale = torch.tensor([initial_scale], device=get_current_device(), dtype=torch.float32)

    @property
    def scale(self) -> Tensor:
        """Returns the loss scale."""

        return self._scale

    @property
    def inv_scale(self) -> Tensor:
        """Returns the inverse of the loss scale."""

        return self._scale.double().reciprocal().float()

    def state_dict(self) -> Dict:
        """Returns the states of the gradient scaler as a dict object."""

        state_dict = dict()
        state_dict["scale"] = self.scale
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load the states of the gradient scaler from a dict object.

        Args:
            state_dict (dict): the states of the gradient scaler
        """

        self._scale = state_dict["scale"]

    @abstractmethod
    def update(self, overflow: bool) -> None:
        """Update the loss scale.

        Args:
            overflow (bool): whether overflow occurs
        """

        pass


class DynamicGradScaler(BaseGradScaler):
    """A gradient scaler which uses dynamic loss scale

    Args:
        initial_scale (float): the initial loss scale, defaults to 2**16
        growth_factor (float): the multiplication factor for increasing loss scale, defaults to 2
        backoff_factor (float): the multiplication factor for decreasing loss scale, defaults to 0.5
        growth_interval (int): the number of steps to increase loss scale when no overflow occurs, defaults to 1000
        min_scale (float): the minimum loss scale, defaults to None
        max_scale (float): the maximum loss scale, defaults to None
        hysteresis (int):  the number of overflows before decreasing loss scale, defaults to 2
    """

    def __init__(
        self,
        initial_scale: float = 2**16,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        min_scale: Optional[float] = None,
        max_scale: Optional[float] = None,
        hysteresis: int = 2,
        dtype=torch.bfloat16,
    ):
        super().__init__(initial_scale)
        if min_scale:
            self._min_scale = torch.tensor([min_scale], device=get_current_device(), dtype=torch.float32)
        else:
            self._min_scale = None

        if max_scale:
            self._max_scale = torch.tensor([max_scale], device=get_current_device(), dtype=torch.float32)
        else:
            self._max_scale = None

        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_step = 0
        self._hysteresis = hysteresis
        self._hysteresis_step = 0
        self._dtype = dtype
        self._sanity_checks()

    def _sanity_checks(self) -> None:
        """Check if the arguments are correct."""

        assert self._dtype in [torch.float16, torch.bfloat16, torch.float32]

        if self._min_scale is not None:
            min_scale = self._min_scale.item()
            assert min_scale > 0, "The minimum gradient scale cannot be zero or negative"

            if self._dtype != torch.float16 and min_scale != 1.0 and gpc.is_rank_for_log():
                logger.warning(f"Detect you use {self._dtype}, but min_scale: {min_scale} != 1.0")

        if self._max_scale:
            max_scale = self._max_scale.item()
            assert max_scale > 0, "The maximum gradient scale cannot be zero or negative"

            if self._dtype != torch.float16 and max_scale != 1.0 and gpc.is_rank_for_log():
                logger.warning(f"Detect you use {self._dtype}, but max_scale: {max_scale} != 1.0")

        if self._dtype == torch.float16:
            assert self._growth_factor > 1.0, "The growth factor cannot be equal or smaller than 1"
            assert self._backoff_factor < 1.0 and self._backoff_factor > 0, "The backoff factor must be between 0 and 1"
        else:
            assert self._growth_factor >= 1.0, "The growth factor cannot be smaller than 1"
            assert (
                self._backoff_factor <= 1.0 and self._backoff_factor > 0
            ), "The backoff factor must be between 0 and 1"

            if self._growth_factor != 1.0 and gpc.is_rank_for_log():
                logger.warning(f"Detect you use {self._dtype}, but growth_factor: {self._growth_factor} != 1.0")
            if self._backoff_factor != 1.0 and gpc.is_rank_for_log():
                logger.warning(f"Detect you use {self._dtype}, but backoff_factor: {self._backoff_factor} != 1.0")

        assert self._hysteresis >= 0, "The hysteresis cannot be negative"

    def update(self, overflow: bool) -> None:
        """Update the loss scale.

        Args:
            overflow (bool): whether overflow occurs
        """
        if overflow:
            self._hysteresis_step += 1
            self._growth_step = 0

            if self._hysteresis_step >= self._hysteresis:
                self._backoff_scale()
                if gpc.is_rank_for_log():
                    logger.warning(f"Overflow occurs, the loss scale is adjusted to {self.scale.item()}")
        else:
            self._growth_step += 1
            if self._growth_step == self._growth_interval:
                self._growth_step = 0
                self._hysteresis_step = 0
                self._grow_scale()
                if gpc.is_rank_for_log():
                    logger.warning(
                        f"No overflow for consecutive {self._growth_interval} steps, "
                        f"the loss scale is adjusted to {self.scale.item()}",
                    )

    def _backoff_scale(self) -> None:
        """Decrease the loss scale"""

        self._scale = self._scale * self._backoff_factor
        if self._min_scale:
            self._scale = torch.max(self._scale, self._min_scale)

    def _grow_scale(self) -> None:
        """Increase the loss scale"""

        self._scale = self._scale * self._growth_factor
        if self._max_scale:
            self._scale = torch.min(self._scale, self._max_scale)

    def state_dict(self):
        """Returns the states of the gradient scaler as a dict object."""

        state_dict = dict()
        state_dict["_scale"] = self._scale.item()
        state_dict["_growth_step"] = self._growth_step
        state_dict["_hysteresis_step"] = self._hysteresis_step

        return state_dict

    def load_state_dict(self, state_dict):
        """Load the states of the gradient scaler from a dict object.

        Args:
            state_dict (dict): the states of the gradient scaler
        """

        self._scale = self._scale.fill_(state_dict["_scale"])
        self._growth_step = state_dict["_growth_step"]
        self._hysteresis_step = state_dict["_hysteresis_step"]
