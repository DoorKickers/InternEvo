#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.distributed as dist

from internlm.core.context import (
    IS_REPLICA_EXPERT_DATA_PARALLEL,
    IS_REPLICA_ZERO_PARALLEL,
    IS_TENSOR_EXPERT_DATA_PARALLEL,
    IS_TENSOR_ZERO_PARALLEL,
    IS_WEIGHT_EXPERT_DATA_PARALLEL,
    IS_WEIGHT_ZERO_PARALLEL,
    ParallelMode,
)
from internlm.core.context import global_context as gpc
from internlm.model.modules.utils import is_gate_param
from internlm.utils.utils import TensorParallelMode


def is_using_hf():
    return "hf" in gpc.config


def is_using_fsdp():
    return (
        "fsdp" in gpc.config.parallel
        and isinstance(gpc.config.parallel["fsdp"], dict)
        and gpc.config.parallel["fsdp"].get("enable", False)
    )


def is_using_sequence_parallel():
    return (
        isinstance(gpc.config.parallel["tensor"], dict)
        and gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name) != TensorParallelMode.mtp.name
        and gpc.config.parallel["tensor"]["size"] > 1
    )


def is_using_isp():
    return (
        isinstance(gpc.config.parallel["tensor"], dict)
        and gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name) == TensorParallelMode.isp.name
    )


def is_replica_zero_parallel_parameter(p):
    return hasattr(p, IS_REPLICA_ZERO_PARALLEL) and getattr(p, IS_REPLICA_ZERO_PARALLEL)


def is_tensor_zero_parallel_parameter(p):
    return (
        gpc.is_initialized(ParallelMode.TENSOR)
        and not is_using_isp()
        and hasattr(p, IS_TENSOR_ZERO_PARALLEL)
        and getattr(p, IS_TENSOR_ZERO_PARALLEL)
    )


def is_weight_zero_parallel_parameter(p):
    return (
        gpc.is_initialized(ParallelMode.WEIGHT)
        and is_using_isp()
        and hasattr(p, IS_WEIGHT_ZERO_PARALLEL)
        and getattr(p, IS_WEIGHT_ZERO_PARALLEL)
    )


def is_tensor_expert_data_parallel_parameter(p):
    return (
        gpc.is_initialized(ParallelMode.TENSOR)
        and hasattr(p, IS_TENSOR_EXPERT_DATA_PARALLEL)
        and getattr(p, IS_TENSOR_EXPERT_DATA_PARALLEL)
    )


def is_weight_expert_data_parallel_parameter(p):
    return (
        gpc.is_initialized(ParallelMode.WEIGHT)
        and hasattr(p, IS_WEIGHT_EXPERT_DATA_PARALLEL)
        and getattr(p, IS_WEIGHT_EXPERT_DATA_PARALLEL)
    )


def is_replica_expert_data_parallel_parameter(p):
    return hasattr(p, IS_REPLICA_EXPERT_DATA_PARALLEL) and getattr(p, IS_REPLICA_EXPERT_DATA_PARALLEL)


def should_reduce_replica_param(p):
    _reduce = False

    if not is_replica_zero_parallel_parameter(p):
        return _reduce

    # for replica parameter
    if gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name) == TensorParallelMode.mtp.name:
        _reduce = False
    elif gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name) in (
        TensorParallelMode.msp.name,
        TensorParallelMode.fsp.name,
    ):
        _reduce = gpc.is_using_parallel_mode(ParallelMode.TENSOR)
    elif gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name) == TensorParallelMode.isp.name:
        _reduce = gpc.is_using_parallel_mode(ParallelMode.WEIGHT)

    if not is_gate_param(p):
        return _reduce

    # for moe gate parameter
    if gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name) == TensorParallelMode.mtp.name:
        _reduce = gpc.is_using_parallel_mode(ParallelMode.TENSOR) and getattr(
            gpc.config.parallel.expert, "no_tp", False
        )

    return _reduce


def sync_model_param(model):
    r"""Make sure data parameters are consistent during Data Parallel Mode.

    Args:
        model (:class:`torch.nn.Module`): A pyTorch model on whose parameters you check the consistency.
    """
    sync_moe_param = gpc.is_using_parallel_mode(ParallelMode.EXPERT_DATA)
    sync_parallel_mode = ParallelMode.WEIGHT_DATA if is_using_isp() else ParallelMode.DATA
    for param in model.parameters():
        if getattr(param, "is_expert", False):
            if sync_moe_param:
                ranks = gpc.get_ranks_in_group(ParallelMode.EXPERT_DATA)
                dist.broadcast(param, src=ranks[0], group=gpc.get_group(ParallelMode.EXPERT_DATA))
        else:
            ranks = gpc.get_ranks_in_group(sync_parallel_mode)
            dist.broadcast(param, src=ranks[0], group=gpc.get_group(sync_parallel_mode))


def sync_model_replica_param_group(model):
    r"""This function is changed from colossalai, which is ``sync_model_param``.

    We modified this function to make sure it only sync IS_REPLICA_ZERO_PARALLEL parameters in tp or wp process group.
    This function is used to make sure parameters that are not splitted are the same across each rank.
    For example, parameters like RMSNorm, LayerNorm...

    Args:
        model (:class:`torch.nn.Module`): A pyTorch model on whose parameters you check the consistency.
    """

    parallel_mode = ParallelMode.WEIGHT if is_using_isp() else ParallelMode.TENSOR
    if gpc.is_using_parallel_mode(parallel_mode):
        for param in model.parameters():
            if is_replica_zero_parallel_parameter(param):
                ranks = gpc.get_ranks_in_group(parallel_mode)
                dist.broadcast(param, src=ranks[0], group=gpc.get_group(parallel_mode))


def get_parallel_log_file_name():
    if gpc.is_rank_for_log():
        fn_prefix = "main_"  # Indicates a rank with more output information
    else:
        fn_prefix = ""

    log_file_name = (
        f"{fn_prefix}dp={gpc.get_local_rank(ParallelMode.DATA)}_"
        f"tp={gpc.get_local_rank(ParallelMode.TENSOR)}_pp={gpc.get_local_rank(ParallelMode.PIPELINE)}"
    )
    return log_file_name
