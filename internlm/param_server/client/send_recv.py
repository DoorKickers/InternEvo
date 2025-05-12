import importlib
import random
import time

import torch
import torch.distributed as dist

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.param_server.client.client import client, get_ps_id
from internlm.param_server.common import config
from internlm.param_server.proto.master_pb2 import (
    ComputeStatus,
    NormStatus,
    OperatorType,
    PullStatus,
    PushStatus,
)
from internlm.train.pipeline import map_fqn_local_to_global, map_layer_attr
from internlm.utils.common import get_current_device
from internlm.utils.logger import get_logger

logger = get_logger(__file__)

current_status_begin_timestamp = time.time()
status_history_list = []
status = 0

def try_interact_with_param_server(model, optimizer, consume_tokens):
    gpc.consume_steps += 1

    if gpc.consume_steps < gpc.config.sync_step:
        return

    if gpc.is_rank_for_log():
        start_ts = time.time()
        logger.info("start try_interact_with_param_server")
    gpc.consume_steps = 0
    dp_rank = gpc.get_local_rank(ParallelMode.DATA)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    global status
    update_status_history()
    status = 0
    # 1. query whether can push weight
    while dp_rank == 0 and tp_rank == 0 and gpc.is_last_rank(ParallelMode.PIPELINE):

        if not client.master_alive:
            logger.warning("Master Error: Master is not detected, skip sync with ps.")
            break

        op = client.update_push_status(PushStatus.PUSH_STATUS_START)
        if op == OperatorType.PUSH:
            update_status_history()
            status = 1
            break

        if op == OperatorType.IGNORE:
            logger.warning(f"PS Error: ignore pushing weight to ps, {consume_tokens=}")
            break

        if op == OperatorType.PULL:
            update_status_history()
            status = 2
            logger.warning(f"skip pushing weight to ps, will direct pull weight, {consume_tokens=}")
            break

        if op == OperatorType.RETRY:
            time.sleep(5)
    if gpc.is_rank_for_log():
        end_update_push_status_ts = time.time()
        logger.info(f"finish call update_push_status, cost: {end_update_push_status_ts-start_ts:.3f}")

    # 2. broadcast query result
    object_list = [status]
    if dp_rank == 0 and tp_rank == 0 and gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
        src = gpc.get_ranks_in_group(ParallelMode.PIPELINE)[-1]
        dist.broadcast_object_list(object_list, src=src, group=gpc.get_group(ParallelMode.PIPELINE))

    if dp_rank == 0:
        src = gpc.get_ranks_in_group(ParallelMode.TENSOR)[0]
        dist.broadcast_object_list(object_list, src=src, group=gpc.get_group(ParallelMode.TENSOR))

    src = gpc.get_ranks_in_group(ParallelMode.DATA)[0]
    dist.broadcast_object_list(object_list, src=src, group=gpc.get_group(ParallelMode.DATA))

    if gpc.is_rank_for_log():
        end_broadcast_update_push_status_ts = time.time()
        logger.info(
            "finish broadcast update_push_status, cost: "
            f"{end_broadcast_update_push_status_ts-end_update_push_status_ts:.3f}"
        )

    if object_list[0] == 0:
        return

    dynamic_config = importlib.reload(config)
    assert dynamic_config is not None

    # 3. status=1: send weight + query compute status + pull weight, status=2: query compute status + pull weight
    send_status = 0
    if object_list[0] == 1:
        send_status = client_send(model, consume_tokens, dynamic_config)
        if gpc.is_rank_for_log():
            end_client_send_ts = time.time()
            logger.info(f"finish client send, cost: {end_client_send_ts-end_broadcast_update_push_status_ts:.3f}")
    else:
        end_client_send_ts = time.time()
        assert object_list[0] == 2, f"unexpect status: {object_list[0]}"

    if send_status != 0:
        update_status_history()
        status = 0
        logger.error(f"Get abnormal sending status: {send_status}. Pass client receiving.")
        return

    update_status_history()
    status = 2

    # 4. query compute status
    start_query_compute_status_ts = time.time()
    compute_status = query_compute_status_and_broadcast()
    end_query_compute_status_ts = time.time()
    logger.info(f"finish query compute status and broadcast, cost: {end_query_compute_status_ts-start_query_compute_status_ts:.3f}")
    # 5. pull weight
    if compute_status == ComputeStatus.COMPUTE_SUCCESS:
        client_recv(model, optimizer, dynamic_config=dynamic_config)
        if gpc.is_rank_for_log():
            end_client_recv_ts = time.time()
            logger.info(f"finish client recv, cost: {end_client_recv_ts-end_query_compute_status_ts:.3f}")
    else:
        logger.error(f"Get abnormal computing status: {compute_status}. Pass client receiving.")
    
    update_status_history()
    status = 0

    if gpc.is_rank_for_log():
        end_ts = time.time()
        logger.info(
            f"finish try_interact_with_param_server, total cost: {end_ts-start_ts:.3f}, "
            f"client send cost: {end_client_send_ts-end_broadcast_update_push_status_ts:.3f} "
            f"client wait compute finish cost: {end_query_compute_status_ts-start_query_compute_status_ts:.3f} "
            f"client recv cost: {end_ts-end_query_compute_status_ts:.3f}"
        )


def client_send(model, consume_tokens, dynamic_config):
    send_state_dict = get_send_state_dict(model)
    dp_rank = gpc.get_local_rank(ParallelMode.DATA)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)

    send_status = 0
    if dp_rank == 0 and tp_rank == 0:
        assert len(send_state_dict) > 0
        # Send updates for each layer
        layer_idxs = list(send_state_dict.keys())
        random.shuffle(layer_idxs)
        logger.info(f"shuffle send layer idx: {layer_idxs}")

        for layer_id in layer_idxs:
            layer_state_dict = send_state_dict[layer_id]
            ps_id = get_ps_id(layer_id, dynamic_config.layer_chunks)
            if ps_id == -1:
                raise ValueError(f"Failed to find PS ID for layer {layer_id}.")
            ps_server = dynamic_config.zmq_servers[ps_id]
            send_status = client.send_update(ps_server, layer_id, layer_state_dict)
            if send_status == 1:
                break
        if gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
            dist.barrier(group=gpc.get_group(ParallelMode.PIPELINE))
            if gpc.is_last_rank(ParallelMode.PIPELINE):
                if send_status == 0:
                    # TODO(caikun): need retry to ensure success
                    client.update_push_status(PushStatus.PUSH_STATUS_FINISH, consume_tokens=consume_tokens)
                else:
                    client.update_push_status(PushStatus.PUSH_STATUS_FAIL, consume_tokens=consume_tokens)
                    logger.warning("Sending error. Skip sync with ps.")
        else:
            if send_status == 0:
                # TODO(caikun): need retry to ensure success
                client.update_push_status(PushStatus.PUSH_STATUS_FINISH, consume_tokens=consume_tokens)
            else:
                client.update_push_status(PushStatus.PUSH_STATUS_FAIL, consume_tokens=consume_tokens)
                logger.warning("Sending error. Skip sync with ps.")

    # check sending status
    status_tensor = torch.tensor(send_status, dtype=torch.int, device=get_current_device())
    dist.all_reduce(status_tensor, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.GLOBAL))
    send_error = status_tensor.item() > 0
    if send_error:
        return 1
    return 0


def query_compute_status_and_broadcast():
    dp_rank = gpc.get_local_rank(ParallelMode.DATA)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)

    is_rank_for_comm = True
    if gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
        is_rank_for_comm = gpc.is_last_rank(ParallelMode.PIPELINE)
    compute_status = ComputeStatus.COMPUTE_STATUS_UNKNOWN
    if dp_rank == 0 and tp_rank == 0 and is_rank_for_comm:
        while True:
            compute_status = client.query_compute_status()
            if compute_status == ComputeStatus.COMPUTE_SUCCESS:
                logger.info("ps compute weight successfully, ready to pull.")
                client.update_pull_status(PullStatus.PULL_STATUS_START)
                break
            elif compute_status == ComputeStatus.COMPUTE_FAIL:
                logger.error("ps compute weight fail.")
                break
            elif compute_status in (
                ComputeStatus.COMPUTING,
                ComputeStatus.RECEIVING,
                NormStatus.NORM_COMPUTING,
            ):
                logger.info(f"Waiting for updates, current status: {compute_status}")
            else:
                logger.error(f"Unknown status: {compute_status}")
                break
            time.sleep(1)

    # broadcast computing status
    object_list = [compute_status]
    if dp_rank == 0 and tp_rank == 0 and gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
        src = gpc.get_ranks_in_group(ParallelMode.PIPELINE)[-1]
        dist.broadcast_object_list(object_list, src=src, group=gpc.get_group(ParallelMode.PIPELINE))

    if dp_rank == 0:
        src = gpc.get_ranks_in_group(ParallelMode.TENSOR)[0]
        dist.broadcast_object_list(object_list, src=src, group=gpc.get_group(ParallelMode.TENSOR))

    src = gpc.get_ranks_in_group(ParallelMode.DATA)[0]
    dist.broadcast_object_list(object_list, src=src, group=gpc.get_group(ParallelMode.DATA))

    return object_list[0]


def client_recv(model, optimizer: torch.optim.Optimizer = None, request_for_ckpt: bool = False, dynamic_config=None):
    dp_rank = gpc.get_local_rank(ParallelMode.DATA)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    # Receive updates for each layer
    all_recv_state_dict = {}
    recv_status = 0
    if dp_rank == 0 and tp_rank == 0:
        for layer_id in range(model.first_layer, model.last_layer):
            ps_id = get_ps_id(layer_id, dynamic_config.layer_chunks)
            if ps_id == -1:
                raise ValueError(f"Failed to find PS ID for layer {layer_id}.")
            ps_server = dynamic_config.zmq_servers[ps_id]

            # try to recive layer statet dict
            recv_state_dict = None
            retry_time = 0
            while recv_state_dict is None and retry_time < 3:
                _, recv_state_dict = client.receive_updates(ps_server, layer_id)
                retry_time += 1
                time.sleep(0.5)

            if recv_state_dict is None:
                logger.error(
                    f"Error: Unsuccessfully received updates for layer {layer_id}. The recv_state_dict is None."
                )
                recv_status = 1
                break

            all_recv_state_dict.update(recv_state_dict)
            logger.info(f"Successfully received updates for layer {layer_id}.")

        if gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
            dist.barrier(group=gpc.get_group(ParallelMode.PIPELINE))
            if gpc.is_last_rank(ParallelMode.PIPELINE):
                if request_for_ckpt:
                    client.request_ckpt_status(PullStatus.PULL_STATUS_FINISH)
                else:
                    client.update_pull_status(PullStatus.PULL_STATUS_FINISH)
        else:
            if request_for_ckpt:
                client.request_ckpt_status(PullStatus.PULL_STATUS_FINISH)
            else:
                client.update_pull_status(PullStatus.PULL_STATUS_FINISH)

    status_tensor = torch.tensor(recv_status, dtype=torch.int, device=get_current_device())
    dist.all_reduce(status_tensor, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.GLOBAL))
    recv_error = status_tensor.item() > 0

    if gpc.is_rank_for_log():
        start_ts = time.time()

    if not recv_error:
        recover_local_state(model, optimizer, all_recv_state_dict)
    else:
        logger.error("recv_error occur. Skip recover_local_state.")

    if gpc.is_rank_for_log():
        end_ts = time.time()
        logger.info(f"recover_local_state cost: {end_ts-start_ts:.3f}")


def get_send_state_dict(model):
    if gpc.get_local_rank(ParallelMode.DATA) == 0:
        if gpc.is_rank_for_log():
            start_ts = time.time()
        state_dict = model.state_dict()
        dst = gpc.get_ranks_in_group(ParallelMode.TENSOR)[0]
        world_size = gpc.get_world_size(ParallelMode.TENSOR)

        # The state dict to be sent to PS
        send_state_dict = {}
        # global layer_id
        layer_id = model.first_layer
        send_state_dict[layer_id] = {}
        for fqn, tensor in state_dict.items():
            if "layer" in fqn:
                assert fqn in map_fqn_local_to_global
            global_fqn = map_fqn_local_to_global[fqn] if fqn in map_fqn_local_to_global else fqn
            complete_size = map_layer_attr[global_fqn]["complete_size"]

            if "layer" in global_fqn:
                if str(layer_id) not in global_fqn:
                    assert str(layer_id + 1) in global_fqn
                    layer_id += 1
                    send_state_dict[layer_id] = {}

            # find tp splited dim
            dim = -1
            for i in range(len(complete_size)):
                if complete_size[i] != tensor.shape[i]:
                    dim = i
                    break

            # dp0 tp0 gather from other tp ranks
            if dim >= 0:
                if gpc.get_global_rank() == dst:
                    gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
                else:
                    gather_list = None
                dist.gather(tensor, gather_list, dst=dst, group=gpc.get_group(ParallelMode.TENSOR))

            if gpc.get_global_rank() == dst:
                if dim >= 0:
                    # cat splited tensors into the complete tensor
                    complete_tensor = torch.cat(gather_list, dim=dim)
                else:
                    # For norm layer, it is not splitted by tp.
                    complete_tensor = tensor.clone()

                assert list(complete_size) == list(complete_tensor.shape)
                send_state_dict[layer_id][global_fqn] = complete_tensor.to("cpu")

        if gpc.is_rank_for_log():
            end_ts = time.time()
            logger.info(f"get_send_state_dict cost: {end_ts-start_ts:.3f}")
        return send_state_dict
    return {}


def recover_local_state(model, optimizer, recv_state_dict):
    if gpc.is_rank_for_log():
        logger.info("Begin to recover local state dict.")

    local_state_dict = model.state_dict()
    tp_src = gpc.get_ranks_in_group(ParallelMode.TENSOR)[0]
    dp_src = gpc.get_ranks_in_group(ParallelMode.DATA)[0]
    tp_world_size = gpc.get_world_size(ParallelMode.TENSOR)

    for fqn, tensor in local_state_dict.items():
        if "layer" in fqn:
            assert fqn in map_fqn_local_to_global
        global_fqn = map_fqn_local_to_global[fqn] if fqn in map_fqn_local_to_global else fqn
        complete_size = map_layer_attr[global_fqn]["complete_size"]

        recv_tensor = torch.empty_like(tensor)
        # dp0 tp0 scatter tensor list to other tp ranks
        if gpc.get_local_rank(ParallelMode.DATA) == 0:
            if gpc.get_global_rank() == tp_src:
                assert global_fqn in recv_state_dict

                # find tp split dim
                dim = -1
                for i in range(len(complete_size)):
                    if complete_size[i] != tensor.shape[i]:
                        dim = i
                        break

                complete_tensor = recv_state_dict[global_fqn]
                complete_tensor = complete_tensor.to(tensor.device).to(tensor.dtype)
                if dim >= 0:
                    # split complete_tensor into tp tensor list
                    scatter_list = list(torch.chunk(complete_tensor, tp_world_size, dim=dim))
                    for i in range(len(scatter_list)):
                        scatter_list[i] = scatter_list[i].contiguous()
                else:
                    # For norm layer, directly scatter
                    scatter_list = [complete_tensor] * tp_world_size
            else:
                scatter_list = None

            dist.scatter(recv_tensor, scatter_list, src=tp_src, group=gpc.get_group(ParallelMode.TENSOR))

        # dp0 broadcast to other dp ranks
        dist.broadcast(recv_tensor, src=dp_src, group=gpc.get_group(ParallelMode.DATA))
        tensor.data.copy_(recv_tensor.data)

    assert optimizer is not None
    if optimizer is not None:
        if gpc.is_rank_for_log():
            logger.info("Optimizer is not none. Begin to reload_zero_fp32_buff.")
        optimizer.reload_zero_fp32_buff()
    else:
        if gpc.is_rank_for_log():
            logger.info("Optimizer is none. Skip reload_zero_fp32_buff.")

    if gpc.is_rank_for_log():
        logger.info("Finish recovering local state dict.")


def client_recv_ckpt_status():
    dp_rank = gpc.get_local_rank(ParallelMode.DATA)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    ckpt_status = 0
    retry_print_flag = False
    while dp_rank == 0 and tp_rank == 0 and gpc.is_last_rank(ParallelMode.PIPELINE):
        op = client.request_ckpt_status(PullStatus.PULL_STATUS_START)
        if op == OperatorType.PULL:
            ckpt_status = 1
            break
        elif op == OperatorType.IGNORE:
            logger.warning("ignore pull weight from ps.")
            break
        elif op == OperatorType.RETRY:
            if not retry_print_flag:
                logger.info("retry to pull weight from ps.")
                retry_print_flag = True
            time.sleep(1)

    object_list = [ckpt_status]
    if dp_rank == 0 and tp_rank == 0 and gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
        src = gpc.get_ranks_in_group(ParallelMode.PIPELINE)[-1]
        dist.broadcast_object_list(object_list, src=src, group=gpc.get_group(ParallelMode.PIPELINE))

    if dp_rank == 0:
        src = gpc.get_ranks_in_group(ParallelMode.TENSOR)[0]
        dist.broadcast_object_list(object_list, src=src, group=gpc.get_group(ParallelMode.TENSOR))

    src = gpc.get_ranks_in_group(ParallelMode.DATA)[0]
    dist.broadcast_object_list(object_list, src=src, group=gpc.get_group(ParallelMode.DATA))

    return object_list[0]

from threading import Lock
lock = Lock()

def update_status_history():
    with lock:
        global current_status_begin_timestamp
        global status_history_list
        current_status_interval = dict(status_name = status, begin_timestamp = current_status_begin_timestamp, end_timestamp=time.time())
        current_status_begin_timestamp = current_status_interval["end_timestamp"]
        if len(status_history_list) != 0 and status_history_list[-1]["status_name"] == current_status_interval["status_name"]:
            current_status_interval["begin_timestamp"] = status_history_list[-1]["begin_timestamp"]
            status_history_list.pop()
        status_history_list.append(current_status_interval)
    
def get_current_status():
    with lock:
        global status
        return status

def get_history_status_list():
    with lock:
        global status_history_list
        return status_history_list