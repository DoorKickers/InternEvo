import sys
sys.path.append(".")
import time

import torch
import threading
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.param_server.client.client import client, get_ps_id
from internlm.param_server.common.config import layer_chunks, zmq_servers

from internlm.utils.logger import get_logger
from internlm.param_server.common.config import master_server, param_shapes
import argparse
import os

logger = get_logger(__file__)


def try_interact_with_param_server(args):
    
    first_layer, last_layer, pp_rank, pp_world_size, group_id, send_state_dict = args
    
    need_heartbeat = True if pp_rank == pp_world_size - 1 else False

    client.start(group_id, master_server, need_heartbeat)

    client_send(send_state_dict)
    print(f"finish client send, ready to client recv, group_id={group_id}", flush=True)
    client_recv(first_layer, last_layer)
    
    end = time.time()
    print(f"send_recv {gpc.get_local_rank(ParallelMode.PIPELINE)}: "
            + f"total_time={end - begin_send}, "
            + f"send_only={finish_send - begin_send}, "
            + f"recv_only={finish_recv - begin_recv}, ", flush=True)


def client_send(send_state_dict):
    
    global begin_send, finish_send
    begin_send = time.time()
    assert len(send_state_dict) > 0
    print(f"local_state_dict len: {len(send_state_dict)}, {send_state_dict.keys()}", flush=True)
    # Send updates for each layer
    for layer_id, layer_state_dict in send_state_dict.items():
        ps_id = get_ps_id(layer_id, layer_chunks)
        # ps_id = 0
        print(f"route: {layer_id}, {ps_id}, {layer_chunks}", flush=True)
        if ps_id == -1:
            raise ValueError(f"Failed to find PS ID for layer {layer_id}.")
        ps_server = zmq_servers[ps_id]
        client.send_update(ps_server, layer_id, layer_state_dict)
    finish_send = time.time()


def client_recv(first_layer, last_layer):
    global begin_recv, finish_recv
    begin_recv = time.time()
    # Receive updates for each layer
    for layer_id in range(first_layer, last_layer):
        ps_id = get_ps_id(layer_id, layer_chunks)
        print(f"route: {layer_id}, {ps_id}, {layer_chunks}", flush=True)
        if ps_id == -1:
            raise ValueError(f"Failed to find PS ID for layer {layer_id}.")
        ps_server = zmq_servers[ps_id]
        # layer statet dict
        _, recv_state_dict = client.receive_updates(ps_server, layer_id)
        assert recv_state_dict is not None, f"Failed to receive updates for layer {layer_id}."
        logger.info(f"Successfully received updates for layer {layer_id}.")
    finish_recv = time.time()


def init_state_dict(first_layer, last_layer):
    send_state_dict = {}
    send_state_dict[first_layer] = {}
    for param, shape in param_shapes.items():
        key = f'layers.{first_layer}.' + param
        send_state_dict[first_layer][key] = torch.rand(shape, dtype=torch.bfloat16)
    
    for layer_id in range(first_layer + 1, last_layer):
        send_state_dict[layer_id] = {}
        for param, shape in param_shapes.items():
            key = f'layers.{layer_id}.' + param
            fisrt_key = f'layers.{first_layer}.' + param
            send_state_dict[layer_id][key] = send_state_dict[first_layer][fisrt_key].clone()
    
    return send_state_dict


def test_interact():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--group_id", type=int, default=0)
    parser.add_argument("--pp_world_size", type=int, default=8)
    parser.add_argument("--pp_rank", type=int, default=0)
    args = parser.parse_args()
    
    # SLURM_NPROCS
    # pp_world_size = int(os.environ['SLURM_NPROCS'])
    # pp_rank = int(os.environ['SLURM_PROCID'])
    pp_world_size = args.pp_world_size
    
    threads = []
    for i in range(1):
        pp_rank = args.pp_rank+i
        print(f"pp_rank:", pp_rank, pp_world_size, flush=True)
        send_state_dict = init_state_dict(pp_rank * 2, pp_rank * 2 + 2)
        thread = threading.Thread(target=try_interact_with_param_server, args=([pp_rank * 2, pp_rank * 2 + 2, pp_rank, pp_world_size, args.group_id, send_state_dict],))
        threads.append(thread)
        thread.start()
        
    for thread in threads:
        thread.join()
    print("all threads finished.")
    # try_interact_with_param_server([pp_rank * 10, pp_rank * 10 + 10, pp_rank, pp_world_size, args.group_id, send_state_dict])


    # ctx = mp.get_context("spawn")
    # with ctx.Pool(processes=8) as pool:
    #     pool.map(
    #         try_interact_with_param_server,
    #         [[pp_rank * 1, pp_rank * 1 + 1, pp_rank, args.group_id] for pp_rank in range(8)],
    #     )
    #     pool.close()
    #     pool.join()


if __name__ == "__main__":
    test_interact()