import argparse
import os
import sys

sys.path.append(".")

from loguru import logger
import torch
from internlm.param_server.common import config
from internlm.param_server.common.initialize import initialize_optimizer, load_model
from internlm.param_server.common.utils import get_last_ckpt_path, load_optimizer_ckpt
from internlm.param_server.ps.server import ParameterServer

parser = argparse.ArgumentParser()
parser.add_argument("--ps_id", type=int, help="ps id for the server")


def main(args):
    ps_id = args.ps_id
    num_layers = config.NUM_LAYERS
    layer_chunks = config.layer_chunks
    param_shapes = config.param_shapes
    local_chunk = layer_chunks[ps_id]
    
    
    local_state_dict = {}
    for param, shape in param_shapes.items():
        key = f'layers.{local_chunk[0]}.' + param
        local_state_dict[key] = torch.rand(shape, dtype=torch.float32)
    
    for layer_id in local_chunk[1:]:
        for param, shape in param_shapes.items():
            key = f'layers.{layer_id}.' + param
            fisrt_key = f'layers.{local_chunk[0]}.' + param
            local_state_dict[key] = local_state_dict[fisrt_key].clone()

    
    

    # logger.info(f"load model from {load_ckpt_path}")
    # local_state_dict = load_model(load_ckpt_path, num_layers, local_chunk, param_shapes)

    # optimizer = initialize_optimizer(config.optimizer.get("name"), config.optimizer, local_state_dict)
    # if last_ckpt_path is not None:
    #     load_optimizer_ckpt(optimizer, last_ckpt_path)

    ps = ParameterServer(
        ps_id,
        num_layers,
        local_state_dict,
        None,
        config.model.get("dtype"),
        local_chunk,
        grpc_port=config.GRPC_PORT,
        zmq_port=config.ZMQ_PORT,
        master_address=config.master_server,
        heartbeat_interval=config.HEARTBEAT_INTERVAL,
        only_communicate=True,
    )

    ps.start()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
