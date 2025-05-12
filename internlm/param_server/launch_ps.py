import argparse
import os
import sys

sys.path.append(".")

from loguru import logger

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
    auto_resume = config.ckpt.get("auto_resume")
    load_ckpt_path = config.ckpt.get("load_ckpt_path")
    save_ckpt_path = config.ckpt.get("save_ckpt_path")
    last_ckpt_path = None
    if auto_resume:
        last_ckpt_path = get_last_ckpt_path(save_ckpt_path, ps_id)
    assert last_ckpt_path is not None or load_ckpt_path is not None, "No checkpoint found to load"

    if last_ckpt_path is not None:
        logger.info(f"auto resume from {last_ckpt_path}")
        load_ckpt_path = os.path.join(last_ckpt_path, "model.pt")
    local_chunk = layer_chunks[ps_id]

    logger.info(f"load model from {load_ckpt_path}")
    local_state_dict = load_model(load_ckpt_path, num_layers, local_chunk, param_shapes)

    optimizer = initialize_optimizer(config.optimizer.get("name"), config.optimizer, local_state_dict)
    if last_ckpt_path is not None:
        load_optimizer_ckpt(optimizer, last_ckpt_path)

    ps = ParameterServer(
        ps_id,
        num_layers,
        local_state_dict,
        optimizer,
        config.model.get("dtype"),
        local_chunk,
        grpc_port=config.GRPC_PORT,
        zmq_port=config.ZMQ_PORT,
        master_address=config.master_server,
        heartbeat_interval=config.HEARTBEAT_INTERVAL,
        save_ckpt_path=save_ckpt_path,
    )

    ps.start()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
