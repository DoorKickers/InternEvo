import ctypes
import os
from typing import Dict, List, Optional
import time
from collections import namedtuple

import grpc
import torch
from loguru import logger

from internlm.param_server.common.config import param_shapes
from internlm.param_server.common.initialize import check_state_dict

# Maximum chunk size set to 512KB
MAX_CHUNK_SIZE = 512 * 1024


# Create a custom wrapper with timeout for the stub
class CustomServiceStub:
    def __init__(self, stub, timeout):
        self._stub = stub
        self._timeout = timeout

    def __getattr__(self, name):
        # Wrap each RPC method to apply the timeout
        rpc_method = getattr(self._stub, name)

        def wrapper(request, *args, **kwargs):
            kwargs["timeout"] = self._timeout
            return rpc_method(request, *args, **kwargs)

        return wrapper


def handle_value_error(method):
    def wrapper(self, request, context):
        try:
            return method(self, request, context)
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            raise

    return wrapper


def compress(chunk: bytes) -> bytes:
    """
    Compresses a given byte chunk using Blosc2 with predefined settings.

    Args:
        chunk (bytes): The data to compress.

    Returns:
        bytes: The compressed data.
    """
    return chunk


def decompress(chunk: bytes) -> bytes:
    """
    Decompresses a given byte chunk using Blosc2.

    Args:
        chunk (bytes): The compressed data.

    Returns:
        bytes: The decompressed data.
    """
    return chunk


def list_to_str(items: List[int]) -> str:
    return ','.join(str(item) for item in items)


def str_to_list(s: str) -> List[int]:
    items = s.split(",")
    return [int(item) for item in items]


def serialize(tensor: torch.Tensor) -> bytes:
    start_ts = time.time()
    address = tensor.data_ptr()
    length = tensor.nbytes
    byte_array_type = ctypes.c_byte * length
    ptr = ctypes.cast(address, ctypes.POINTER(byte_array_type))
    byte_array = bytes(ptr.contents)
    end_ts = time.time()
    # logger.info(f"serialize success, cost: {end_ts-start_ts:.3f}, length: {length}")
    return byte_array


def deserialize(data: bytes, shape: List[int], dtype=torch.bfloat16) -> torch.Tensor:
    start_ts = time.time()
    recovered_tensor = torch.frombuffer(data, dtype=dtype).reshape(shape)
    end_ts = time.time()
    # logger.info(f"deserialize success, cost: {end_ts-start_ts:.3f}. length: {len(data)}")
    return recovered_tensor


def serialize_layer(state_dict: Dict[str, torch.Tensor]) -> List[bytes]:
    """
    serialize layer

    Args:
        state_dict(Dict[str, torch.Tensor]): layer weight

    Returns:
        List[bytes]: data_size, key_num, chunk_lens_of_one_key, keys, shapes, chunks
    """
    start_ts = time.time()
    data_size = 0
    keys = []
    shapes = []
    chunks = []
    chunk_lens = []

    for key, value in state_dict.items():
        serialized_state = serialize(value)
        serialized_parts = [
            serialized_state[i : i + MAX_CHUNK_SIZE] for i in range(0, len(serialized_state), MAX_CHUNK_SIZE)
        ]
        data_size += len(serialized_state)
        chunks += serialized_parts
        keys.append(key.encode("utf-8"))
        shapes.append(list_to_str(value.shape).encode("utf-8"))
        chunk_lens.append(len(serialized_parts))

    key_num = len(state_dict)
    result = [str(data_size).encode("utf-8"), str(key_num).encode("utf-8"), list_to_str(chunk_lens).encode("utf-8")] + keys + shapes + chunks
    end_ts = time.time()
    logger.info(f"serialize_layer success, cost: {end_ts-start_ts:.3f}. length: {data_size}")
    return result


def deserialize_layer(group_id: int, layer_id: int, message_parts: List[bytes]) -> Dict[str, torch.Tensor]:
    start_ts = time.time()
    if len(message_parts) < 3:
        logger.warning(f"Malformed PUT request, {group_id=} {layer_id=}")
        return None

    try:
        data_size = int(message_parts[0].decode("utf-8"))
        key_num = int(message_parts[1].decode("utf-8"))
        chunk_lens = str_to_list(message_parts[2].decode("utf-8"))
        chunk_num = 3 + 2*key_num + sum(chunk_lens)
        if len(message_parts) != chunk_num:
            raise ValueError(f"invalid PUT request, expect {chunk_num=} parameters, but got {len(message_parts)=}")
        keys = [key.decode("utf-8") for key in message_parts[3:3+key_num]]
        shapes = [str_to_list(shape_str.decode("utf-8")) for shape_str in message_parts[3+key_num:3+key_num*2]]
    except ValueError as e:
        logger.warning(f"Invalid PUT parameters from {group_id=} {layer_id=} {e}")
        return None

    start_idx = 3 + key_num*2
    total_len = 0
    state_dict = {}
    for idx in range(key_num):
        key = keys[idx]
        shape = shapes[idx]
        chunk_len = chunk_lens[idx]
        tensor_data = b"".join(message_parts[start_idx : start_idx+chunk_len])
        start_idx += chunk_len
        total_len += len(tensor_data)
        try:
            dtype = torch.bfloat16
            if "feed_forward.moe_layer.gate.wg.weight" in key:
                dtype = torch.float32
            state_dict[key] = deserialize(tensor_data, shape, dtype)
        except Exception as e:
            logger.error(f"deserialize fail, {shape=} {len(tensor_data)=} {key=}")
            raise

    if data_size != total_len:
        logger.warning(
            f"Received incomplete state_dict from {group_id=} {layer_id=}, "
            f"Expected {data_size} bytes, got {total_len} bytes.")
        return None

    check_state_dict(state_dict, param_shapes)
    end_ts = time.time()
    logger.info(f"deserialize_layer success, cost: {end_ts-start_ts:.3f}. length: {data_size}")
    return state_dict


def save_state_dict(state_dict, path, name):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{name}")
    torch.save(state_dict, file_path)


def save_ps_checkpoint(model_state_dict: Dict, optimizer: torch.optim.Optimizer, save_path, version):
    os.makedirs(save_path, exist_ok=True)
    meta_file = os.path.join(save_path, "version.txt")

    with open(meta_file, "w") as f:
        f.write(f"{version}")

    model_name = "model.pt"
    save_state_dict(model_state_dict, save_path, model_name)

    optim_name = "optim_state.pt"
    save_state_dict(optimizer.state_dict(), save_path, optim_name)


def get_last_ckpt_path(ckpt_path, ps_id):
    snapshot_path = os.path.join(ckpt_path, f"ps_{ps_id}", "snapshot")
    if not os.path.exists(snapshot_path):
        return None
    last_version = 0
    last_snapshot = "0"
    snapshots = os.listdir(snapshot_path)
    if len(snapshots) == 0:
        return None
    for snapshot_counter in snapshots:
        with open(os.path.join(snapshot_path, snapshot_counter, "version.txt"), "r") as f:
            version = f.read().strip()
            version = int(version)
            if version > last_version:
                last_version = version
                last_snapshot = snapshot_counter
    return os.path.join(snapshot_path, last_snapshot)


def load_optimizer_ckpt(optimizer: torch.optim.Optimizer, ckpt_path):
    optimizer_ckpt_path = os.path.join(ckpt_path, "optim_state.pt")
    optimizer_state = torch.load(optimizer_ckpt_path, map_location="cpu")
    optimizer.load_state_dict(optimizer_state)
    logger.info(f"Load optimizer checkpoint from {optimizer_ckpt_path}.")
