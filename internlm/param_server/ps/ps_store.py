import gc
from collections import namedtuple
from typing import Dict, Literal, Optional

import json

from loguru import logger

import dlslime

import torch

from internlm.param_server.transport.rdma_transport import rdma_endpoint_ctx
from internlm.param_server.ps.ps_request import RDMAConnectionInfo, LayerInfo


BufferKey = namedtuple("BufferKey", ["group_id", "layer_id"])


class BufferManager:
    """
    Manages buffers for storing and retrieving compressed and serialized data chunks.
    """

    def __init__(self, model_state_dict: Dict[int, Dict[str, torch.Tensor]], origin_dtype: torch.dtype, global_num_layers: int):
        """
        Initializes the BufferManager.
        """
        self.state_dict: Dict[BufferKey, Dict[str, torch.Tensor]] = {}
        self.model_state_dict: Dict[int, Dict[str, torch.Tensor]] = model_state_dict
        self.origin_dtype = origin_dtype
        self.global_num_layers = global_num_layers

    def get_buffer_key(self, group_id: int, layer_id: int) -> BufferKey:
        """
        Generates a unique key for the buffer based on group and layer IDs.

        Args:
            group_id (int): Group ID.
            layer_id (int): Layer ID.

        Returns:
            str: A unique string key for the buffer.
        """
        return BufferKey(group_id, layer_id)

    def add_layer_state_dict(self, group_id: int, layer_id: int, layer_state_dict: Dict[str, torch.Tensor]):
        buffer_key = self.get_buffer_key(group_id, layer_id)
        self.state_dict[buffer_key] = layer_state_dict

    def get_weight(self, group_id: int, layer_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieves the state dictionary associated with the buffer.

        Args:
            group_id (int): The group ID of the buffer.
            layer_id (int): The layer ID of the buffer.

        Returns:
            Optional[Dict]: The state dictionary if available, else None.
        """
        buffer_key = self.get_buffer_key(group_id=group_id, layer_id=layer_id)
        return self.state_dict.get(buffer_key, None)

    def destroy(self):
        self.state_dict.clear()
        n = gc.collect()
        logger.info(f"destroy layer state dict and collect {n} python objects")

    def rdma_oneside(
        self,
        opcode: Literal["READ", "WRITE"],
        group_id,
        endpoint: dlslime.RDMAEndpoint,
        rdma_connection_info: RDMAConnectionInfo, 
        layer_info: LayerInfo,
        async_op=False
    ):
        logger.info(f"RDMA {opcode} for layer {layer_info.layer_id}")
        layer_id = layer_info.layer_id
        tensor_info = layer_info.tensor_info
        rdma_info = json.loads(rdma_connection_info.rdma_info)
        mr_info = rdma_info["mr_info"] or {}
        for mr_key, remote_mr_info in mr_info.items():
            endpoint.register_remote_memory_region(mr_key, remote_mr_info)
        if opcode == "READ":
            fn = endpoint.read_batch
            layer_dict = self.get_weight(group_id, layer_id)
        else:
            fn = endpoint.write_batch
            layer_dict = self.get_layer_state_dict(layer_id)
        batch = []
        for t_info in tensor_info:
            mr_key = t_info.key
            t = layer_dict[t_info.key]
            assert list(t.shape) == list(t_info.shape_info), ("PS Server t.shape != Client Reuqest's t_info.shape_info, "
                                                              f"{mr_key=}, {t.shape=}, {t_info.shape_info=}")
            assert str(t.dtype) == t_info.dtype, ("PS Server t.dtype != Client Reuqest's t_info.dtype, "
                                                  f"{mr_key=}, {t.dtype=}, {t_info.dtype=}")
            endpoint.register_memory_region(mr_key, t.data_ptr(), t.storage_offset(), t.numel() * t.itemsize)
            batch.append(dlslime.Assignment(mr_key=mr_key, target_offset=0, source_offset=0, length=t.numel() * t.itemsize))

        return fn(batch, async_op=async_op)

    def allocate_and_rdma_register(self,
        group_id: int,
        endpoint: dlslime.RDMAEndpoint,
        layer_info: LayerInfo
    ):
        layer_id = layer_info.layer_id
        tensor_info = layer_info.tensor_info
        layer_state_dict = {}
        old_layer_state_dict = self.get_weight(group_id, layer_id) or {}
        for t_info in tensor_info:
            key = t_info.key
            dtype = eval(t_info.dtype)
            shape_info = t_info.shape_info
            if key in old_layer_state_dict:
                # logger.info(f"{group_id, layer_id, key} cache hit")
                t = old_layer_state_dict[key]
            else:
                t = torch.empty(shape_info, dtype=dtype)
            endpoint.register_memory_region(key, t.data_ptr(), t.storage_offset(), t.numel() * t.itemsize)
            layer_state_dict[key] = t
        self.add_layer_state_dict(group_id, layer_id, layer_state_dict)

    def get_layer_state_dict(self, layer_id: int) -> Dict[str, torch.Tensor]:
        # TODO: Critical Path
        layer_state_dict = {}
        for key, value in self.model_state_dict.items():
            dtype = self.origin_dtype
            if "feed_forward.moe_layer.gate.wg.weight" in key:
                dtype = torch.float32
            if key.startswith("layers"):
                layer_idx = int(key.split(".")[1])
                if layer_idx == layer_id:
                    if dtype == value.dtype:
                        value = value.clone()
                    layer_state_dict[key] = value.to(dtype)
            else:
                if layer_id == 0 and (key.startswith("tok_embeddings") or key.startswith("embed_tokens")):
                    layer_state_dict[key] = value.to(dtype)
                elif layer_id == self.global_num_layers - 1 and (key.startswith("norm") or key.startswith("output")):
                    layer_state_dict[key] = value.to(dtype)
        return layer_state_dict


