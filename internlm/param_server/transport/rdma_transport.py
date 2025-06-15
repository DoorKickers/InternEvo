import json
from typing import Dict, List, Optional, Tuple

import dlslime

import torch


class RDMAEndpointContext:
    def __init__(self):
        self.rdma_endpoints: Dict[Tuple[str, int, int], dlslime.RDMAEndpoint] =  {}
        self.devices = dlslime.available_nic()
        self.rr_idx = 0

    def create(self, worker_id:str, group_id:int, rank_id:int, ps_server: Optional[str] = None):
        tmp_rr_idx = self.rr_idx
        self.rr_idx += 1
        endpoint = dlslime.RDMAEndpoint(self.devices[tmp_rr_idx % len(self.devices)])
        self.rdma_endpoints[(worker_id, group_id, rank_id, ps_server)] = endpoint
        return endpoint

    def get(self, worker_id:str, group_id:int, rank_id:int, ps_server: Optional[str] = None):
        return self.rdma_endpoints[(worker_id, group_id, rank_id, ps_server)]

    def connect(self):
        raise NotImplementedError

    def register(self):
        raise NotImplementedError


_RDMA_ENDPOINT_CTX = RDMAEndpointContext()
def rdma_endpoint_ctx():
    global _RDMA_ENDPOINT_CTX
    return _RDMA_ENDPOINT_CTX


def endpoint_info_serialize(endpoint:dlslime.RDMAEndpoint):
    return json.dumps(endpoint.endpoint_info).encode("utf-8")


def endpoint_info_deserialize(b: bytes):
    return json.loads(b.decode("utf-8"))


def register_memory_region(endpoint:dlslime.RDMAEndpoint, state_dict: Dict[str, torch.Tensor]):
    for k, t in state_dict.items():
        endpoint.register_memory_region(k, t.data_ptr(), t.storage_offset(), t.numel() * t.itemsize)

