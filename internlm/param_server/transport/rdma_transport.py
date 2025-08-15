import json
from typing import Dict, List, Optional, Tuple, Union

import torch

try:
    from internlm.param_server.common.config import USE_DLSLIME_RDMA_TRANSFER
    if USE_DLSLIME_RDMA_TRANSFER:
        import dlslime
        DLSLIME_AVAILABLE = True
    else:
        DLSLIME_AVAILABLE = False
        dlslime = None
except ImportError:
    USE_DLSLIME_RDMA_TRANSFER = False
    DLSLIME_AVAILABLE = False
    dlslime = None


class RDMAEndpointContext:
    def __init__(self):
        if not DLSLIME_AVAILABLE:
            raise RuntimeError("dlslime is not available. Set USE_DLSLIME_RDMA_TRANSFER=True to enable RDMA functionality.")
        
        self.rdma_endpoints: Dict[Tuple[str, int, int], Union[dlslime.RDMAEndpoint, None]] =  {}
        self.devices = dlslime.available_nic()
        self.rr_idx = 0

    def create(self, worker_id:str, group_id:int, rank_id:int, ps_server: Optional[str] = None):
        if not DLSLIME_AVAILABLE:
            raise RuntimeError("dlslime is not available. Set USE_DLSLIME_RDMA_TRANSFER=True to enable RDMA functionality.")
        
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


_RDMA_ENDPOINT_CTX = None
def rdma_endpoint_ctx():
    global _RDMA_ENDPOINT_CTX
    if _RDMA_ENDPOINT_CTX is None:
        if DLSLIME_AVAILABLE:
            _RDMA_ENDPOINT_CTX = RDMAEndpointContext()
        else:
            _RDMA_ENDPOINT_CTX = None
    return _RDMA_ENDPOINT_CTX


def endpoint_info_serialize(endpoint):
    if not DLSLIME_AVAILABLE:
        raise RuntimeError("dlslime is not available. Set USE_DLSLIME_RDMA_TRANSFER=True to enable RDMA functionality.")
    
    return json.dumps(endpoint.endpoint_info).encode("utf-8")


def endpoint_info_deserialize(b: bytes):
    if not DLSLIME_AVAILABLE:
        raise RuntimeError("dlslime is not available. Set USE_DLSLIME_RDMA_TRANSFER=True to enable RDMA functionality.")
    
    return json.loads(b.decode("utf-8"))


def register_memory_region(endpoint, state_dict: Dict[str, torch.Tensor]):
    if not DLSLIME_AVAILABLE:
        raise RuntimeError("dlslime is not available. Set USE_DLSLIME_RDMA_TRANSFER=True to enable RDMA functionality.")
    
    for k, t in state_dict.items():
        endpoint.register_memory_region(k, t.data_ptr(), t.storage_offset(), t.numel() * t.itemsize)

