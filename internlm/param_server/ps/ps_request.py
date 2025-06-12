from typing import Type, Tuple, TypeVar, List, Literal

from pydantic import BaseModel


T = TypeVar("T", bound='BaseModel')


class PSBaseModel(BaseModel):
    def to_bytes(self) -> bytes:
        return self.model_dump_json().encode('utf-8')

    @classmethod
    def from_bytes(cls: Type[T], data: bytes) -> T:
        return cls.model_validate_json(data.decode('utf-8'))


class PSRequestHeader(PSBaseModel):
    client_id: str
    group_id: int


class RDMAConnectionInfo(PSBaseModel):
    hash_id: Tuple[str, int, int, str] # client id, group id, global_rank, zmq_server
    rdma_info: str


class TensorInfo(PSBaseModel):
    key: str
    dtype: Literal["torch.float32", "torch.bfloat16"]
    shape_info: List[int]


class LayerInfo(PSBaseModel):
    layer_id: int
    tensor_info: List[TensorInfo]


class RDMAOneSideRequest(PSBaseModel):
    opcode: Literal["READ", "WRITE"]
    connection_info: RDMAConnectionInfo
    layer_info: List[LayerInfo]


