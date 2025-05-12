from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PushStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PUSH_STATUS_UNKNOWN: _ClassVar[PushStatus]
    PUSH_STATUS_START: _ClassVar[PushStatus]
    PUSH_STATUS_FINISH: _ClassVar[PushStatus]
    PUSH_STATUS_FAIL: _ClassVar[PushStatus]
    PUSH_STATUS_LOW_VERSION: _ClassVar[PushStatus]

class OperatorType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OPERATOR_TYPE_UNKNOWN: _ClassVar[OperatorType]
    PUSH: _ClassVar[OperatorType]
    RETRY: _ClassVar[OperatorType]
    IGNORE: _ClassVar[OperatorType]
    PULL: _ClassVar[OperatorType]
    FINISH: _ClassVar[OperatorType]

class ComputeStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COMPUTE_STATUS_UNKNOWN: _ClassVar[ComputeStatus]
    RECEIVING: _ClassVar[ComputeStatus]
    COMPUTE_SUCCESS: _ClassVar[ComputeStatus]
    COMPUTING: _ClassVar[ComputeStatus]
    COMPUTE_FAIL: _ClassVar[ComputeStatus]

class PullStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PULL_STATUS_UNKNOWN: _ClassVar[PullStatus]
    PULL_STATUS_START: _ClassVar[PullStatus]
    PULL_STATUS_FINISH: _ClassVar[PullStatus]
    PULL_STATUS_FAIL: _ClassVar[PullStatus]

class NormStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NORM_STATUS_UNKNOWN: _ClassVar[NormStatus]
    NORM_SUCCESS: _ClassVar[NormStatus]
    NORM_COMPUTING: _ClassVar[NormStatus]
    NORM_FAIL: _ClassVar[NormStatus]
PUSH_STATUS_UNKNOWN: PushStatus
PUSH_STATUS_START: PushStatus
PUSH_STATUS_FINISH: PushStatus
PUSH_STATUS_FAIL: PushStatus
PUSH_STATUS_LOW_VERSION: PushStatus
OPERATOR_TYPE_UNKNOWN: OperatorType
PUSH: OperatorType
RETRY: OperatorType
IGNORE: OperatorType
PULL: OperatorType
FINISH: OperatorType
COMPUTE_STATUS_UNKNOWN: ComputeStatus
RECEIVING: ComputeStatus
COMPUTE_SUCCESS: ComputeStatus
COMPUTING: ComputeStatus
COMPUTE_FAIL: ComputeStatus
PULL_STATUS_UNKNOWN: PullStatus
PULL_STATUS_START: PullStatus
PULL_STATUS_FINISH: PullStatus
PULL_STATUS_FAIL: PullStatus
NORM_STATUS_UNKNOWN: NormStatus
NORM_SUCCESS: NormStatus
NORM_COMPUTING: NormStatus
NORM_FAIL: NormStatus

class PSHeartbeatRequest(_message.Message):
    __slots__ = ("ps_id",)
    PS_ID_FIELD_NUMBER: _ClassVar[int]
    ps_id: int
    def __init__(self, ps_id: _Optional[int] = ...) -> None: ...

class PSHeartbeatResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ClientHeartbeatRequest(_message.Message):
    __slots__ = ("group_id",)
    GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    group_id: int
    def __init__(self, group_id: _Optional[int] = ...) -> None: ...

class ClientHeartbeatResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class UpdatePushStatusRequest(_message.Message):
    __slots__ = ("group_id", "push_status", "consume_tokens", "version")
    GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    PUSH_STATUS_FIELD_NUMBER: _ClassVar[int]
    CONSUME_TOKENS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    group_id: int
    push_status: PushStatus
    consume_tokens: int
    version: int
    def __init__(self, group_id: _Optional[int] = ..., push_status: _Optional[_Union[PushStatus, str]] = ..., consume_tokens: _Optional[int] = ..., version: _Optional[int] = ...) -> None: ...

class UpdatePushStatusResponse(_message.Message):
    __slots__ = ("op",)
    OP_FIELD_NUMBER: _ClassVar[int]
    op: OperatorType
    def __init__(self, op: _Optional[_Union[OperatorType, str]] = ...) -> None: ...

class UpdateComputeStatusRequest(_message.Message):
    __slots__ = ("ps_id", "compute_status")
    PS_ID_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_STATUS_FIELD_NUMBER: _ClassVar[int]
    ps_id: int
    compute_status: ComputeStatus
    def __init__(self, ps_id: _Optional[int] = ..., compute_status: _Optional[_Union[ComputeStatus, str]] = ...) -> None: ...

class UpdateComputeStatusResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class QueryComputeStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class QueryComputeStatusResponse(_message.Message):
    __slots__ = ("compute_status",)
    COMPUTE_STATUS_FIELD_NUMBER: _ClassVar[int]
    compute_status: ComputeStatus
    def __init__(self, compute_status: _Optional[_Union[ComputeStatus, str]] = ...) -> None: ...

class UpdatePullStatusRequest(_message.Message):
    __slots__ = ("group_id", "pull_status")
    GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    PULL_STATUS_FIELD_NUMBER: _ClassVar[int]
    group_id: int
    pull_status: PullStatus
    def __init__(self, group_id: _Optional[int] = ..., pull_status: _Optional[_Union[PullStatus, str]] = ...) -> None: ...

class UpdatePullStatusResponse(_message.Message):
    __slots__ = ("version",)
    VERSION_FIELD_NUMBER: _ClassVar[int]
    version: int
    def __init__(self, version: _Optional[int] = ...) -> None: ...

class UpdateClientCkptRequest(_message.Message):
    __slots__ = ("group_id", "pull_status")
    GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    PULL_STATUS_FIELD_NUMBER: _ClassVar[int]
    group_id: int
    pull_status: PullStatus
    def __init__(self, group_id: _Optional[int] = ..., pull_status: _Optional[_Union[PullStatus, str]] = ...) -> None: ...

class UpdateClientCkptResponse(_message.Message):
    __slots__ = ("op", "version")
    OP_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    op: OperatorType
    version: int
    def __init__(self, op: _Optional[_Union[OperatorType, str]] = ..., version: _Optional[int] = ...) -> None: ...

class UpdateClientWeightFactorRequest(_message.Message):
    __slots__ = ("group_id", "factor")
    GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    FACTOR_FIELD_NUMBER: _ClassVar[int]
    group_id: int
    factor: float
    def __init__(self, group_id: _Optional[int] = ..., factor: _Optional[float] = ...) -> None: ...

class UpdateClientWeightFactorResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class UpdateNormStatusRequest(_message.Message):
    __slots__ = ("ps_id", "norm_status")
    PS_ID_FIELD_NUMBER: _ClassVar[int]
    NORM_STATUS_FIELD_NUMBER: _ClassVar[int]
    ps_id: int
    norm_status: NormStatus
    def __init__(self, ps_id: _Optional[int] = ..., norm_status: _Optional[_Union[NormStatus, str]] = ...) -> None: ...

class UpdateNormStatusResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GroupNorm(_message.Message):
    __slots__ = ("group_id", "norm")
    GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    NORM_FIELD_NUMBER: _ClassVar[int]
    group_id: int
    norm: float
    def __init__(self, group_id: _Optional[int] = ..., norm: _Optional[float] = ...) -> None: ...

class UpdatePSNormRequest(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[GroupNorm]
    def __init__(self, items: _Optional[_Iterable[_Union[GroupNorm, _Mapping]]] = ...) -> None: ...

class UpdatePSNormResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HistoryGlobalStatus(_message.Message):
    __slots__ = ("status_name", "start_time", "end_time")
    STATUS_NAME_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    status_name: str
    start_time: str
    end_time: str
    def __init__(self, status_name: _Optional[str] = ..., start_time: _Optional[str] = ..., end_time: _Optional[str] = ...) -> None: ...

class CurrentGlobalStatus(_message.Message):
    __slots__ = ("status_name",)
    STATUS_NAME_FIELD_NUMBER: _ClassVar[int]
    status_name: str
    def __init__(self, status_name: _Optional[str] = ...) -> None: ...

class QueryGlobalStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class QueryGlobalStatusResponse(_message.Message):
    __slots__ = ("current_global_status", "history_global_status_list")
    CURRENT_GLOBAL_STATUS_FIELD_NUMBER: _ClassVar[int]
    HISTORY_GLOBAL_STATUS_LIST_FIELD_NUMBER: _ClassVar[int]
    current_global_status: CurrentGlobalStatus
    history_global_status_list: _containers.RepeatedCompositeFieldContainer[HistoryGlobalStatus]
    def __init__(self, current_global_status: _Optional[_Union[CurrentGlobalStatus, _Mapping]] = ..., history_global_status_list: _Optional[_Iterable[_Union[HistoryGlobalStatus, _Mapping]]] = ...) -> None: ...
