from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GroupWeightFactor(_message.Message):
    __slots__ = ("group_id", "norm")
    GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    NORM_FIELD_NUMBER: _ClassVar[int]
    group_id: int
    norm: float
    def __init__(self, group_id: _Optional[int] = ..., norm: _Optional[float] = ...) -> None: ...

class StartUpdateWeightRequest(_message.Message):
    __slots__ = ("groups", "factors", "version", "consume_tokens", "save_ckpt", "clip_coef")
    GROUPS_FIELD_NUMBER: _ClassVar[int]
    FACTORS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    CONSUME_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SAVE_CKPT_FIELD_NUMBER: _ClassVar[int]
    CLIP_COEF_FIELD_NUMBER: _ClassVar[int]
    groups: _containers.RepeatedScalarFieldContainer[int]
    factors: _containers.RepeatedCompositeFieldContainer[GroupWeightFactor]
    version: int
    consume_tokens: int
    save_ckpt: bool
    clip_coef: float
    def __init__(self, groups: _Optional[_Iterable[int]] = ..., factors: _Optional[_Iterable[_Union[GroupWeightFactor, _Mapping]]] = ..., version: _Optional[int] = ..., consume_tokens: _Optional[int] = ..., save_ckpt: bool = ..., clip_coef: _Optional[float] = ...) -> None: ...

class StartUpdateWeightResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class UpdateGroupListRequest(_message.Message):
    __slots__ = ("groups",)
    GROUPS_FIELD_NUMBER: _ClassVar[int]
    groups: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, groups: _Optional[_Iterable[int]] = ...) -> None: ...

class UpdateGroupListResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StartComputeNormRequest(_message.Message):
    __slots__ = ("groups",)
    GROUPS_FIELD_NUMBER: _ClassVar[int]
    groups: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, groups: _Optional[_Iterable[int]] = ...) -> None: ...

class StartComputeNormResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class FinishPullWeightRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class FinishPullWeightResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
