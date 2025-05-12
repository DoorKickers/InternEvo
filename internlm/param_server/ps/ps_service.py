from threading import Lock

from loguru import logger

from internlm.param_server.common.utils import handle_value_error
from internlm.param_server.proto import ps_pb2, ps_pb2_grpc
from internlm.param_server.proto.master_pb2 import ComputeStatus, NormStatus


class PSControlServicer(ps_pb2_grpc.ControlServiceServicer):
    def __init__(self, parameter_server):
        self.parameter_server = parameter_server
        self.lock = Lock()
        self.group_ids = []

    @handle_value_error
    def StartUpdateWeight(self, request: ps_pb2.StartUpdateWeightRequest, context):
        logger.info(
            f"receive StartUpdateWeight, group_ids={request.groups}, version={request.version}, "
            f"save_ckpt={request.save_ckpt}, consume_tokens={request.consume_tokens}"
        )
        with self.parameter_server.lock:
            self.parameter_server.recved_groups = list(request.groups)
            self.parameter_server.compute_status = ComputeStatus.COMPUTING
            self.parameter_server.is_time_to_save_ckpt = request.save_ckpt
            self.parameter_server.version = request.version
            self.parameter_server.consume_tokens = request.consume_tokens
            self.parameter_server.clip_coef = request.clip_coef
            for item in list(request.factors):
                self.parameter_server.weight_factor[item.group_id] = item.norm
            logger.info(f"weight_factor={self.parameter_server.weight_factor} clip_coef={request.clip_coef}")
        self.parameter_server.start_update_weight_event.set()
        return ps_pb2.StartUpdateWeightResponse()

    @handle_value_error
    def UpdateGroupList(self, request, context):
        logger.info(f"receive UpdateGroupList, group_ids={request.groups}")
        with self.lock:
            self.group_ids = request.groups
        return ps_pb2.UpdateGroupListResponse()

    @handle_value_error
    def StartComputeNorm(self, request: ps_pb2.StartComputeNormRequest, context):
        logger.info(f"receive StartComputeNorm, group_ids={request.groups}")
        with self.parameter_server.lock:
            self.parameter_server.recved_groups = list(request.groups)
            self.parameter_server.compute_status = NormStatus.NORM_COMPUTING
        self.parameter_server.start_update_weight_event.set()
        return ps_pb2.StartComputeNormResponse()

    @handle_value_error
    def FinishPullWeight(self, request, context):
        logger.info("receive FinishPullWeight")
        self.parameter_server.finish_pull_weight_event.set()
        return ps_pb2.FinishPullWeightResponse()
