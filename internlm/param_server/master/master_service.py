from loguru import logger

from internlm.param_server.common.utils import handle_value_error
from internlm.param_server.master.manager import StateManager
from internlm.param_server.proto import master_pb2, master_pb2_grpc



class MasterControlFlowServicer(master_pb2_grpc.ControlFlowServiceServicer):
    def __init__(self):
        self.state_manager = StateManager()

    @handle_value_error
    def PSHeartbeat(self, request, context):
        logger.info(f"receive ps heartbeat, {request.ps_id=}, peer addr: {context.peer()}")
        self.state_manager.ps_heartbeat(request.ps_id)
        return master_pb2.PSHeartbeatResponse()

    @handle_value_error
    def ClientHeartbeat(self, request, context):
        logger.info(f"receive client heartbeat, {request.group_id=}")
        self.state_manager.client_heartbeat(request.group_id)
        return master_pb2.ClientHeartbeatResponse()

    @handle_value_error
    def UpdatePushStatus(self, request, context):
        logger.info(
            f"receive client update push status, {request.group_id=} {request.push_status=} {request.consume_tokens=}"
        )
        return self.state_manager.update_push_status(request)

    @handle_value_error
    def UpdateComputeStatus(self, request, context):
        logger.info(f"receive ps update compute status, {request.ps_id=} {request.compute_status=}")
        return self.state_manager.update_compute_status(request)

    @handle_value_error
    def UpdateNormStatus(self, request, context):
        logger.info(f"receive ps update norm status, {request.ps_id=} {request.norm_status=}")
        return self.state_manager.update_norm_status(request)

    @handle_value_error
    def QueryComputeStatus(self, request, context):
        logger.info("receive client query compute status")
        return self.state_manager.query_compute_status()

    @handle_value_error
    def UpdatePullStatus(self, request, context):
        logger.info(f"receive update pull status, {request.group_id=} {request.pull_status=}")
        return self.state_manager.update_pull_status(request)

    @handle_value_error
    def UpdateClientCkpt(self, request, context):
        logger.info(f"receive update client ckpt, {request.group_id=} {request.pull_status=}")
        return self.state_manager.update_client_ckpt_status(request)

    @handle_value_error
    def UpdatePSNorm(self, request, context):
        logger.info(f"receive update ps norm, {request.items=}")
        return self.state_manager.update_ps_norm(request)

    @handle_value_error
    def UpdateClientWeightFactor(self, request, context):
        logger.info(f"receive update client weight factor, {request.group_id=} {request.factor=}")
        return self.state_manager.update_client_weight_factor(request)

    @handle_value_error
    def QueryGlobalStatus(self, request, context):
        logger.info(f"receive platform query global status")
        return self.state_manager.query_global_status(request)
    
    @handle_value_error
    def UpdateMetricLine(self, request, context):
        logger.info(f"receive update metric line, {request.group_id=} {request.metric_line=}")
        return self.state_manager.update_metric_line(request)