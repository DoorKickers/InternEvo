import importlib
import socket
import threading
import time
from threading import Lock
from typing import Dict, List, Tuple

import grpc
import torch
import zmq

import json
# Configure logging
from loguru import logger

from internlm.core.context import ParallelMode, global_context as gpc  # noqa: E402
from internlm.param_server.common import config
from internlm.param_server.common.utils import CustomServiceStub, serialize_layer, deserialize_layer
from internlm.param_server.proto import master_pb2_grpc
from internlm.param_server.proto.master_pb2 import (
    ClientHeartbeatRequest,
    ComputeStatus,
    OperatorType,
    PullStatus,
    PushStatus,
    QueryComputeStatusRequest,
    UpdateClientCkptRequest,
    UpdateClientCkptResponse,
    UpdateClientWeightFactorRequest,
    UpdatePullStatusRequest,
    UpdatePushStatusRequest,
    QueryGlobalStatusRequest,
    QueryGlobalStatusResponse,
    UpdateMetricLineRequest,
    UpdateMetricLineResponse,
)

from internlm.param_server.ps.ps_request import PSRequestHeader, RDMAConnectionInfo, LayerInfo, TensorInfo, RDMAOneSideRequest
from internlm.param_server.transport.rdma_transport import rdma_endpoint_ctx, endpoint_info_serialize, endpoint_info_deserialize


class ParameterClient:
    """
    The ParameterClient class represents a client that interacts with the parameter server.

    Attributes:
        worker_id (str): The ID of the worker.
        group_id (int): The ID of the group this client belongs to.
        zmq_context (zmq.Context): The ZMQ context for communication.
        zmq_socket (zmq.Socket): The ZMQ socket for communication.
        master_server (str): The address of the master server.
        master_stub (master_pb2_grpc.ControlFlowServiceStub): The gRPC stub for the master server.
        heartbeat_stop_event (threading.Event): The event to stop the heartbeat thread.
        heartbeat_thread (threading.Thread): The thread for sending heartbeat messages.
        lock (threading.Lock): The lock for thread safety.
    """

    def __init__(
        self,
    ):
        self.worker_id = socket.gethostname()
        # Initialize threading lock for thread safety
        self.lock = Lock()
        self.version = -1


    def start(self, group_id: int, group_weight: float, master_server: str, need_heartbeat: bool):
        """
        Initialize the ParameterClient.

        Args:
            group_id (int): The group ID this client belongs to.
            grpc_server (str): The gRPC server address.
            zmq_server (str): The ZMQ server address.
        """
        self.group_id = group_id
        # Initialize ZMQ
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, config.SOCKET_TIME_IN_SECOND * 1000)  # ms
        self.zmq_socket.setsockopt(zmq.SNDHWM, 0)
        self.zmq_socket.setsockopt(zmq.RCVHWM, 0)

        # Initialize gRPC
        self.master_alive = False
        self.master_server = master_server
        self.master_channel = grpc.insecure_channel(master_server)
        self.master_stub = CustomServiceStub(
            master_pb2_grpc.ControlFlowServiceStub(self.master_channel), config.GRPC_TIMEOUT
        )
        logger.info(f"Connected to gRPC server at {master_server}.")
        if need_heartbeat:
            self.send_weight_factor(group_weight)
            self.heartbeat_stop_event = threading.Event()
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
            logger.info("Heartbeat thread started.")
        if config.USE_DLSLIME_RDMA_TRANSFER:
            self.rdma_connection()

    def rdma_connection(self):
        dp_rank = gpc.get_local_rank(ParallelMode.DATA)
        tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
        wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT_DATA)
        wdp_rank = gpc.get_local_rank(ParallelMode.WEIGHT_DATA)
        global_rank = gpc.get_global_rank()

        dynamic_config = importlib.reload(config)
        assert dynamic_config is not None
        dynamic_config

        if dp_rank == 0 and tp_rank == 0 and wp_rank ==0 and wdp_rank == 0:
            dynamic_config.layer_chunks
            for _, zmq_server in dynamic_config.zmq_servers.items():
                with self.lock:
                    # Step 1: create RDMA endpoint
                    endpoint = rdma_endpoint_ctx().create(self.worker_id, self.group_id, global_rank, zmq_server)
                    # Step 2: send RDMA endpoint info and connect
                    self.zmq_socket.connect(zmq_server)
                    header = PSRequestHeader(
                        client_id=self.worker_id,
                        group_id=self.group_id
                    ).to_bytes()
                    connection_info = RDMAConnectionInfo(
                        hash_id=(self.worker_id, self.group_id, global_rank, zmq_server),
                        rdma_info=json.dumps(endpoint.endpoint_info)
                    ).to_bytes()
                    send_parts = [
                        b"RDMA_CONNECT",
                        header,
                        connection_info
                    ]
                    self.zmq_socket.send_multipart(send_parts)
                    message_parts: RDMAConnectionInfo = RDMAConnectionInfo.model_validate(self.zmq_socket.recv_json())
                    endpoint.connect(json.loads(message_parts.rdma_info))
                    self.zmq_socket.disconnect(zmq_server)
        print("rdma_connection done!!!!!")
        print(rdma_endpoint_ctx())

    def clear_receive_queue(self, socket, timeout=1000):
        """消耗接收队列中的所有消息"""
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        socks = dict(poller.poll(timeout))  # timeout in milliseconds

        while zmq.POLLIN in socks:
            message_parts = socket.recv_multipart()  # 消耗接收队列中的消息
            socks = dict(poller.poll(timeout))  # 再次检查队列
            print(f"message_parts: {message_parts}", flush=True)

        logger.info("finish clear receiving queue")

    def restart_zmq(self, ps_server):
        self.zmq_socket.disconnect(ps_server)
        self.zmq_socket.close()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, config.SOCKET_TIME_IN_SECOND * 1000)  # 等待超时, 单位毫秒
        self.zmq_socket.setsockopt(zmq.SNDHWM, 0)
        self.zmq_socket.setsockopt(zmq.RCVHWM, 0)
        logger.info("Finish zmq restart")

    def receive_updates(self, ps_server: str, layer_id: int) -> Tuple[int, Dict[str, torch.Tensor]]:
        """
        Receive updates for a specific layer from the server.

        Args:
            layer_id (int): The ID of the layer to receive updates for.

        Returns:
            tuple: A tuple containing the layer ID and the received state dictionary.
                   Returns (layer_id, None) if an error occurs.
        """
        header = PSRequestHeader(
            client_id=self.worker_id,
            group_id=self.group_id
        ).to_bytes()

        send_parts = [
            b"GET",
            header,
            str(layer_id).encode("utf-8"),
        ]
        try:
            start_ts = time.time()
            self.zmq_socket.connect(ps_server)
            self.zmq_socket.send_multipart(send_parts)
            message_parts: List[bytes] = self.zmq_socket.recv_multipart()
            self.zmq_socket.disconnect(ps_server)
            if not message_parts:
                logger.error("No response received from server.")
                return layer_id, None

            status = message_parts[0].decode("utf-8")
            if status == "OK":
                if len(message_parts) < 4:
                    logger.error("Incomplete response received from server.")
                    return layer_id, None

                ps_id = int(message_parts[1].decode("utf-8"))
                # recv_group_id = int(message_parts[2].decode("utf-8"))
                recv_layer_id = int(message_parts[3].decode("utf-8"))
                if recv_layer_id != layer_id:
                    logger.error(f"Received layer ID {recv_layer_id} does not match requested layer ID {layer_id}.")
                    return layer_id, None

                state_dict = deserialize_layer(self.group_id, layer_id, message_parts[4:])
                if state_dict is None:
                    logger.error(f"deserialize layer error {recv_layer_id=}")
                    return layer_id, None
                end_ts = time.time()
                logger.info(
                    f"Received update for layer {recv_layer_id=} from PS {ps_id}, "
                    f"cost time: {end_ts-start_ts:.3f} seconds."
                )
                return layer_id, state_dict
            else:
                logger.error(f"Error in receiving update for layer {layer_id}: {status}")
                return layer_id, None
        except zmq.ZMQError as e:
            logger.error(f"ZMQ Error in receive_updates: {e}")
            self.restart_zmq(ps_server)
            return layer_id, None
        except Exception as e:
            logger.error(f"Unexpected error in receive_updates: {e}")
            self.restart_zmq(ps_server)
            return layer_id, None

    def update_master_address(self, master_server):
        self.master_channel.close()
        self.master_server = master_server
        self.master_channel = grpc.insecure_channel(master_server)
        self.master_stub = CustomServiceStub(
            master_pb2_grpc.ControlFlowServiceStub(self.master_channel), config.GRPC_TIMEOUT
        )
        logger.info(f"update master stub successfully, master={master_server}")

    def _heartbeat_loop(self):
        """
        Periodically send heartbeat messages to the server to indicate the client is alive.
        """
        while not self.heartbeat_stop_event.is_set():
            try:
                request = ClientHeartbeatRequest(group_id=self.group_id)
                self.master_stub.ClientHeartbeat(request)
                if not self.master_alive:
                    self.master_alive = True
                    logger.info(f"Successfully connect to master {self.master_server}")
                time.sleep(config.HEARTBEAT_INTERVAL)  # Send heartbeat every HEARTBEAT_INTERVAL seconds
            except grpc.RpcError as grpc_error:
                logger.error(f"rpc error in heartbeat: {grpc_error}")
                logger.info("Begin to reload master address.")
                self.master_alive = False
                dynamic_config = importlib.reload(config)
                self.update_master_address(dynamic_config.master_server)
                time.sleep(5)  # check error every 5 seconds
            except Exception as e:  # pylint: disable=W0703
                logger.error(f"Unexpected error in heartbeat: {e}")
                logger.info("Begin to reload master address.")
                self.master_alive = False
                dynamic_config = importlib.reload(config)
                self.update_master_address(dynamic_config.master_server)
                time.sleep(5)  # check error every 5 seconds

    def update_push_status(self, push_status: PushStatus, consume_tokens: int = 0) -> OperatorType:
        try:
            version = 0
            with self.lock:
                version = self.version
            request = UpdatePushStatusRequest(
                group_id=self.group_id, push_status=push_status, consume_tokens=consume_tokens, version=version
            )
            response = self.master_stub.UpdatePushStatus(request)
            return response.op
        except grpc.RpcError as grpc_error:
            logger.error(f"rpc error in update_push_status: {grpc_error}")
            return OperatorType.RETRY
        except Exception as e:  # pylint: disable=W0703
            logger.error(f"Unexpected error in update_push_status: {e}")
            return OperatorType.RETRY

    def update_pull_status(self, pull_status: PullStatus) -> bool:
        try:
            request = UpdatePullStatusRequest(group_id=self.group_id, pull_status=pull_status)
            response = self.master_stub.UpdatePullStatus(request)
            if pull_status == PullStatus.PULL_STATUS_FINISH:
                with self.lock:
                    self.version = response.version
            return True
        except grpc.RpcError as grpc_error:
            logger.error(f"rpc error in update_push_status: {grpc_error}")
            return False
        except Exception as e:  # pylint: disable=W0703
            logger.error(f"Unexpected error in update_push_status: {e}")
            return False

    def query_compute_status(self) -> ComputeStatus:
        try:
            request = QueryComputeStatusRequest()
            response = self.master_stub.QueryComputeStatus(request)
            return response.compute_status
        except grpc.RpcError as grpc_error:
            logger.error(f"rpc error in query_compute_status: {grpc_error}")
            return ComputeStatus.COMPUTE_STATUS_UNKNOWN
        except Exception as e:  # pylint: disable=W0703
            logger.error(f"Unexpected error in query_compute_status: {e}")
            return ComputeStatus.COMPUTE_STATUS_UNKNOWN

    def request_ckpt_status(self, pull_status: PullStatus) -> OperatorType:
        try:
            request = UpdateClientCkptRequest(group_id=self.group_id, pull_status=pull_status)
            response: UpdateClientCkptResponse = self.master_stub.UpdateClientCkpt(request)
            if pull_status == PullStatus.PULL_STATUS_FINISH:
                with self.lock:
                    self.version = response.version
            return response.op
        except grpc.RpcError as grpc_error:
            logger.error(f"rpc error in request_ckpt_status: {grpc_error}")
            return OperatorType.RETRY
        except Exception as e:  # pylint: disable=W0703
            logger.error(f"Unexpected error in request_ckpt_status: {e}")
            return OperatorType.IGNORE

    def send_weight_factor(self, weight_factor: float):
        request = UpdateClientWeightFactorRequest(group_id=self.group_id, factor=weight_factor)
        self.master_stub.UpdateClientWeightFactor(request)

    def send_update_rdma(self, ps_server: str, p_state_dict: Dict[str, Dict[str, torch.Tensor]]):
        status = 0
        try:
            p_state_dict_info = {}
            global_rank = gpc.get_global_rank()
            endpoint = rdma_endpoint_ctx().get(self.worker_id, self.group_id, global_rank, ps_server)
            layer_info = []
            for layer_id, layer_state_dict in p_state_dict.items():
                tensor_info = []
                p_state_dict_info[layer_id] = {}
                for key, t in layer_state_dict.items():
                    endpoint.register_memory_region(key, t.data_ptr(), t.storage_offset(), t.numel() * t.itemsize)
                    tensor_info.append(TensorInfo(key=key, dtype=str(t.dtype), shape_info=list(t.shape)))
                layer_info.append(LayerInfo(layer_id=int(layer_id), tensor_info=tensor_info))

            header = PSRequestHeader(client_id=self.worker_id, group_id=self.group_id).to_bytes()
            
            one_side_request = RDMAOneSideRequest(
                opcode="READ",
                connection_info=RDMAConnectionInfo(
                    hash_id=(self.worker_id, self.group_id, global_rank, ps_server),
                    rdma_info=json.dumps(endpoint.endpoint_info)
                ),
                layer_info=layer_info
            ).to_bytes()

            send_parts = [
                b"RDMA_PUT",
                header,
                one_side_request
            ]

            with self.lock:
                self.zmq_socket.connect(ps_server)
                self.zmq_socket.send_multipart(send_parts)
                self.zmq_socket.recv()
                self.zmq_socket.disconnect(ps_server)
        except zmq.ZMQError as e:
            status = 1
            logger.error(f"ZMQ Error in send_update: {e}")
            self.restart_zmq(ps_server)
        except Exception as e:
            status = 1
            logger.error(f"Unexpected error in send_update: {e}")
            self.restart_zmq(ps_server)
        finally:
            return status

    def recv_update_rdma(self, ps_server, p_state_dict: Dict[str, Dict[str, torch.Tensor]]):
        status = 0
        try:
            p_state_dict_info = {}
            global_rank = gpc.get_global_rank()
            endpoint = rdma_endpoint_ctx().get(self.worker_id, self.group_id, global_rank, ps_server)
            layer_info = []
            for layer_id, layer_state_dict in p_state_dict.items():
                tensor_info = []
                p_state_dict_info[layer_id] = {}
                torch.cuda.synchronize()
                for key, t in layer_state_dict.items():
                    endpoint.register_memory_region(key, t.data_ptr(), t.storage_offset(), t.numel() * t.itemsize)
                    tensor_info.append(TensorInfo(key=key, dtype=str(t.dtype), shape_info=list(t.shape)))
                layer_info.append(LayerInfo(layer_id=int(layer_id), tensor_info=tensor_info))

            header = PSRequestHeader(
                client_id=self.worker_id,
                group_id=self.group_id
            ).to_bytes()

            oneside_request = RDMAOneSideRequest(
                opcode="WRITE",
                connection_info=RDMAConnectionInfo(
                    hash_id=(self.worker_id, self.group_id, global_rank, ps_server),
                    rdma_info=json.dumps(endpoint.endpoint_info)
                ),
                layer_info=layer_info
            ).to_bytes()
            
            send_parts = [
                b"RDMA_GET",
                header,
                oneside_request
            ]

            with self.lock:
                self.zmq_socket.connect(ps_server)
                self.zmq_socket.send_multipart(send_parts)
                self.zmq_socket.recv()
                self.zmq_socket.disconnect(ps_server)
        except zmq.ZMQError as e:
            status = 1
            logger.error(f"ZMQ Error in send_update: {e}")
            self.restart_zmq(ps_server)
        except Exception as e:
            status = 1
            logger.error(f"Unexpected error in send_update: {e}")
            self.restart_zmq(ps_server)
        finally:
            return status

    def send_update(self, ps_server: str, layer_id: int, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Send parameter updates for a specific layer to the server.

        Args:
            layer_id (int): The ID of the layer being updated.
            state_dict (Dict[str, torch.Tensor]): The state dictionary containing updated parameters.
        """
        status = 0
        try:
            header = PSRequestHeader(
                client_id=self.worker_id,
                group_id=self.group_id
            ).to_bytes()

            send_parts = [
                b"PUT",
                header,
                str(layer_id).encode("utf-8"),
            ] + serialize_layer(state_dict)

            with self.lock:
                self.zmq_socket.connect(ps_server)
                while True:
                    logger.info(f"Sent update for layer {layer_id} to server {ps_server}.")
                    start_ts = time.time()
                    self.zmq_socket.send_multipart(send_parts)

                    # Wait for server response
                    message = self.zmq_socket.recv_string()
                    end_ts = time.time()
                    logger.info(f"finish send and received server response: {message}, send cost: {end_ts-start_ts:.3f}")

                    if "Retry" in message:
                        time.sleep(1)
                        logger.info(f"Retry to send layer {layer_id} to server  {ps_server}.")
                    elif "successful" in message:
                        logger.info(f"Update for layer {layer_id} was successful.")
                        break
                    else:
                        logger.error(f"Update for layer {layer_id} failed: {message}")
                        break
                self.zmq_socket.disconnect(ps_server)
        except zmq.ZMQError as e:
            status = 1
            logger.error(f"ZMQ Error in send_update: {e}")
            self.restart_zmq(ps_server)
        except Exception as e:
            status = 1
            logger.error(f"Unexpected error in send_update: {e}")
            self.restart_zmq(ps_server)
        finally:
            return status
        
    def send_metric(self, metric_line: str):
        request = UpdateMetricLineRequest(group_id=self.group_id, metric_line=metric_line)
        self.master_stub.UpdateMetricLine(request)

    def shutdown(self):
        """Shut down client components gracefully."""
        try:
            self.heartbeat_stop_event.set()
            self.heartbeat_thread.join()
            self.zmq_socket.close()
            self.zmq_context.term()
            for channel in self.channels:
                channel.close()
            logger.info("Client shutdown complete.")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def get_layer_state_dict(
    model_state_dict: Dict[str, torch.Tensor], num_layers: int, layer_id: int
) -> Dict[str, torch.Tensor]:
    """
    Extract the state dictionary for a specific layer.

    Args:
        model_state_dict (Dict[str, torch.Tensor]): The full model state dictionary.
        num_layers (int): The total number of layers in the model.
        layer_id (int): The ID of the layer to extract.

    Returns:
        Dict[str, torch.Tensor]: The state dictionary for the specified layer.
    """
    layer_state_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith("layers"):
            layer_idx = int(key.split(".")[1])
            if layer_idx == layer_id:
                layer_state_dict[key] = value
        else:
            if layer_id == 0 and key.startswith("tok_embeddings"):
                layer_state_dict[key] = value
            elif layer_id == num_layers - 1 and (key.startswith("norm") or key.startswith("output")):
                layer_state_dict[key] = value
    return layer_state_dict


def get_ps_id(layer_id: int, layer_chunks: List[List[int]]) -> int:
    """
    Get the PS ID for a given layer ID.

    Args:
        layer_id (int): The ID of the layer.
        layer_chunks (List[List[int]]): A list of layer chunks.

    Returns:
        int: The ID of the PS server.
    """
    for ps_id, chunk in enumerate(layer_chunks):
        if layer_id in chunk:
            return ps_id
    return -1


client = ParameterClient()
http_server = None
# client.start(1, 1.0, "127.0.0.1:55500", True)
# request = QueryGlobalStatusRequest()
# res = client.master_stub.QueryGlobalStatus(request)
# print(res)


# def load_ckpt(ckpt_path: str) -> Dict[str, torch.Tensor]:
#     """
#     Load model from checkpoint file.

#     Args:
#         ckpt_path (str): Path to the checkpoint file.

#     Returns:
#         Dict[str, torch.Tensor]: The loaded state dictionary.
#     """
#     try:
#         state_dict = torch.load(ckpt_path, map_location="cpu")
#         logger.info(f"Checkpoint loaded from {ckpt_path}.")
#         return state_dict
#     except FileNotFoundError:
#         logger.error(f"Checkpoint file not found at {ckpt_path}.")
#         raise
#     except Exception as e:
#         logger.error(f"Error loading checkpoint: {e}")
#         raise

# if __name__ == "__main__":
#     import argparse
#     import internlm.param_server.config as config

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--group_id", type=int, help="The group ID for the client.")
#     args = parser.parse_args()
#     group_id = args.group_id
#     # Configuration
#     CHECKPOINT_PATH = "/mnt/petrelfs/share_data/jiaopenglong/internlm2-1_8b/model_tp0_pp0.pt"
#     NUM_LAYERS = 24
#     CONSUMED_TOKENS = 10000000
#     ps_servers = config.ps_servers
#     grpc_port = config.GRPC_PORT
#     zmq_port = config.ZMQ_PORT
#     grpc_servers = [f"{server}:{grpc_port}" for server in ps_servers]
#     zmq_servers = [f"tcp://{server}:{zmq_port}" for server in ps_servers]
#     layer_chunks = config.layer_chunks
#     num_ps = config.NUM_PS

#     # Load model checkpoint
#     try:
#         model_state_dict = load_ckpt(CHECKPOINT_PATH)
#     except Exception as e:
#         logger.error(f"Failed to load checkpoint: {e}")
#         exit(1)

#     # Initialize client
#     client = ParameterClient(group_id, grpc_servers)

#     # Send updates for each layer
#     for layer_id in range(NUM_LAYERS):
#         layer_state_dict = get_layer_state_dict(model_state_dict, NUM_LAYERS, layer_id)
#         ps_id = get_ps_id(layer_id, layer_chunks)
#         if ps_id == -1:
#             raise ValueError(f"Failed to find PS ID for layer {layer_id}.")
#         ps_server = zmq_servers[ps_id]
#         if layer_state_dict:
#             client.send_update(ps_server, layer_id, CONSUMED_TOKENS, layer_state_dict)
#         else:
#             logger.warning(f"No state dict found for layer {layer_id}.")

#     # Perform double check
#     double_check_response = client.double_check(message="upload finished")
#     if double_check_response.status != 200:
#         logger.error("Double check failed.")
#     else:
#         logger.info("Double check succeeded.")

#     # Wait for updates
#     MAX_WAIT_TIME = 60  # seconds
#     start_time = time.time()
#     while True:
#         status, step_weight = client.get_update()
#         if status == 1:
#             logger.info("Received successful update status.")
#             break
#         elif status == -1:
#             logger.error("Failed to retrieve update status.")
#             break
#         else:
#             logger.info(f"Waiting for updates, current status: {status}")
#         if time.time() - start_time > MAX_WAIT_TIME:
#             logger.error("Timeout waiting for updates.")
#             break
#         time.sleep(1)

#     # Receive updates for each layer
#     received_state_dicts = {}
#     for layer_id in range(NUM_LAYERS):
#         ps_id = get_ps_id(layer_id, layer_chunks)
#         if ps_id == -1:
#             raise ValueError(f"Failed to find PS ID for layer {layer_id}.")
#         ps_server = zmq_servers[ps_id]
#         recv_layer_id, state_dict = client.receive_updates(ps_server, layer_id)
#         if state_dict:
#             received_state_dicts[recv_layer_id] = state_dict
#             logger.info(f"Successfully received updates for layer {recv_layer_id}.")
#         else:
#             logger.error(f"Failed to receive updates for layer {recv_layer_id}.")
#     double_check_response = client.double_check(message="download finished")
#     if double_check_response.status != 200:
#         logger.error("Double check failed.")
#     else:
#         logger.info("Double check succeeded.")

#     logger.info("All updates received.")

#     # Shutdown client
#     client.shutdown()
