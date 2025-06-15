import importlib
import os
import threading
import time
from concurrent import futures
from queue import Queue
from threading import Event, Lock
from typing import Any, Dict, List, Literal, Tuple, Callable
import functools

import json
import gc
import grpc

import torch

import zmq
from loguru import logger

from internlm.param_server.common import config
from internlm.param_server.common.utils import (
    save_ps_checkpoint,
    CustomServiceStub,
    serialize_layer,
    deserialize_layer,
)
from internlm.param_server.proto import master_pb2, master_pb2_grpc, ps_pb2_grpc
from internlm.param_server.proto.master_pb2 import ComputeStatus, NormStatus
from internlm.param_server.ps.ps_service import PSControlServicer
from internlm.param_server.transport.rdma_transport import rdma_endpoint_ctx

from internlm.param_server.ps.ps_store import BufferManager
from internlm.param_server.ps.ps_request import PSRequestHeader, RDMAConnectionInfo, RDMAOneSideRequest, LayerInfo


class WorkerThread(threading.Thread):
    fn: Dict[str, Callable[[Any], None]] = {}
    def __init__(self, context, id, ps: "ParameterServer"):
        threading.Thread.__init__(self)
        self.context:zmq.Context = context
        self.id = id
        self.ps: "ParameterServer" = ps
        self.zmq_poller = zmq.Poller()

    def rdma_connect(self, header: PSRequestHeader, multi_parts: List[bytes]):
        logger.info(f"RDMA Connection Request, {header=}")
        rdma_connection_info = RDMAConnectionInfo.from_bytes(multi_parts[0])
        logger.info(f"RDMA Connection Request, hash_id: {rdma_connection_info.hash_id}")
        endpoint = rdma_endpoint_ctx().create(*rdma_connection_info.hash_id)
        with self.ps.lock:
            endpoint.connect(json.loads(rdma_connection_info.rdma_info))
            logger.info(f"RDMA Connecting of {rdma_connection_info.hash_id}, {endpoint.endpoint_info=}")
            self.zmq_socket.send_json(
                RDMAConnectionInfo(
                    hash_id=rdma_connection_info.hash_id,
                    rdma_info=json.dumps(endpoint.endpoint_info)
                ).model_dump()
            )
            logger.info(f"RDMA Connecting of {rdma_connection_info.hash_id} done !!!")

    def rdma_put(self, header: PSRequestHeader, multi_parts: List[bytes]):
        group_id = header.group_id
        logger.info("rdma_put start")
        start_time = time.time()
        rdma_oneside_request = RDMAOneSideRequest.from_bytes(multi_parts[0])
        assert rdma_oneside_request.opcode == "READ"
        rdma_connection_info = rdma_oneside_request.connection_info
        endpoint = rdma_endpoint_ctx().get(*rdma_connection_info.hash_id)

        all_layer_info = rdma_oneside_request.layer_info
        if not all_layer_info:
            logger.warning("empty rdma put request")
        else:
            allocate_layer = all_layer_info[0]
            self.ps.storage.allocate_and_rdma_register(group_id, endpoint, allocate_layer)
            read_layer = allocate_layer
            for allocate_layer in all_layer_info[1:]:
                task = self.ps.storage.rdma_oneside("READ", group_id, endpoint, rdma_connection_info, read_layer, True)
                self.ps.storage.allocate_and_rdma_register(group_id, endpoint, allocate_layer)
                task.wait()
                read_layer = allocate_layer

            self.ps.storage.rdma_oneside("READ", group_id, endpoint, rdma_connection_info, read_layer)
        logger.info(f"rdma_put end, total cost: {time.time() - start_time}")
        response = f"get from group {header.group_id} successful"
        self.zmq_socket.send(response.encode())
        self.ps.groups_in_receiving.add(header.group_id)

    def rdma_get(self, header: PSRequestHeader, multi_parts: List[bytes]):
        group_id = header.group_id
        logger.info("rdma_get start")
        start_time = time.time()
        rdma_oneside_request = RDMAOneSideRequest.from_bytes(multi_parts[0])
        rdma_connection_info = rdma_oneside_request.connection_info
        endpoint = rdma_endpoint_ctx().get(*rdma_connection_info.hash_id)

        for layer_info in rdma_oneside_request.layer_info:
            self.ps.storage.rdma_oneside("WRITE", group_id, endpoint, rdma_connection_info, layer_info)

        logger.info(f"rdma_get end, total cost: {time.time() - start_time}")
        response = f"get from ps {self.ps.ps_id} successful"
        self.zmq_socket.send(response.encode())

    def zmq_put(self, header: PSRequestHeader, message_parts: List[bytes]):
        start_ts = time.time()
        layer_id = int(message_parts[0].decode("utf-8"))
        try:
            layer_state_dict = deserialize_layer(header.group_id, layer_id, message_parts[1:])
            if layer_state_dict is None:
                self.zmq_socket.send(b"ERROR: Malformed PUT request")
        except Exception as e:
            logger.warning(f"worker {self.id} request from client {header.client_id} {e}")
            self.zmq_socket.send(b"ERROR: Malformed PUT request")

        with self.ps.lock:
            self.ps.storage.add_layer_state_dict(header.group_id, layer_id, layer_state_dict)

        end_ts = time.time()
        logger.info(f"worker {self.id} Received complete state_dict from client {header.client_id} {header.group_id=} {layer_id=}, cost: {end_ts-start_ts:.3f}")
        response = f"send to ps {self.ps.ps_id} successful"
        self.zmq_socket.send(response.encode())
        self.ps.groups_in_receiving.add(header.group_id)

    def zmq_get(self, header: PSRequestHeader, message_parts: List[bytes]):
        layer_id = int(message_parts[0].decode("utf-8"))
        start_ts = time.time()
        chunks = self.ps.serialize_consumer.get_layer_chunks(layer_id)
        if chunks:
            # ps_id, group_id, layer_id, start_time, state_dict
            send_parts = [
                b"OK",
                str(self.ps.ps_id).encode("utf-8"),
                str(header.group_id).encode("utf-8"),
                str(layer_id).encode("utf-8"),
            ] + chunks
            self.zmq_socket.send_multipart(send_parts, copy=False, track=False)
            end_ts = time.time()
            logger.info(f"worker {self.id} Sent state_dict to client {header.client_id} {header.group_id=} {layer_id=} cost: {end_ts-start_ts:.3f}")
        else:
            self.zmq_socket.send_multipart([b"ERROR", b"State_dict not found"])
            logger.info(f"worker {self.id} State_dict not found for client {header.client_id}")

    def run(self):
        self.zmq_socket: zmq.Socket = self.context.socket(zmq.REP)
        self.zmq_socket.connect("inproc://backend")
        self.zmq_socket.setsockopt(zmq.SNDHWM, 0)
        self.zmq_socket.setsockopt(zmq.RCVHWM, 0)
        self.zmq_poller.register(self.zmq_socket, zmq.POLLIN)
        logger.info(f"Worker {self.id} started")

        while True:
            try:
                events = dict(self.zmq_poller.poll(timeout=1000 * 10))  # 10 second timeout
                if self.zmq_socket in events:
                    
                    message_parts = self.zmq_socket.recv_multipart()
                    if len(message_parts) < 3:
                        logger.warning("worker {self.id} Received malformed message with insufficient parts.")
                        self.zmq_socket.send(b"ERROR: Malformed message")
                        continue
                    try:
                        request_type: Literal["RDMA_PUT", "RDMA_GET", "RDMA_CONNECT", "PUT", "GET"] = message_parts[0].decode("utf-8")
                        header = PSRequestHeader.from_bytes(message_parts[1])
                        logger.info(
                            f"worker {self.id} Received request from client "
                            f"{header.client_id=}, {request_type=} {header.group_id=}"
                        )
                    except ValueError:
                        logger.warning(f"worker {self.id} Invalid group_id or from client {header.client_id}")
                        self.zmq_socket.send(b"ERROR: Invalid group_id")
                        continue
                    if request_type == "RDMA_CONNECT":
                        self.rdma_connect(header, message_parts[2:])
                    elif request_type == "RDMA_PUT":
                        self.rdma_put(header, message_parts[2:])
                    elif request_type == "RDMA_GET":
                        self.rdma_get(header, message_parts[2:])
                    elif request_type == "PUT":
                        self.zmq_put(header, message_parts[2:])
                    elif request_type == "GET":
                        self.zmq_get(header, message_parts[2:])
                    else:
                        logger.warning(
                            f"worker {self.id} Unknown request type '{request_type}' from client {header.client_id}"
                        )
                        self.zmq_socket.send(b"ERROR: Unknown request")
            except zmq.ZMQError as e:
                logger.error(f"worker {self.id} ZMQ Error: {e}")
                response = f"send to ps {self.ps.ps_id} failed. ZMQ Error: {e}"
                self.zmq_socket.send(response.encode())
            except Exception as e:
                logger.error(f"worker {self.id} Unexpected error in handle_data_flow: {e}")
                response = f"send to ps {self.ps.ps_id} failed. Error: {e}"
                self.zmq_socket.send(response.encode())

        self.zmq_socket.close()


class ZmqThread(threading.Thread):
    def __init__(self, io_threads, work_threads, zmq_port, ps: "ParameterServer"):
        threading.Thread.__init__(self)
        self.io_threads = io_threads
        self.work_threads = work_threads
        self.zmq_port = zmq_port
        self.ps = ps

    def run(self):
        try:
            context = zmq.Context(io_threads=self.io_threads)
            frontend = context.socket(zmq.ROUTER)
            frontend.setsockopt(zmq.SNDHWM, 0)
            frontend.setsockopt(zmq.RCVHWM, 0)
            frontend.bind(f"tcp://*:{self.zmq_port}")

            backend = context.socket(zmq.DEALER)
            backend.setsockopt(zmq.SNDHWM, 0)
            backend.setsockopt(zmq.RCVHWM, 0)
            backend.bind("inproc://backend")

            workers = []
            for i in range(self.work_threads):
                worker = WorkerThread(context, i, self.ps)
                worker.start()
                workers.append(worker)

            zmq.proxy(frontend, backend)

            frontend.close()
            backend.close()
            context.term()
        except Exception as e:
            logger.error(f"ZmqThread run fail, {e}")
            os._exit(1)


class SerializeConsumer:
    def __init__(self, ps, layers, num_thread=1):
        self.num_thread = num_thread
        self.ps = ps
        self.threads = []
        self.layer2chunks = {}
        self.lock = Lock()
        self.layers: List[int] = layers
        self.thread_events: Dict[int, Event] = {}
        for layer_id in self.layers:
            self.thread_events[layer_id] = Event()

    def _reset_thread_events(self):
        for layer_id in self.layers:
            self.thread_events[layer_id].clear()

    def start(self):
        for i in range(self.num_thread):
            t = threading.Thread(name=f"SerializeConsumer_{i}", target=self.thread_work)
            t.setDaemon(True)
            self.threads.append(t)
            t.start()
            logger.info(f"SerializeConsumer_{i} start")

    def get_layer_chunks(self, layer_id):
        if self.ps.only_communicate:
            self.do_serialize(layer_id)

        chunks = self._get_layer_chunks(layer_id)
        if chunks:
            return chunks

        logger.info(f"{layer_id=} not ready, wait for signal")
        self.thread_events[layer_id].wait()
        return self._get_layer_chunks(layer_id)

    def _get_layer_chunks(self, layer_id):
        with self.lock:
            if layer_id in self.layer2chunks:
                return self.layer2chunks[layer_id]

        return None

    def do_serialize(self, layer_id):
        start_ts = time.time()
        state_dict = self.ps.storage.get_layer_state_dict(layer_id)
        end_clone_layer_ts = time.time()
        if not state_dict:
            logger.warning(f"{layer_id=} not in model_state_dict")
            return

        try:
            chunks = serialize_layer(state_dict)
            with self.lock:
                self.layer2chunks[layer_id] = chunks
                self.thread_events[layer_id].set()
        except Exception as e:
            logger.error(f"{threading.current_thread().name} serialization failed: {e}")
        end_ts = time.time()
        logger.info(
            f"{threading.current_thread().name} serialize {layer_id=} done, "
            f"clone cost: {end_clone_layer_ts-start_ts:.3f} "
            f"total cost: {end_ts-start_ts:.3f}"
        )

    def thread_work(self):
        while True:
            layer_id = self.ps.task_queue.get()
            self.do_serialize(layer_id)
            self.ps.task_queue.task_done()

    def clear(self):
        with self.lock:
            self.layer2chunks.clear()
            self._reset_thread_events()


def multi_tensor_l2norm_torch(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    norms_tensor = torch.stack([torch.norm(tensor, p=2) for tensor in tensor_list])
    l2_norm = torch.norm(norms_tensor, p=2).unsqueeze(0)

    return l2_norm


class ParameterServer:
    def __init__(
        self,
        ps_id: int,
        global_num_layers: int,
        model_state_dict: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        origin_dtype: torch.bfloat16,
        layer_chunk: List[int],
        grpc_port: int = 50051,
        zmq_port: int = 5555,
        master_address: str = "localhost:50051",
        heartbeat_interval: int = 10,
        save_ckpt_path: str = None,
        only_communicate: bool = False,
    ):

        self.ps_id = ps_id
        self.global_num_layers = global_num_layers
        self.optimizer = optimizer
        self.origin_dtype = origin_dtype
        self.layer_chunk = layer_chunk
        self.heartbeat_interval = heartbeat_interval
        self.zmq_port = zmq_port
        self.storage = BufferManager(
            model_state_dict,
            self.origin_dtype,
            self.global_num_layers,
        )
        self.groups_in_receiving = set()
        self.recved_groups = []
        self.compute_status = ComputeStatus.RECEIVING
        self.master_server = master_address
        self.master_channel = grpc.insecure_channel(master_address)
        self.master_stub = CustomServiceStub(master_pb2_grpc.ControlFlowServiceStub(self.master_channel), config.GRPC_TIMEOUT)

        self.zmq_thread = ZmqThread(16, 16, self.zmq_port, self)

        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
        ps_pb2_grpc.add_ControlServiceServicer_to_server(PSControlServicer(self), self.grpc_server)
        self.grpc_server.add_insecure_port(f"[::]:{grpc_port}")

        # heartbeat
        self.heartbeat_thread = threading.Thread(target=self.heartbeat, daemon=True)

        # Initialize a lock for thread safety
        self.lock = Lock()

        # ckpt
        self.save_ckpt_path = save_ckpt_path
        self.is_time_to_save_ckpt = False
        self.version = 0
        self.consume_tokens = 0
        self.snapshot_counter = 0
        self.total_snapshot = 2
        self.save_ckpt = False

        # for test
        self.only_communicate = only_communicate
        self.task_queue = Queue()
        self.serialize_consumer = SerializeConsumer(self, config.layer_chunks[self.ps_id])

        self.finish_pull_weight_event = Event()
        self.start_update_weight_event = Event()

        # pseudo_gradient penalty
        self.pseudo_gradient = None
        self.weight_factor = {}
        self.clip_coef = 1.0

        logger.info(f"init parameter server, ps_id={self.ps_id}, layer_chunks={config.layer_chunks}")

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

    def compute_pseudo_gradient_naive(self, groups: List[int]) -> Dict[str, torch.Tensor]:
        pseudo_gradient = {}
        for group_id in groups:
            for layer_id in self.layer_chunk:
                cur_state_dict = self.storage.get_weight(group_id, layer_id)
                for key, weight in cur_state_dict.items():
                    local_weight = self.storage.model_state_dict[key]
                    if key not in pseudo_gradient:
                        pseudo_gradient[key] = local_weight - weight
                    else:
                        pseudo_gradient[key] += local_weight - weight
        for key, weight in pseudo_gradient.items():
            pseudo_gradient[key] = weight / len(groups)

        return pseudo_gradient

    def get_pseudo_gradient_by_group(self, groups: List[int]) -> Dict[str, torch.Tensor]:
        start_ts = time.time()
        logger.info(f"begin get_pseudo_gradient_by_group")
        pseudo_gradient = {}
        for group_id in groups:
            pseudo_gradient[group_id] = {}
            for layer_id in self.layer_chunk:
                cur_state_dict = self.storage.get_weight(group_id, layer_id)
                for key, weight in cur_state_dict.items():
                    local_weight = self.storage.model_state_dict[key]
                    pseudo_gradient[group_id][key] = local_weight - weight.float()
        self.pseudo_gradient = pseudo_gradient
        end_ts = time.time()
        logger.info(f"finish get_pseudo_gradient_by_group, cost: {end_ts-start_ts:.3f}")

    def get_weighted_pseudo_gradient(
        self, groups: List[int] = None, weight_factor: Dict[int, float] = None
    ) -> Dict[str, torch.Tensor]:
        start_ts = time.time()
        pseudo_gradient = {}
        if weight_factor is not None:
            for group_id, group_weight_factor in weight_factor.items():
                for key, weight in self.pseudo_gradient[group_id].items():
                    if key not in pseudo_gradient:
                        pseudo_gradient[key] = weight * group_weight_factor
                    else:
                        pseudo_gradient[key] += weight * group_weight_factor
        elif groups is not None:
            for group_id in groups:
                for key, weight in self.pseudo_gradient[group_id].items():
                    if key not in pseudo_gradient:
                        pseudo_gradient[key] = weight
                    else:
                        pseudo_gradient[key] += weight
            for key, weight in pseudo_gradient.items():
                pseudo_gradient[key] = weight / len(groups)
        else:
            raise ValueError("groups or weight_factor should be provided.")
        end_ts = time.time()
        logger.info(f"finish get_weighted_pseudo_gradient, cost: {end_ts-start_ts:.3f}")
        return pseudo_gradient

    def get_gradient_norm_by_group(self, pseudo_gradient_by_groups: Dict[str, Dict]) -> Dict[int, float]:
        start_ts = time.time()
        gradient_norms = {}
        for group_id, pseudo_gradient_group in pseudo_gradient_by_groups.items():
            tensor_list = list(pseudo_gradient_group.values())
            l2_norm = multi_tensor_l2norm_torch(tensor_list) ** 2
            gradient_norms[group_id] = l2_norm.item()
        end_ts = time.time()
        logger.info(f"finish get_gradient_norm_by_group, cost: {end_ts-start_ts:.3f}")
        return gradient_norms

    def send_norm(self, group_norms: Dict[int, float]):
        send_norms = [master_pb2.GroupNorm(group_id=key, norm=value) for key, value in group_norms.items()]
        request = master_pb2.UpdatePSNormRequest(items=send_norms)
        self.master_stub.UpdatePSNorm(request)

    def step(self, pseudo_gradient: Dict[str, torch.Tensor]):
        start_ts = time.time()
        for key, param in self.storage.model_state_dict.items():
            if key in pseudo_gradient:
                param.grad = pseudo_gradient[key].clone()
            else:
                raise ValueError(f"Key {key} not found in pseudo_gradient")
        self.optimizer.step()
        self.optimizer.zero_grad()
        end_ts = time.time()
        logger.info(f"ParameterServer.step cost: {end_ts-start_ts:.3f}")

    def update_master_address(self, master_server):
        self.master_channel.close()
        self.master_server = master_server
        self.master_channel = grpc.insecure_channel(master_server)
        self.master_stub = CustomServiceStub(master_pb2_grpc.ControlFlowServiceStub(self.master_channel), config.GRPC_TIMEOUT)
        logger.info(f"update master stub successfully, master={master_server}")

    def heartbeat(self):
        successfully_connect = False
        while True:
            try:
                request = master_pb2.PSHeartbeatRequest(ps_id=self.ps_id)
                self.master_stub.PSHeartbeat(request)
                if not successfully_connect:
                    successfully_connect = True
                    logger.info(f"Successfully connect to master {self.master_server}")
            except grpc.RpcError as e:
                logger.warning(f"(PS HeartBeat) RPC error: {e}")
                logger.info(f"Begin to reload master address.")
                successfully_connect = False
                dynamic_config = importlib.reload(config)
                self.update_master_address(dynamic_config.master_server)
            except Exception as e:
                logger.warning(f"(PS HeartBeat) RPC error: {e}")
                logger.info(f"Begin to reload master address.")
                successfully_connect = False
                dynamic_config = importlib.reload(config)
                self.update_master_address(dynamic_config.master_server)
            finally:
                time.sleep(self.heartbeat_interval)

    def start(self):
        self.grpc_server.start()
        self.zmq_thread.start()
        self.heartbeat_thread.start()

        for layer_id in config.layer_chunks[self.ps_id]:
            self.serialize_consumer.do_serialize(layer_id)

        self.serialize_consumer.start()
        logger.info("ParameterServer started and running.")
        try:
            while True:
                logger.info("waiting for start update weight event")
                self.start_update_weight_event.wait()
                self.start_update_weight_event.clear()
                if self.compute_status == ComputeStatus.COMPUTING:
                    logger.info(
                        f"receive start update weight event, ready to step, compute_status={self.compute_status}"
                    )
                elif self.compute_status == NormStatus.NORM_COMPUTING:
                    logger.info(
                        "receive start update norm event, ready to compute gradient norm, "
                        f"compute_status={self.compute_status}"
                    )
                with self.lock:
                    if self.compute_status == NormStatus.NORM_COMPUTING:
                        try:
                            norm_computing_start_ts = time.time()
                            self.get_pseudo_gradient_by_group(self.recved_groups)
                            if not config.USE_DLSLIME_RDMA_TRANSFER:
                                self.storage.destroy()
                                gc.collect()
                            gradient_norms = self.get_gradient_norm_by_group(self.pseudo_gradient)
                            logger.info(f"gradient_norms: {gradient_norms}")
                            self.send_norm(gradient_norms)
                            self.compute_status = NormStatus.NORM_SUCCESS  # Norm computation successful
                            norm_computing_end_ts = time.time()
                            logger.info(f"finish NORM_COMPUTING, cost: {norm_computing_end_ts-norm_computing_start_ts:.3f}")
                        except Exception as e:
                            logger.error(f"Error during update: {e}")
                            self.compute_status = NormStatus.NORM_FAIL  # Update failed

                    elif self.compute_status == ComputeStatus.COMPUTING:
                        try:
                            if self.only_communicate:
                                self.serialize_consumer.clear()
                                if not config.USE_DLSLIME_RDMA_TRANSFER:
                                    for layer_id in config.layer_chunks[self.ps_id]:
                                        self.task_queue.put(layer_id)
                                self.compute_status = ComputeStatus.COMPUTE_SUCCESS
                                logger.info("Only communicate, no update.")
                            else:
                                computing_start_ts = time.time()
                                if self.pseudo_gradient is None:
                                    self.get_pseudo_gradient_by_group(self.recved_groups)
                                    if not config.USE_DLSLIME_RDMA_TRANSFER:
                                        self.storage.destroy()
                                        gc.collect()
                                # weighted pseudo_gradient
                                pseudo_gradient = self.get_weighted_pseudo_gradient(weight_factor=self.weight_factor)

                                # Clip pseudo_gradient
                                if self.clip_coef < 1.0:
                                    clip_coef_start_ts = time.time()
                                    for _, weight in pseudo_gradient.items():
                                        weight.mul_(self.clip_coef)
                                    clip_coef_end_ts = time.time()
                                    logger.info(f"finish clip coef, clip_coef: {self.clip_coef} cost: {clip_coef_end_ts-clip_coef_start_ts:.3f}")
                                self.pseudo_gradient = None
                                self.weight_factor.clear()
                                self.step(pseudo_gradient)
                                self.serialize_consumer.clear()
                                if not config.USE_DLSLIME_RDMA_TRANSFER:
                                    for layer_id in config.layer_chunks[self.ps_id]:
                                        self.task_queue.put(layer_id)
                                self.compute_status = ComputeStatus.COMPUTE_SUCCESS  # Update successful
                                logger.info("Update step completed successfully.")
                                assert self.save_ckpt is False
                                self.save_ckpt = True
                                computing_end_ts = time.time()
                                logger.info(f"finish COMPUTING, cost: {computing_end_ts-computing_start_ts:.3f}")
                        except Exception as e:
                            logger.error(f"Error during update: {e}")
                            self.compute_status = ComputeStatus.COMPUTE_FAIL  # Update failed
                        finally:
                            self.master_stub.UpdateComputeStatus(
                                master_pb2.UpdateComputeStatusRequest(
                                    ps_id=self.ps_id, compute_status=self.compute_status
                                )
                            )
                    else:
                        logger.error(
                            f"invalid compute status, expect {ComputeStatus.COMPUTING}, "
                            f"actual compute_status={self.compute_status}"
                        )

                try:
                    if self.compute_status in (NormStatus.NORM_SUCCESS, NormStatus.NORM_FAIL):
                        self.master_stub.UpdateNormStatus(
                            master_pb2.UpdateNormStatusRequest(ps_id=self.ps_id, norm_status=self.compute_status)
                        )
                except Exception as e:
                    logger.error(f"PS UpdateNormStatus Error: {e}")
                    raise e

                try:
                    if self.save_ckpt:
                        logger.info("waiting for all serialize task done")
                        self.task_queue.join()
                        logger.info("all serialize task done, waiting for pull done")
                        self.finish_pull_weight_event.wait()
                        self.finish_pull_weight_event.clear()
                        logger.info("pull weight done, ready to save snapshot")
                        self.save_ckpt = False
                        logger.info(f"Begin save checkpoint for snapshot {self.snapshot_counter}.")
                        save_path = os.path.join(
                            self.save_ckpt_path, f"ps_{self.ps_id}", "snapshot", f"{self.snapshot_counter}"
                        )
                        save_ps_checkpoint(
                            self.storage.model_state_dict,
                            self.optimizer,
                            save_path,
                            self.version,
                        )
                        logger.info(f"Finish save checkpoint for snapshot {self.snapshot_counter}.")
                        self.snapshot_counter = (self.snapshot_counter + 1) % 2
                        if self.is_time_to_save_ckpt:
                            # consume_tokens = int(self.consume_tokens // 1e9)  # unit: Billion
                            consume_tokens = int(self.consume_tokens)  # unit: Billion
                            logger.info(
                                f"Begin save checkpoint for history. consume_tokens={consume_tokens}B, version={self.version}"
                            )
                            save_path = os.path.join(
                                self.save_ckpt_path, f"ps_{self.ps_id}", f"{self.version}_{consume_tokens}B"
                            )
                            save_ps_checkpoint(
                                self.storage.model_state_dict,
                                self.optimizer,
                                save_path,
                                self.version,
                            )
                            logger.info("Finish save checkpoint for history.")
                except Exception as e:
                    logger.error(f"PS Save Checkpoint Error: {e}")

        except KeyboardInterrupt:
            logger.info("Shutting down server...")
        finally:
            self.task_queue.join()
            self.grpc_server.stop(0)
            self.master_channel.close()
            logger.info("Server shutdown complete.")
