import copy
import importlib
import math
import os
import time
from collections import defaultdict
from enum import Enum
from threading import Lock
from typing import Dict, Tuple, List

import grpc
import torch
from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from internlm.param_server.common import config
from internlm.param_server.master.grad_penalty import GroupNorm, OnlineDynamicEWMA
from internlm.param_server.proto import ps_pb2_grpc
from internlm.param_server.proto.master_pb2 import (
    ComputeStatus,
    NormStatus,
    OperatorType,
    PullStatus,
    PushStatus,
    HistoryGlobalStatus,
    CurrentGlobalStatus,
    QueryComputeStatusResponse,
    UpdateClientCkptRequest,
    UpdateClientCkptResponse,
    UpdateClientWeightFactorRequest,
    UpdateClientWeightFactorResponse,
    UpdateComputeStatusRequest,
    UpdateComputeStatusResponse,
    UpdateNormStatusRequest,
    UpdateNormStatusResponse,
    UpdatePSNormRequest,
    UpdatePSNormResponse,
    UpdatePullStatusRequest,
    UpdatePullStatusResponse,
    UpdatePushStatusRequest,
    UpdatePushStatusResponse,
    QueryGlobalStatusRequest,
    QueryGlobalStatusResponse
)
from internlm.param_server.proto.ps_pb2 import (
    FinishPullWeightRequest,
    GroupWeightFactor,
    StartComputeNormRequest,
    StartUpdateWeightRequest,
    UpdateGroupListRequest,
)
from internlm.param_server.common.utils import CustomServiceStub


class WorkerState:
    def __init__(self, offline_threshold, name):
        self.lock = Lock()
        self.alive_workers: Dict[int, float] = {}
        self.offline_threshold = offline_threshold
        self.name = name
        self.status_history_dict: Dict[int, List[Tuple[float, str]]] = {}

    def update(self, worker_id):
        if worker_id < 0:
            info = f"invalid {self.name} id: {worker_id}, should ge zero"
            logger.error(info)
            raise ValueError(info)

        ts = time.time()
        with self.lock:
            if self.alive_workers.get(worker_id, None) == None:
                print("aaaaaaaaaaaaaaaaaaaaaaaa")
                print(worker_id)
                # self.status_history_dict.setdefault(worker_id, []).append((time.time(), "CONNECTED"))
                # print(self.status_history_dict.setdefault(worker_id))
            self.alive_workers[worker_id] = ts

    def alive_worker(self):
        with self.lock:
            return set(self.alive_workers.keys())

    def check_alive(self):
        current_ts = time.time()
        dead_workers = []
        with self.lock:
            for key, value in self.alive_workers.items():
                if current_ts - value > self.offline_threshold:
                    dead_workers.append(key)

            for dead_worker in dead_workers:
                self.alive_workers.pop(dead_worker, None)
                self.status_history_dict.setdefault(dead_worker, []).append((time.time(), "LOST_CONNECTION"))

        if len(dead_workers) != 0:
            logger.error(f"some {self.name} workers are dead, {dead_workers=}")
        return dead_workers

    def __repr__(self):
        with self.lock:
            return str(list(self.alive_workers.keys()))


class PushState:
    def __init__(self):
        self.status_dict: Dict[int, PushStatus] = {}
        self.partial_sync_rate_threshold = config.PARTIAL_SYNC_RATE_THRESHOLD
        self.status_history_dict: Dict[int, List[Tuple[float, str]]] = {}
        self.name_mapping: Dict[int, str] = {
            0: "PUSH_UNKNOWN",
            1: "PUSH_START",
            2: "PUSH_FINISH",
            3: "PUSH_FAIL", 
            4: "PUSH_LOW_VERSION"
        }

    def reset(self):
        self.status_dict.clear()

    def check_parameter(self, group_id, push_status):
        if group_id < 0:
            info = f"invalid group_id: {group_id}, should ge zero"
            logger.error(info)
            raise ValueError(info)

        if push_status == PushStatus.PUSH_STATUS_UNKNOWN:
            info = f"invalid push_status: {push_status}"
            logger.error(info)
            raise ValueError(info)

    # TODO(caikun): how to handle error
    def update_state(self, group_id, push_status):
        if push_status == PushStatus.PUSH_STATUS_START:
            if group_id in self.status_dict:
                logger.warning(f"item already exist, {group_id=} value={push_status}")
            self.status_dict[group_id] = push_status
        elif push_status in [PushStatus.PUSH_STATUS_FAIL, PushStatus.PUSH_STATUS_FINISH]:
            if group_id not in self.status_dict:
                logger.warning(f"item not exist, {group_id=} value={push_status}")
            self.status_dict[group_id] = push_status
        else:
            self.status_dict[group_id] = push_status
        self.status_history_dict.setdefault(group_id, []).append((time.time(), self.name_mapping[self.status_dict[group_id]]))

    def remove_state(self, group_ids):
        for group_id in group_ids:
            self.status_dict.pop(group_id, None)

    def empty(self):
        return len(self.status_dict) == 0

    def all_finish(self, group_ids: set):
        for group_id in group_ids:
            if group_id not in self.status_dict:
                return False
            val = self.status_dict[group_id]
            if val == PushStatus.PUSH_STATUS_START:
                return False

        return True

    def contain_unfinish(self, group_ids):
        for group_id in group_ids:
            if group_id in self.status_dict and self.status_dict[group_id] == PushStatus.PUSH_STATUS_START:
                return True
        return False

    def get_push_fail_groups(self):
        fail_groups = set()
        for group_id, status in self.status_dict.items():
            if status in [PushStatus.PUSH_STATUS_FAIL, PushStatus.PUSH_STATUS_LOW_VERSION]:
                fail_groups.add(group_id)
        return fail_groups

    def get_received_groups(self):
        received_groups = set()
        for group_id, status in self.status_dict.items():
            if status == PushStatus.PUSH_STATUS_FINISH:
                received_groups.add(group_id)
        return received_groups

    def partial_finish(self, group_ids: set):
        is_all_finished = True
        finish_group_num = 0
        received_groups = set()
        valid_group_num = 0

        for group_id in group_ids:
            if group_id not in self.status_dict:
                is_all_finished = False
                valid_group_num += 1
                continue

            val = self.status_dict[group_id]
            if val == PushStatus.PUSH_STATUS_FINISH:
                finish_group_num += 1
                valid_group_num += 1
                received_groups.add(group_id)
            elif val == PushStatus.PUSH_STATUS_START:
                valid_group_num += 1
                is_all_finished = False

        is_partial_finished = False
        if is_all_finished:
            is_partial_finished = True
        else:
            is_partial_finished = (finish_group_num >= int(valid_group_num * self.partial_sync_rate_threshold))
        return is_partial_finished, is_all_finished, received_groups

    def __repr__(self):
        return f"status_dict: {self.status_dict}"


class GlobalStatus(Enum):
    INIT = 1
    PUSH = 2
    GRACE = 3
    PARTIAL_SYNC = 4
    COMPUTE_NORM = 5
    COMPUTE_STEP = 6
    PULL = 7
    FAIL = 8


class ComputeState:
    def __init__(self, num_ps):
        self.num_ps = num_ps
        self.status_dict: Dict[int, ComputeStatus] = {}
        self.reset()

    def reset(self):
        for ps_id in range(self.num_ps):
            self.status_dict[ps_id] = ComputeStatus.COMPUTE_STATUS_UNKNOWN

    def check_parameter(self, ps_id, compute_status):
        if ps_id < 0 or ps_id >= self.num_ps:
            info = f"invalid ps_id: {ps_id}, should in [0, {self.num_ps})"
            logger.error(info)
            raise ValueError(info)

        if compute_status not in [
            ComputeStatus.COMPUTE_SUCCESS,
            ComputeStatus.COMPUTE_FAIL,
        ]:
            info = f"invalid compute_status: {compute_status}"
            logger.error(info)
            raise ValueError(info)

    def update(self, ps_id, compute_status):
        self.status_dict[ps_id] = compute_status

    def all_finish(self):
        # 0: not ready, 1: success, 2: fail
        statuses = list(self.status_dict.values())
        if ComputeStatus.COMPUTE_STATUS_UNKNOWN in statuses:
            return 0
        if ComputeStatus.COMPUTE_FAIL in statuses:
            return 2
        return 1

    def __repr__(self):
        return f"{self.status_dict}"


class NormState:
    def __init__(self, num_ps):
        self.num_ps = num_ps
        self.status_dict: Dict[int, NormStatus] = {}
        self.reset()

    def reset(self):
        for ps_id in range(self.num_ps):
            self.status_dict[ps_id] = NormStatus.NORM_STATUS_UNKNOWN

    def check_parameter(self, ps_id, norm_status):
        if ps_id < 0 or ps_id >= self.num_ps:
            info = f"invalid ps_id: {ps_id}, should in [0, {self.num_ps})"
            logger.error(info)
            raise ValueError(info)

        if norm_status not in [
            NormStatus.NORM_SUCCESS,
            NormStatus.NORM_FAIL,
        ]:
            info = f"invalid norm_status: {norm_status}"
            logger.error(info)
            raise ValueError(info)

    def update(self, ps_id, norm_status):
        self.status_dict[ps_id] = norm_status

    def all_finish(self):
        # 0: not ready, 1: success, 2: fail
        statuses = list(self.status_dict.values())
        if NormStatus.NORM_STATUS_UNKNOWN in statuses:
            return 0
        if NormStatus.NORM_FAIL in statuses:
            return 2
        return 1

    def __repr__(self):
        return f"{self.status_dict}"


class PullState:
    def __init__(self):
        self.status_dict: Dict[int, PullStatus] = {}
        self.status_history_dict: Dict[int, List[Tuple[float, str]]] = {}
        self.name_mapping: Dict[int, str] = {
            0: "PULL_UNKNOWN",
            1: "PULL_START",
            2: "PULL_FINISH",
            3: "PULL_FAIL", 
        }

    def reset(self):
        self.status_dict.clear()

    def check_parameter(self, group_id, pull_status):
        if group_id < 0:
            info = f"invalid group_id: {group_id}, should ge zero"
            logger.error(info)
            raise ValueError(info)

        if pull_status != PullStatus.PULL_STATUS_FINISH and pull_status != PullStatus.PULL_STATUS_START:
            info = f"invalid pull_status: {pull_status}"
            logger.error(info)
            raise ValueError(info)

    def update(self, group_id, pull_status):
        self.status_dict[group_id] = pull_status
        print("ppppppppppppppppp", pull_status)
        self.status_history_dict.setdefault(group_id, []).append((time.time(), self.name_mapping[self.status_dict[group_id]]))

    def remove_state(self, group_ids):
        for group_id in group_ids:
            self.status_dict.pop(group_id, None)

    def empty(self):
        return len(self.status_dict) == 0

    def all_finish(self, group_ids: set):
        for group_id in group_ids:
            if group_id not in self.status_dict:
                return False
            if self.status_dict[group_id] == PullStatus.PULL_STATUS_START:
                return False

        return True


class ConsumeTokenState:
    def __init__(self):
        self.num_tokens: Dict[int, int] = {}  # group_id -> num_token
        self.last_num_tokens: Dict[int, int] = {}
        self.total_tokens_last_cycle = 0

    def update(self, group_id, num_token):
        if group_id in self.last_num_tokens:
            if self.last_num_tokens[group_id] >= num_token:
                logger.warning(
                    f"group_id {group_id}: last_num_tokens > cur_num_token, "
                    f"last_num_tokens: {self.last_num_tokens[group_id]}, cur_num_token: {num_token}"
                )
                self.last_num_tokens[group_id] = 0
            else:
                self.last_num_tokens[group_id] = copy.deepcopy(self.num_tokens[group_id])
        else:
            self.last_num_tokens[group_id] = 0
        self.num_tokens[group_id] = num_token
        logger.info(f"update consume token, {group_id=}, {num_token=}")

    def total_tokens(self):
        return sum(self.num_tokens.values())

    def step(self):
        self.total_tokens_last_cycle = self.total_tokens()

    def total_tokens_this_cycle(self):
        return self.total_tokens() - self.total_tokens_last_cycle

    def cur_consume_tokens(self, groups):
        consume_tokens = {}
        for group_id in groups:
            consume_tokens[group_id] = self.num_tokens[group_id] - self.last_num_tokens[group_id]
        return consume_tokens

    def load_state_dict(self, state_dict):
        self.num_tokens = state_dict["num_tokens"]
        self.last_num_tokens = state_dict["last_num_tokens"]
        self.total_tokens_last_cycle = state_dict["total_tokens_last_cycle"]

    def state_dict(self):
        return {
            "num_tokens": self.num_tokens,
            "last_num_tokens": self.last_num_tokens,
            "total_tokens_last_cycle": self.total_tokens_last_cycle,
        }


class VersionManager:
    def __init__(self):
        self.version = 0

    def load_version(self, version):
        self.version = version
        logger.info(f"load version, version: {self.version}")

    def incr(self):
        self.version += 1
        logger.info(f"update master version successfully, version: {self.version}")


class StateManager:
    def __init__(self):
        self.lock = Lock()
        self.connect_timeout = 300  # 5min

        save_ckpt_path = config.ckpt.get("save_ckpt_path")
        self.ckpt_path = os.path.join(save_ckpt_path, "master")
        os.makedirs(self.ckpt_path, exist_ok=True)

        self.global_status: GlobalStatus = GlobalStatus.INIT
        self.current_global_status_begin_timestamp = time.time()
        self.global_status_history_list = []
        self.version_manager = VersionManager()

        self.group_status = dict()

        self.ps_state = WorkerState(config.PS_OFFLINE_THRESHOLD, name="ps")
        self.group_state = WorkerState(config.GROUPS_OFFLINE_THRESHOLD, name="group")

        self.push_state = PushState()
        self.compute_state = ComputeState(config.NUM_PS)
        self.norm_state = NormState(config.NUM_PS)
        self.pull_state = PullState()
        self.consume_token_state = ConsumeTokenState()

        self.grace_start = None
        self.partial_sync_start = None
        self.update_method = config.UPDATE_METHOD

        self.num_ps = config.NUM_PS
        self.ps_servers = config.grpc_servers.values()
        self.ps_stubs: ps_pb2_grpc.ControlServiceStub = []
        for ps_server in self.ps_servers:
            channel = grpc.insecure_channel(ps_server)
            stub = CustomServiceStub(ps_pb2_grpc.ControlServiceStub(channel), config.GRPC_TIMEOUT)
            self.ps_stubs.append(stub)
        logger.info(f"init ps stub successfully, ps_servers={self.ps_servers}")

        self.scheduler = BackgroundScheduler()

        def dump_state():
            logger.info(str(self))

        self.dump_job = self.scheduler.add_job(dump_state, "interval", seconds=config.DUMP_STATE_INTERVAL)
        self.check_alive_job = self.scheduler.add_job(self.check_alive, "interval", seconds=config.ALIVE_CHECK_INTERVAL)
        self.scheduler.start()
        self.grace_check_job = None
        self.partial_sync_check_job = None

        self.received_groups = set()
        self.push_fail_groups = set()

        # group request for ckpt
        self.groups_request_for_ckpt = set()

        self.snapshot_counter = 0

        self.groups_fail = set()
        self.ps_all_alive = False

        self.already_send_start_update_weight_req = False

        # group weight factor
        self.groups_weight_factor: Dict[int, float] = defaultdict(float)

        # group grad norm
        self.groups_norm = GroupNorm()

        self.use_ewma_outlier = config.grad.get("use_ewma_outlier", False)
        self.use_weighted_grad = config.grad.get("use_weighted_avg", False)
        self.clip_pseudo_grad = config.grad.get("clip_pseudo_grad", None)
        self.anomaly_detector = None
        self.has_compute_norm_fail = False
        if self.use_ewma_outlier:
            self.anomaly_detector = OnlineDynamicEWMA(
                alpha=config.grad["ewma"].get("alpha", 0.02),
                warmup_steps=config.grad["ewma"].get("warmup_steps", 100),
                base_threshold=config.grad["ewma"].get("base_threshold", 3),
            )

        if config.ckpt.get("auto_resume"):
            self.load_last_ckpt()

        self.need_pull_groups = set()
        self.save_history = False

    def state_dict(self):
        # 1. OnlineDynamicEWMA
        # 2. version
        # 3. token
        return {
            "anomaly_detector": self.anomaly_detector.state_dict() if self.use_ewma_outlier else None,
            "version_manager": self.version_manager.version,
            "consume_token_state": self.consume_token_state.state_dict(),
            "groups_weight_factor": self.groups_weight_factor,
        }

    def load_state_dict(self, state_dict):
        if self.use_ewma_outlier:
            if state_dict["anomaly_detector"] is None:
                logger.warning("use_ewma_outlier is True, but anomaly_detector is None!")
            else:
                self.anomaly_detector.load_state_dict(state_dict["anomaly_detector"])
        self.version_manager.load_version(state_dict["version_manager"])
        self.consume_token_state.load_state_dict(state_dict["consume_token_state"])
        self.groups_weight_factor = state_dict["groups_weight_factor"]

        logger.info("load state dict successfully")

    def save_ckpt(self):
        logger.info(f"Begin save checkpoint for snapshot {self.snapshot_counter}.")
        snapshot_path = os.path.join(self.ckpt_path, "snapshot", f"{self.snapshot_counter}")
        os.makedirs(snapshot_path, exist_ok=True)
        snapshot_file = os.path.join(snapshot_path, "master.pt")
        state_dict = self.state_dict()
        torch.save(state_dict, snapshot_file)
        self.snapshot_counter = (self.snapshot_counter + 1) % 2
        logger.info(f"Finish save checkpoint for snapshot {self.snapshot_counter}.")
        
        if self.save_history:
            self.save_history = False
            # consume_tokens = int(self.consume_token_state.total_tokens() // 1e9)
            consume_tokens = int(self.consume_token_state.total_tokens())
            logger.info(f"Begin save checkpoint for history version={self.version_manager.version}, tokens={consume_tokens}B.")
            save_path = os.path.join(self.ckpt_path, f"{self.version_manager.version}_{consume_tokens}B")
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, "master.pt")
            torch.save(state_dict, save_file)
            logger.info(f"Finish save checkpoint for history.")
            

    def load_last_ckpt(self):
        snapshots_path = os.path.join(self.ckpt_path, "snapshot")
        snapshots = os.listdir(snapshots_path)
        if len(snapshots) == 0:
            return
        last_version = 0
        last_state_dict = None
        for snapshot_counter in snapshots:
            ckpt_file = os.path.join(snapshots_path, snapshot_counter, "master.pt")
            if os.path.exists(ckpt_file):
                cur_state_dict = torch.load(ckpt_file)
                version = cur_state_dict["version_manager"]
                if version > last_version:
                    last_version = version
                    last_state_dict = cur_state_dict
                    latest_path = os.path.join(snapshots_path, snapshot_counter)
        if last_state_dict is None:
            logger.warning(f"no valid ckpt file in {snapshots_path}")
            return
        logger.info(f"Find latest state dict: {latest_path}, auto resume begin loading {last_state_dict}.")
        self.load_state_dict(last_state_dict)

    def ps_heartbeat(self, ps_id):
        print("1111111111111111111")
        self.ps_state.update(ps_id)

    def client_heartbeat(self, group_id):
        # self.group_status[group_id]["current_status_begin_timestamp"] = time.time()
        # self.group_status[group_id]["current_status"] = 0
        self.group_state.update(group_id)

    def __repr__(self):
        return str(
            f"global_status: {self.global_status}, update_method: {self.update_method},"
            f" alive ps: {self.ps_state}, alive groups: {self.group_state}"
        )

    def update_ps_address(self, ps_servers):
        self.ps_servers = ps_servers
        self.ps_stubs = []
        for ps_server in self.ps_servers:
            channel = grpc.insecure_channel(ps_server)
            stub = CustomServiceStub(ps_pb2_grpc.ControlServiceStub(channel), config.GRPC_TIMEOUT)
            self.ps_stubs.append(stub)
        logger.info(f"update ps stub successfully, ps_servers={ps_servers}")

    def check_alive(self):
        # TODO: check interval
        self.ps_state.check_alive()
        dead_workers = self.group_state.check_alive()

        if config.NUM_PS > len(self.ps_state.alive_workers):
            logger.info("Begin to reload ps address.")
            self.ps_all_alive = False
            dynamic_config = importlib.reload(config)
            self.update_ps_address(dynamic_config.grpc_servers.values())
            self.reset()
        elif config.NUM_PS == len(self.ps_state.alive_workers):
            self.ps_all_alive = True
        else:
            assert False, str(
                f"The ps num is abnormal. The origin num {config.NUM_PS} must always "
                f"be greater than or equal to current alive_ps {len(self.ps_state.alive_workers)}"
            )

        if len(dead_workers) != 0:
            with self.lock:
                self.push_state.remove_state(dead_workers)
                self.pull_state.remove_state(dead_workers)
                for group_id in dead_workers:
                    if group_id in self.received_groups:
                        self.received_groups.discard(group_id)
                    self.need_pull_groups.discard(group_id)

                if self.global_status == GlobalStatus.PUSH:
                    if self.push_state.empty():
                        self.reset()
                    elif self.push_state.all_finish(self.group_state.alive_worker() - self.need_pull_groups):
                        self.received_groups = self.push_state.get_received_groups()
                        self.finish_push()
                elif self.global_status == GlobalStatus.GRACE:
                    if self.push_state.empty():
                        self.reset()
                    elif self.push_state.all_finish((self.group_state.alive_worker() & self.received_groups) - self.need_pull_groups):
                        self.finish_push()
                elif self.global_status == GlobalStatus.PARTIAL_SYNC:
                    if self.push_state.empty():
                        self.reset()
                    elif self.push_state.all_finish(self.group_state.alive_worker() - self.need_pull_groups):
                        self.received_groups = self.push_state.get_received_groups()
                        self.finish_push()
                elif self.global_status == GlobalStatus.PULL and self.pull_state.all_finish(
                    self.group_state.alive_worker() & self.received_groups
                ):
                    self.finish_pull()

            alive_workers = self.group_state.alive_worker()
            req = UpdateGroupListRequest(groups=alive_workers)
            for stub in self.ps_stubs:
                try:
                    stub.UpdateGroupList(req)
                except grpc.RpcError as e:
                    logger.error(f"rpc error when call UpdateGroupList: {e.code()=} {e.details()=}")

    def need_save_ckpt(self):
        return self.consume_token_state.total_tokens_this_cycle() >= config.CONSUME_TOKEN_WHEN_PS_SAVE_CKPT

    def _update_push_status_sync_mode(self, request: UpdatePushStatusRequest):
        if self.push_state.all_finish(self.group_state.alive_worker() - self.need_pull_groups):
            self.received_groups = self.push_state.get_received_groups()
            self.finish_push()

        if request.push_status == PushStatus.PUSH_STATUS_LOW_VERSION:
            return UpdatePushStatusResponse(op=OperatorType.PULL)
        return UpdatePushStatusResponse(op=OperatorType.PUSH)

    def _update_push_status_async_buffer_mode(self, request: UpdatePushStatusRequest):
        if (
            request.push_status == PushStatus.PUSH_STATUS_FINISH
            and self.grace_start is None
            and self.update_method == "async_buffer"
        ):
            self.grace_start = time.time()
            self.grace_check_job = self.scheduler.add_job(
                self.grace_check, "interval", seconds=config.GRACE_TIME_IN_SECOND
            )

        return self._update_push_status_sync_mode(request)

    def _update_push_status_partial_sync_mode(self, request: UpdatePushStatusRequest):
        if self.partial_sync_start is not None:
            return self._update_push_status_sync_mode(request)

        if request.push_status == PushStatus.PUSH_STATUS_START:
            return UpdatePushStatusResponse(op=OperatorType.PUSH)

        is_partial_finished, is_all_finished, received_groups = self.push_state.partial_finish(
            self.group_state.alive_worker() - self.need_pull_groups
        )
        logger.info(f"partial_finish returns: {is_partial_finished=} {is_all_finished=} {received_groups=}")
        if is_all_finished:
            self.received_groups = received_groups
            self.finish_push()
        elif is_partial_finished:
            self.partial_sync_start = time.time()
            self.partial_sync_check_job = self.scheduler.add_job(
                self.partial_sync_check, "interval", seconds=config.PARTIAL_SYNC_WAIT_TIME
            )
            logger.info("add partial_sync_check job")

        if request.push_status == PushStatus.PUSH_STATUS_LOW_VERSION:
            return UpdatePushStatusResponse(op=OperatorType.PULL)
        else:
            return UpdatePushStatusResponse(op=OperatorType.PUSH)

    def update_push_status(self, request: UpdatePushStatusRequest):
        if not self.ps_all_alive:
            return UpdatePushStatusResponse(op=OperatorType.IGNORE)

        self.push_state.check_parameter(request.group_id, request.push_status)
        if request.consume_tokens < 0:
            raise ValueError(f"invalid consume_tokens: {request.consume_tokens}, should ge zero")
        with self.lock:
            # request.version == master.version: global status must in [INIT, PUSH, GRACE, PARTIAL_SYNC]
            # request.version < master.version: return OperatorType.PULL
            if request.version < self.version_manager.version:
                logger.warning(f"version error, {request.version=} {self.version_manager.version=}")
                request.push_status = PushStatus.PUSH_STATUS_LOW_VERSION
            elif self.global_status not in [GlobalStatus.INIT, GlobalStatus.PUSH, GlobalStatus.GRACE, GlobalStatus.PARTIAL_SYNC]:
                return UpdatePushStatusResponse(op=OperatorType.RETRY)

            if self.global_status == GlobalStatus.INIT:
                self.push_state.update_state(request.group_id, request.push_status)
                self.update_global_status_history()
                self.global_status = GlobalStatus.PUSH
                logger.info("finish init, global status change to push")
                if request.push_status == PushStatus.PUSH_STATUS_LOW_VERSION:
                    return UpdatePushStatusResponse(op=OperatorType.PULL)
                else:
                    return UpdatePushStatusResponse(op=OperatorType.PUSH)
            elif self.global_status == GlobalStatus.PUSH:
                self.push_state.update_state(request.group_id, request.push_status)

                if request.push_status == PushStatus.PUSH_STATUS_FAIL:
                    self.groups_fail.add(request.group_id)

                if request.push_status == PushStatus.PUSH_STATUS_FINISH:
                    self.consume_token_state.update(request.group_id, request.consume_tokens)

                if self.update_method == "sync":
                    return self._update_push_status_sync_mode(request)
                if self.update_method == "async_buffer":
                    return self._update_push_status_async_buffer_mode(request)
                if self.update_method == "partial_sync":
                    return self._update_push_status_partial_sync_mode(request)
                raise ValueError(f"invalid update_method, update_method={self.update_method}")

            elif self.global_status == GlobalStatus.GRACE:
                if request.push_status == PushStatus.PUSH_STATUS_START:
                    return UpdatePushStatusResponse(op=OperatorType.RETRY)
                self.push_state.update_state(request.group_id, request.push_status)
                if request.push_status == PushStatus.PUSH_STATUS_FAIL:
                    self.groups_fail.add(request.group_id)

                if request.push_status == PushStatus.PUSH_STATUS_FINISH:
                    self.consume_token_state.update(request.group_id, request.consume_tokens)

                if self.push_state.all_finish(self.group_state.alive_worker() & self.received_groups):
                    self.finish_push()
                if request.push_status == PushStatus.PUSH_STATUS_LOW_VERSION:
                    return UpdatePushStatusResponse(op=OperatorType.PULL)
                else:
                    return UpdatePushStatusResponse(op=OperatorType.PUSH)

            elif self.global_status == GlobalStatus.PARTIAL_SYNC:
                self.push_state.update_state(request.group_id, request.push_status)
                if request.push_status == PushStatus.PUSH_STATUS_FAIL:
                    self.groups_fail.add(request.group_id)

                if request.push_status == PushStatus.PUSH_STATUS_FINISH:
                    self.consume_token_state.update(request.group_id, request.consume_tokens)

                if not self.push_state.contain_unfinish(self.group_state.alive_worker()):
                    self.received_groups = self.push_state.get_received_groups()
                    self.finish_push()
                if request.push_status == PushStatus.PUSH_STATUS_LOW_VERSION:
                    return UpdatePushStatusResponse(op=OperatorType.PULL)
                else:
                    return UpdatePushStatusResponse(op=OperatorType.PUSH)
            else:
                if request.push_status != PushStatus.PUSH_STATUS_LOW_VERSION:
                    raise ValueError(F"invalid global_status {self.global_status}")
                else:
                    self.received_groups.add(request.group_id)
                    return UpdatePushStatusResponse(op=OperatorType.PULL)

    def finish_push(self):
        if self.grace_check_job is not None:
            self.grace_check_job.remove()
            self.grace_check_job = None
        if self.partial_sync_check_job is not None:
            self.partial_sync_check_job.remove()
            self.partial_sync_check_job = None

        for group_id in self.groups_fail:
            self.received_groups.discard(group_id)

        self.push_fail_groups = self.push_state.get_push_fail_groups()

        self.grace_start = None
        self.partial_sync_start = None
        self.push_state.reset()
        logger.info(f"finish push, new received_groups={self.received_groups}")
        if len(self.received_groups) == 0:
            logger.error("All received groups are failed, change to PULL status")
            self.update_global_status_history()
            self.global_status = GlobalStatus.PULL
            return

        need_update_weight_groups = self.received_groups & self.group_state.alive_worker()
        logger.info(f"finish push, {need_update_weight_groups=}")
        self.send_norm_signal(need_update_weight_groups)
        if self.has_compute_norm_fail:
            self.send_step_signal(need_update_weight_groups)
            self.update_global_status_history()
            self.global_status = GlobalStatus.COMPUTE_STEP
        else:
            self.update_global_status_history()
            self.global_status = GlobalStatus.COMPUTE_NORM

        if self.global_status == GlobalStatus.COMPUTE_STEP:
            logger.info("finish push, global status change to compute step")
        else:
            logger.info("finish push, global status change to compute norm")

    def send_norm_signal(self, need_norm_groups):
        logger.info(f"send norm signal to ps, {need_norm_groups=}")
        req = StartComputeNormRequest(groups=list(need_norm_groups))
        for stub in self.ps_stubs:
            try:
                stub.StartComputeNorm(req)
            except grpc.RpcError as e:
                logger.error(f"rpc error when call StartComputeNorm: {e.code()=} {e.details()=}")
                self.has_compute_norm_fail = True

    def check_norm(self, groups: set):
        if self.has_compute_norm_fail:
            self.has_compute_norm_fail = False
            self.groups_norm.reset()
            self.norm_state.reset()
            return groups

        self.groups_norm.update_norm()
        group_norms = self.groups_norm.total_norms()
        passed_groups = set(group_norms.keys())
        for groud_id, norm in group_norms.items():
            if norm == float("inf") or norm == -float("inf"):
                passed_groups.discard(groud_id)
                logger.warning(f"group_id {groud_id}: Overflow occurs, please check it.")
            if math.isnan(norm):
                passed_groups.discard(groud_id)
                logger.warning(f"group_id {groud_id}: Nan grad norm occurs, please check it.")

        if self.use_ewma_outlier:
            # z score
            for groud_id, norm in group_norms.items():
                if self.anomaly_detector.is_outlier(norm):
                    passed_groups.discard(groud_id)
                    logger.warning(f"anomaly detected, {groud_id=}, {norm=}")
                else:
                    self.anomaly_detector.update(norm)

        logger.info(f"filter group outlier, origin groups: {groups}, {passed_groups=}")
        self.norm_state.reset()
        return passed_groups

    def get_clip_coef(self, groups_weight_factor):
        # compute new total norm
        total_norm = self.groups_norm.get_weighted_norm(groups_weight_factor)
        clip_coef = self.clip_pseudo_grad / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)
        return clip_coef

    def send_step_signal(self, need_update_weight_groups):
        self.already_send_start_update_weight_req = True
        need_update_weight_groups = self.check_norm(need_update_weight_groups)
        if len(need_update_weight_groups) == 0:
            # add fail groups to received_groups, because fail group need pull and finish pull need received_groups
            self.received_groups = self.received_groups | self.push_fail_groups
            self.update_global_status_history()
            self.global_status = GlobalStatus.PULL
            logger.warning("need_update_weight_groups is empty, global status change to PULL status")
            return

        if self.use_weighted_grad:
            cur_consume_tokens = self.consume_token_state.cur_consume_tokens(need_update_weight_groups)
            logger.info(f"cur_consume_tokens: {cur_consume_tokens}")
            groups_weight_factor = self.get_cur_groups_weight_factor(need_update_weight_groups, cur_consume_tokens)
        else:
            groups_weight_factor = {group: 1.0 / len(need_update_weight_groups) for group in need_update_weight_groups}

        if self.clip_pseudo_grad is not None:
            clip_coef = self.get_clip_coef(groups_weight_factor)
        else:
            clip_coef = 1.0
        self.groups_norm.reset()
        need_save_ckpt = self.need_save_ckpt()

        # wait for group pull ckpt
        start = time.time()
        while len(self.groups_request_for_ckpt) > 0:
            time.sleep(0.1)
            if time.time() - start > self.connect_timeout:
                logger.error("wait for group pull ckpt timeout")
                break

        logger.info(
            f"send step signal to ps, {need_update_weight_groups=}, {groups_weight_factor=},"
            f" {clip_coef=}, {need_save_ckpt=}"
        )
        factors = [
            GroupWeightFactor(group_id=group, norm=groups_weight_factor[group]) for group in need_update_weight_groups
        ]
        req = StartUpdateWeightRequest(
            groups=list(need_update_weight_groups),
            factors=factors,
            version=self.version_manager.version,
            consume_tokens=self.consume_token_state.total_tokens(),
            save_ckpt=need_save_ckpt,
            clip_coef=clip_coef,
        )
        for stub in self.ps_stubs:
            try:
                stub.StartUpdateWeight(req)
            except grpc.RpcError as e:
                logger.error(f"rpc error when call StartUpdateWeight: {e.code()=} {e.details()=}")

        # add fail groups to received_groups, because fail group need pull and finish pull need received_groups
        self.received_groups = self.received_groups | self.push_fail_groups
        self.push_fail_groups = set()
        if need_save_ckpt:
            self.save_history = True
            self.consume_token_state.step()

    def grace_check(self):
        logger.info("start grace check")
        with self.lock:
            self.grace_start = None
            if self.grace_check_job is not None:
                self.grace_check_job.remove()
                self.grace_check_job = None

            if self.global_status != GlobalStatus.PUSH:
                logger.info(f"grace_check no need run, global_status: {self.global_status}")
                return

            self.update_global_status_history()
            self.global_status = GlobalStatus.GRACE
            self.received_groups = set(self.push_state.status_dict.keys()) & self.group_state.alive_worker()
            if self.push_state.all_finish(self.received_groups):
                self.finish_push()

    def partial_sync_check(self):
        logger.info("start partial sync check")
        with self.lock:
            self.partial_sync_start = None
            if self.partial_sync_check_job is not None:
                self.partial_sync_check_job.remove()
                self.partial_sync_check_job = None

            if self.global_status != GlobalStatus.PUSH:
                logger.info(f"partial_sync_check no need run, global_status: {self.global_status}")
                return

            self.update_global_status_history()
            self.global_status = GlobalStatus.PARTIAL_SYNC
            if self.push_state.contain_unfinish(self.group_state.alive_worker()):
                logger.info("partial check: contain unfinish group, waiting for push finish")
                return
            self.received_groups = self.push_state.get_received_groups()
            self.finish_push()

    def update_compute_status(self, request: UpdateComputeStatusRequest):
        self.compute_state.check_parameter(request.ps_id, request.compute_status)
        with self.lock:
            if self.global_status != GlobalStatus.COMPUTE_STEP:
                raise ValueError(f"global status {self.global_status}, not allow update compute status")

            # update step compute status
            self.compute_state.update(request.ps_id, request.compute_status)
            compute_is_finish = self.compute_state.all_finish()
            # 0: not ready, 1: success, 2: fail
            if compute_is_finish in (1, 2):
                self.version_manager.incr()
                self.save_ckpt()
            if compute_is_finish == 1:
                self.update_global_status_history()
                self.global_status = GlobalStatus.PULL
                logger.info("finish compute, global status change to PULL")
            elif compute_is_finish == 2:
                # if compute fail, reset all state
                self.reset()
                logger.info("compute fail, global status change to INIT")
            return UpdateComputeStatusResponse()

    def update_norm_status(self, request: UpdateNormStatusRequest):
        self.norm_state.check_parameter(request.ps_id, request.norm_status)
        with self.lock:
            if self.global_status != GlobalStatus.COMPUTE_NORM:
                if self.global_status == GlobalStatus.COMPUTE_STEP:
                    logger.warning(
                        "compute step, not allow update norm status. "
                        "One possible reason is that the norm update for some PS failed."
                    )
                else:
                    raise ValueError(f"global status {self.global_status}, not allow update norm status")

            self.norm_state.update(request.ps_id, request.norm_status)
            norm_finish_status = self.norm_state.all_finish()
            if norm_finish_status in (1, 2):
                need_update_weight_groups = self.received_groups & self.group_state.alive_worker()
                self.send_step_signal(need_update_weight_groups)
                self.update_global_status_history()
                self.global_status = GlobalStatus.COMPUTE_STEP
            if norm_finish_status == 1:
                logger.info("finish compute norm, global status change to COMPUTE_STEP")
            elif norm_finish_status == 2:
                # if compute fail, reset all state
                logger.info("failed compute norm, global status change to COMPUTE_STEP")
            return UpdateNormStatusResponse()

    def query_compute_status(self):
        gs2cs = {
            GlobalStatus.PULL: ComputeStatus.COMPUTE_SUCCESS,
            GlobalStatus.COMPUTE_NORM: NormStatus.NORM_COMPUTING,
            GlobalStatus.COMPUTE_STEP: ComputeStatus.COMPUTING,
            GlobalStatus.GRACE: ComputeStatus.RECEIVING,
            GlobalStatus.PARTIAL_SYNC: ComputeStatus.RECEIVING,
            GlobalStatus.PUSH: ComputeStatus.RECEIVING,
            GlobalStatus.INIT: ComputeStatus.COMPUTE_FAIL,
            GlobalStatus.FAIL: ComputeStatus.COMPUTE_FAIL,
        }
        with self.lock:
            return QueryComputeStatusResponse(compute_status=gs2cs[self.global_status])

    def reset(self):
        self.update_global_status_history()
        self.global_status = GlobalStatus.INIT
        self.push_state.reset()
        self.compute_state.reset()
        self.pull_state.reset()
        self.received_groups = set()
        self.groups_fail = set()
        self.push_fail_groups = set()
        self.need_pull_groups = set()
        self.already_send_start_update_weight_req = False

    def update_pull_status(self, request: UpdatePullStatusRequest):
        self.pull_state.check_parameter(request.group_id, request.pull_status)
        with self.lock:
            if self.global_status != GlobalStatus.PULL:
                raise ValueError(f"global status {self.global_status}, not allow update pull status")

            self.pull_state.update(request.group_id, request.pull_status)
            if self.pull_state.all_finish(self.group_state.alive_worker() & self.received_groups):
                self.finish_pull()
            # self.update_group_status_history(self.group_status[request.group_id].get("", []), )
            # self.group_status[request.group_id][""]
            return UpdatePullStatusResponse(version=self.version_manager.version)

    def finish_pull(self):
        if self.already_send_start_update_weight_req:
            req = FinishPullWeightRequest()
            for stub in self.ps_stubs:
                try:
                    stub.FinishPullWeight(req)
                except grpc.RpcError as e:
                   logger.error(f"rpc error when call FinishPullWeight: {e.code()=} {e.details()=}")
        else:
            logger.info(f"master does not send start update weight request, no need send finish pull weight as well")
        self.reset()
        logger.info("finish pull, global status change to INIT")



    def update_client_ckpt_status(self, request: UpdateClientCkptRequest):
        group_id = request.group_id
        status = request.pull_status

        with self.lock:
            if status == PullStatus.PULL_STATUS_START:
                self.need_pull_groups.add(group_id)
                self.pull_state.update(group_id, status);
            if self.global_status == GlobalStatus.INIT:
                self.update_global_status_history()
                self.global_status = GlobalStatus.PULL
                logger.info(f"finish INIT, global status change to PULL")
            if self.global_status != GlobalStatus.PULL:
                return UpdateClientCkptResponse(op=OperatorType.RETRY)

            if status == PullStatus.PULL_STATUS_START:
                self.received_groups.add(group_id)
                return UpdateClientCkptResponse(op=OperatorType.PULL)
            elif status == PullStatus.PULL_STATUS_FINISH:
                self.need_pull_groups.discard(group_id)
                self.pull_state.update(group_id, status)
                if self.pull_state.all_finish(self.group_state.alive_worker() & self.received_groups):
                    self.finish_pull()
                return UpdateClientCkptResponse(op=OperatorType.FINISH, version=self.version_manager.version)
            elif status == PullStatus.PULL_STATUS_FAIL:
                self.need_pull_groups.discard(group_id)
                self.pull_state.update(group_id, status)
                if self.pull_state.all_finish(self.group_state.alive_worker() & self.received_groups):
                    self.finish_pull()
                return UpdateClientCkptResponse(op=OperatorType.FINISH)
            else:
                raise ValueError(f"invalid pull status: {status}")

    def update_client_weight_factor(self, request: UpdateClientWeightFactorRequest):
        with self.lock:
            if request.group_id in self.groups_weight_factor:
                logger.warning(f"weight factor already exist, {request.group_id=}")
            self.groups_weight_factor[request.group_id] = request.factor
            logger.info(f"update weight factor, {self.groups_weight_factor=}")
            return UpdateClientWeightFactorResponse()

    def update_ps_norm(self, request: UpdatePSNormRequest):
        with self.lock:
            self.groups_norm.recv_norm(request.items)
            return UpdatePSNormResponse()

    def get_cur_groups_weight_factor(self, groups: set, cur_consumed_tokens: Dict[int, int]):
        if not groups:
            raise ValueError("groups should not be empty")
        logger.info(f"get_cur_groups_weight_factor, {groups=}, {cur_consumed_tokens=}")

        total_consumed = sum(cur_consumed_tokens[group] for group in groups)

        # 避免除以零的情况
        if total_consumed == 0:
            logger.warning(f"total consumed tokens is zero, {total_consumed=}")
            raise ValueError("total consumed tokens should not be zero")

        # 计算每个组的消耗因子
        consumed_tokens_factor = {group: cur_consumed_tokens[group] / total_consumed for group in groups}

        # 获取每个组的初始权重因子
        # 一般情况下，所有alive的group均有weight_factor
        if len(self.groups_weight_factor) > 0:
            # 对于不在groups_weight_factor中的组，weight factor设置为0
            deafult_init_factor = 0.0
        else:
            deafult_init_factor = 1.0 / len(self.group_state.alive_worker())
        groups_weight_init_factor = {
            group: self.groups_weight_factor.get(group, deafult_init_factor) for group in groups
        }

        # normalize groups_weight_factor
        total_wf = sum(groups_weight_init_factor.values())
        normalized_groups_weight_init_factor = {group: wf / total_wf for group, wf in groups_weight_init_factor.items()}
        logger.info(f"get_cur_groups_weight_factor, {normalized_groups_weight_init_factor=}")

        # 计算权重因子的乘积
        weight_factor = {group: normalized_groups_weight_init_factor[group] * consumed_tokens_factor[group] for group in groups}

        # 计算权重因子的总和以进行归一化
        total_weight = sum(weight_factor.values())

        if total_weight == 0:
            logger.warning(f"total weight is zero, {weight_factor=}, {consumed_tokens_factor=}")
            logger.warning(f"total weight is zero, {normalized_groups_weight_init_factor=}, {self.groups_weight_factor=}")
            raise ValueError("total weight should not be zero")

        # 归一化权重因子
        normalized_weight_factor = {group: wf / total_weight for group, wf in weight_factor.items()}

        logger.info(f"get_cur_groups_weight_factor, {normalized_weight_factor=}")

        return normalized_weight_factor

    def query_global_status(self, request: QueryGlobalStatusRequest):
        with self.lock:
            current_status = CurrentGlobalStatus(status_name="TEST")
            history_status = [HistoryGlobalStatus(status_name="TEST2", start_time="yesterday", end_time="now"), HistoryGlobalStatus(status_name="TEST3", start_time="yesterday2", end_time="now2")]
            return QueryGlobalStatusResponse(
                current_global_status=current_status,
                history_global_status_list=history_status,
            )
    
    def get_global_status_history_list(self, last_of_n = 10):
        with self.lock:
            return self.global_status_history_list[-last_of_n:]
    
    def get_current_global_status(self):
        with self.lock:
            return self.global_status.name
    
    def update_global_status_history(self):
        current_global_status_interval = dict(status_name = self.global_status.name, begin_timestamp = self.current_global_status_begin_timestamp, end_timestamp=time.time())
        self.current_global_status_begin_timestamp = current_global_status_interval["end_timestamp"]
        if len(self.global_status_history_list) != 0 and self.global_status_history_list[-1]["status_name"] == current_global_status_interval["status_name"]:
            current_global_status_interval["begin_timestamp"] = self.global_status_history_list[-1]["begin_timestamp"]
            self.global_status_history_list.pop()
        self.global_status_history_list.append(current_global_status_interval)
    
    def update_group_status_history(self, current_group_status, group_status_history_list, current_group_status_begin_timestamp):
        current_group_status_interval = dict(status_name = current_group_status, begin_timestamp = current_group_status_begin_timestamp, end_timestamp=time.time())
        current_group_status_begin_timestamp = current_group_status_interval["end_timestamp"]
        if len(group_status_history_list) != 0 and group_status_history_list[-1]["status_name"] == current_group_status_interval["status_name"]:
            current_group_status_interval["begin_timestamp"] = group_status_history_list[-1]["begin_timestamp"]
            group_status_history_list.pop()
        group_status_history_list.append(current_group_status_interval)
    
    def get_alive_ps(self):
        with self.lock:
            return repr(self.ps_state)


    def get_alive_group(self):
        with self.lock:
            return repr(self.group_state)

    def get_group_status(self, group_id = 0, last_of_n = 10):
        with self.lock:
            all_keys = set(self.push_state.status_history_dict.keys()) | \
                   set(self.pull_state.status_history_dict.keys()) | \
                   set(self.group_state.status_history_dict.keys())
            result = dict()
            for group_id in sorted(all_keys):
                meta_event_list = sorted(self.push_state.status_history_dict.get(group_id, []) + self.pull_state.status_history_dict.get(group_id, []) + self.group_state.status_history_dict.get(group_id, []), key=lambda x: x[0])[-last_of_n:]
                group_status = dict()
                readable_event_list = []
                for i in range(1, len(meta_event_list)):
                    last_event = meta_event_list[i - 1]
                    current_event = meta_event_list[i]
                    event_dict = dict()
                    if last_event[1] == "PULL_FINISH" and current_event[1] == "PUSH_START":
                        event_dict["event_name"] = "TRAIN"
                    elif last_event[1] == "PUSH_START" and current_event[1] == "PUSH_FINISH":
                        event_dict["event_name"] = "PUSH"
                    elif last_event[1] == "PULL_START" and current_event[1] == "PULL_FINISH":
                        event_dict["event_name"] = "PULL"
                    elif last_event[1] == "PUSH_FINISH" and current_event[1] == "PULL_START":
                        event_dict["event_name"] = "WAIT"
                    else:
                        event_dict["event_name"] = "ABNORMAL"

                    event_dict["start_timestamp"] = last_event[0]
                    event_dict["end_timestamp"] = current_event[0]
                    readable_event_list.append(event_dict)

                group_status["history_status_list"] = readable_event_list
                if meta_event_list[-1][1] == "PULL_FINISH":
                    group_status["current_status"] = "TRAIN"
                elif meta_event_list[-1][1] == "PUSH_START":
                    group_status["current_status"] = "PUSH"
                elif meta_event_list[-1][1] == "PULL_START":
                    group_status["current_status"] = "PULL"
                elif meta_event_list[-1][1] == "PUSH_FINISH":
                    group_status["current_status"] = "WAIT"
                else:
                    group_status["current_status"] = "ABNORMAL"
                result[group_id] = group_status

            return result


