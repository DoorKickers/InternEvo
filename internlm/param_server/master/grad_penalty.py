import math
from collections import deque
from typing import Any, Dict, List

import numpy as np
import torch
from loguru import logger


class GroupNorm:
    def __init__(self):
        self.recv_group_norms: Dict[int, List] = {}
        self.group_total_norms: Dict[int, float] = {}

    def recv_norm(self, group_norms):
        logger.info("begin recv group norms")
        for item in group_norms:
            if item.group_id not in self.recv_group_norms:
                self.recv_group_norms[item.group_id] = []
            self.recv_group_norms[item.group_id].append(item.norm)
        logger.info(f"recv_group_norms: {self.recv_group_norms}")

    def update_norm(self):
        for group_id, norms in self.recv_group_norms.items():
            self.group_total_norms[group_id] = math.sqrt(sum(norms))
        logger.info(f"Group norms: {self.group_total_norms}")

    def total_norms(self):
        return self.group_total_norms

    def reset(self):
        self.recv_group_norms.clear()
        self.group_total_norms.clear()

    def check_all_received(self, group_ids: set, num_ps):
        recv_groups = set(self.recv_group_norms.keys())
        if len(recv_groups - group_ids) > 0:
            return False
        for group_id in group_ids:
            if len(self.recv_group_norms[group_id]) != num_ps:
                return False
        return True

    def get_weighted_norm(self, groups_weight_factor):
        weighted_total_norms = {}
        for group_id, weight_factor in groups_weight_factor.items():
            if group_id not in self.recv_group_norms:
                logger.warning(f"group_id {group_id} not in recv_group_norms!")
                continue
            norms = self.recv_group_norms[group_id]
            weighted_total_norms[group_id] = np.sum(np.array(norms) * (weight_factor**2))

        weighted_total_norm = math.sqrt(sum(weighted_total_norms.values()))
        return weighted_total_norm


class OnlineDynamicEWMA:
    def __init__(self, alpha=0.02, warmup_steps=100, base_threshold=3):
        self.alpha = alpha
        self.mean = None
        self.M2 = None
        self.count = 0
        self.warmup_steps = warmup_steps
        self.base_threshold = base_threshold
        self.z_scores_window = deque(maxlen=warmup_steps)

    def state_dict(self):
        state_dict: Dict[str, Any] = {}
        state_dict["mean"] = self.mean
        state_dict["M2"] = self.M2
        state_dict["count"] = self.count
        state_dict["warmup_steps"] = self.warmup_steps
        state_dict["z_scores_window"] = list(self.z_scores_window)
        return state_dict

    def load_state_dict(self, state_dict: Dict):
        self.mean = state_dict.get("mean", self.mean)
        self.M2 = state_dict.get("M2", self.M2)
        self.count = state_dict.get("count", self.count)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.z_scores_window = deque(state_dict.get("z_scores_window", self.z_scores_window), maxlen=self.warmup_steps)

    def update(self, value):
        if self.mean is None:
            self.mean = torch.tensor(0.0)
            self.M2 = torch.tensor(0.0)

        self.count += 1
        delta = value - self.mean
        self.mean += self.alpha * delta
        delta2 = value - self.mean
        self.M2 = (1 - self.alpha) * (self.M2 + self.alpha * delta * delta2)
        self.z_scores_window.append(self.get_z_score(value))

    def get_variance(self):
        if self.count < 2:
            return torch.tensor(0.0)  # Not enough data to compute variance
        return self.M2

    def get_std(self):
        return torch.sqrt(self.get_variance())

    def get_mean(self):
        return self.mean

    def get_z_score(self, value):
        std_dev = self.get_std()
        if (std_dev == 0) or (self.count < self.warmup_steps):
            return torch.tensor(0.0)  # No variation in data
        return (value - self.mean) / std_dev

    def dynamic_threshold_factor(self):
        if self.count < self.warmup_steps:
            return torch.tensor(1.0)
        std_recent = torch.stack(list(self.z_scores_window)).std()
        return torch.max(torch.tensor(1.0), std_recent)

    def is_outlier(self, norm):
        if norm == float("inf") or norm == -float("inf"):
            logger.warning("Overflow occurs, please check it.")
            return True
        if math.isnan(norm):
            logger.warning("Nan grad norm occurs, please check it.")
            return True

        if self.count < self.warmup_steps:
            return False  # Skip outlier detection during warm-up period

        z_score = self.get_z_score(norm)
        threshold = self.base_threshold * self.dynamic_threshold_factor()
        logger.info(f"z_score: {z_score}, threshold: {threshold}")
        return z_score > threshold  # Negative z-score is good for gradnorm
