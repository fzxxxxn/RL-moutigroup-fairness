"""RL Fair Toolkit.

A small, pluggable module that uses reinforcement learning to search for
group-level sampling / weighting policies that improve fairness while
minimizing predictive performance loss.

This package contains NO datasets and does not hard-code any protected
attribute names. You provide column mappings at runtime.
"""

from .config import DatasetSpec, TrainConfig, RewardConfig
from .grouping import make_intersectional_groups
from .rl.train import train_reweighter
from .rl.env import ReweightingEnv
