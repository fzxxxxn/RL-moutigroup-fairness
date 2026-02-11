from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from ..config import DatasetSpec, TrainConfig, RewardConfig
from ..privacy import SafeLogger
from .env import ReweightingEnv
from .replay_buffer import ReplayBuffer
from .dqn_agent import DQNAgent, DQNConfig


@dataclass
class TrainResult:
    best_weights: np.ndarray
    best_perf: float
    best_disparity: float
    baseline_perf: float
    baseline_disparity: float
    history: Dict[str, list]


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return float(end)
    t = step / max(1, decay_steps)
    return float(start + t * (end - start))


def train_reweighter(
    df: pd.DataFrame,
    spec: DatasetSpec,
    train_cfg: TrainConfig = TrainConfig(),
    reward_cfg: RewardConfig = RewardConfig(),
    *,
    perf_metric: str = "auc",
    disparity_metric: str = "dp",
) -> TrainResult:
    """Run DQN to learn group weights that optimize fairness/accuracy trade-off.

    Returns:
      TrainResult with best weights and training history.

    Privacy:
      - This function will not print feature/label/protected column names.
      - Do not commit your config that contains sensitive column names if that is a concern.
    """
    log = SafeLogger(enabled=train_cfg.verbose)

    env = ReweightingEnv(
        df=df,
        spec=spec,
        train_cfg=train_cfg,
        reward_cfg=reward_cfg,
        perf_metric=perf_metric,
        disparity_metric=disparity_metric,
        cache=True,
    )

    obs = env.reset()
    rb = ReplayBuffer(train_cfg.replay_size, env.obs_dim)

    agent = DQNAgent(env.obs_dim, env.act_dim, DQNConfig(gamma=train_cfg.gamma, lr=train_cfg.lr, target_update_every=train_cfg.target_update_every))
    rng = np.random.default_rng(train_cfg.seed)

    best_perf, best_disp = env.baseline_perf, env.baseline_disp
    best_w = env.weights.copy()

    hist = {"perf": [], "disparity": [], "reward": [], "epsilon": [], "loss": []}

    for step in range(train_cfg.num_steps):
        eps = linear_epsilon(step, train_cfg.epsilon_start, train_cfg.epsilon_end, train_cfg.epsilon_decay_steps)
        a = agent.act(obs, eps, rng)
        obs2, r, done, info = env.step(a)

        rb.push(obs, a, r, obs2, done)
        obs = obs2

        loss = 0.0
        if rb.size >= train_cfg.warmup_steps:
            batch = rb.sample(train_cfg.batch_size, rng)
            loss = agent.update(batch, rng)
            agent.maybe_update_target(step)

        # track best: prioritize lower disparity, break ties by higher perf
        if (info.disparity < best_disp - 1e-6) or (abs(info.disparity - best_disp) <= 1e-6 and info.perf > best_perf):
            best_disp = info.disparity
            best_perf = info.perf
            best_w = info.weights.copy()

        hist["perf"].append(info.perf)
        hist["disparity"].append(info.disparity)
        hist["reward"].append(info.reward)
        hist["epsilon"].append(eps)
        hist["loss"].append(loss)

        if train_cfg.verbose and (step % 10 == 0 or step == train_cfg.num_steps - 1):
            log.log_metrics(step=step, perf=info.perf, disparity=info.disparity, reward=info.reward)

        if done:
            break

    return TrainResult(
        best_weights=best_w,
        best_perf=float(best_perf),
        best_disparity=float(best_disp),
        baseline_perf=float(env.baseline_perf),
        baseline_disparity=float(env.baseline_disp),
        history=hist,
    )
