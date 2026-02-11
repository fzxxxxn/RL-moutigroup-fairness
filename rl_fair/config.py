from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence, Dict, Any


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset schema declaration.

    You must provide:
      - feature_cols: the input feature columns
      - label_col: ground-truth target label column
      - protected_cols: columns used to define groups (intersectional groups are supported)
    """
    feature_cols: Sequence[str]
    label_col: str
    protected_cols: Sequence[str]

    # Optional: provide your own grouping function. It must return a 1D array-like
    # of group ids (ints or strings) aligned with df rows.
    grouping_fn: Optional[Callable[[Any], Any]] = None


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the reward function.

    We define fairness as "lower disparity is better" (0 is ideal).
    Reward = -(lambda_fair * disparity + lambda_acc * max(0, baseline_perf - perf))
    """
    lambda_fair: float = 1.0
    lambda_acc: float = 1.0
    # Optional regularization to discourage extreme weights.
    lambda_entropy: float = 0.01


@dataclass
class TrainConfig:
    """Training settings for the RL loop."""
    seed: int = 42

    # Data split
    val_size: float = 0.2

    # Resampling
    total_samples: Optional[int] = None  # if None, use len(train_split)
    min_group_weight: float = 0.02
    max_group_weight: float = 0.98

    # RL
    num_steps: int = 200
    warmup_steps: int = 20
    gamma: float = 0.98
    lr: float = 3e-4
    batch_size: int = 64
    replay_size: int = 10_000
    target_update_every: int = 25
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 150

    # Weight adjustment action granularity
    delta: float = 0.05

    # Model training
    # model_kwargs are forwarded to the model factory.
    model_name: str = "logreg"  # "logreg" | "mlp" | "rf"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Metrics / logging
    verbose: bool = True
