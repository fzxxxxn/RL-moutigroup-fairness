from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import DatasetSpec, RewardConfig, TrainConfig
from ..grouping import make_intersectional_groups, encode_groups
from ..resampling import normalize_weights, resample_by_group
from ..metrics.classification import accuracy as acc_fn, auc_roc, demographic_parity_diff, equal_opportunity_diff
from ..metrics.abroca import abroca_disparity
from ..models.sklearn_wrapper import make_sklearn_model, fit_predict


@dataclass
class StepInfo:
    perf: float
    disparity: float
    reward: float
    weights: np.ndarray


class ReweightingEnv:
    """A lightweight environment for RL-based group reweighting.

    Observation: concat([weights, perf, disparity])
    Action: discrete adjustments (increase or decrease one group's weight by delta)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        spec: DatasetSpec,
        train_cfg: TrainConfig,
        reward_cfg: RewardConfig,
        *,
        perf_metric: str = "auc",  # "auc" or "acc"
        disparity_metric: str = "dp",  # "dp" | "eo" | "abroca" (abroca only for binary protected attr)
        cache: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.spec = spec
        self.train_cfg = train_cfg
        self.reward_cfg = reward_cfg
        self.perf_metric = perf_metric
        self.disparity_metric = disparity_metric
        self.cache = cache

        self.rng = np.random.default_rng(train_cfg.seed)

        # build group ids
        if spec.grouping_fn is not None:
            raw_groups = spec.grouping_fn(self.df)
        else:
            raw_groups = make_intersectional_groups(self.df, spec.protected_cols)
        self.group_raw = np.asarray(raw_groups)
        self.group_ids, self.group_labels = encode_groups(self.group_raw)
        self.n_groups = int(self.group_ids.max()) + 1

        # train/val split
        train_idx, val_idx = train_test_split(
            np.arange(len(self.df)),
            test_size=float(train_cfg.val_size),
            random_state=int(train_cfg.seed),
            shuffle=True,
            stratify=None,
        )
        self.df_train = self.df.iloc[train_idx].reset_index(drop=True)
        self.df_val = self.df.iloc[val_idx].reset_index(drop=True)
        self.g_train = self.group_ids[train_idx]
        self.g_val = self.group_ids[val_idx]

        self.total_samples = int(train_cfg.total_samples or len(self.df_train))

        # initial weights uniform
        self.weights = np.ones(self.n_groups, dtype=float) / self.n_groups

        # cache
        self._eval_cache: Dict[Tuple[float, ...], Tuple[float, float]] = {}


        # baseline metrics (on original data distribution)
        self.baseline_perf, self.baseline_disp = self._evaluate(self.weights)

        # cache

        self.step_count = 0

        # Action space: for each group -> (increase) and (decrease)
        self.act_dim = self.n_groups * 2

    @property
    def obs_dim(self) -> int:
        return self.n_groups + 2

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.weights = np.ones(self.n_groups, dtype=float) / self.n_groups

        # cache

        perf, disp = self._evaluate(self.weights)
        return self._obs(perf, disp)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        self.step_count += 1
        w = self.weights.copy()
        g = int(action) // 2
        direction = +1 if (int(action) % 2 == 0) else -1
        delta = float(self.train_cfg.delta) * direction

        w[g] = w[g] + delta
        w = normalize_weights(w, self.train_cfg.min_group_weight, self.train_cfg.max_group_weight)

        perf, disp = self._evaluate(w)
        reward = self._reward(perf, disp, w)

        self.weights = w

        done = self.step_count >= int(self.train_cfg.num_steps)
        info = StepInfo(perf=perf, disparity=disp, reward=reward, weights=w.copy())
        return self._obs(perf, disp), reward, done, info

    def _obs(self, perf: float, disp: float) -> np.ndarray:
        return np.concatenate([self.weights.astype(np.float32), np.array([perf, disp], dtype=np.float32)])

    def _reward(self, perf: float, disp: float, w: np.ndarray) -> float:
        # Disparity is "lower is better". Penalize disparity directly.
        fair_term = self.reward_cfg.lambda_fair * disp

        # Performance term: penalize dropping below baseline.
        acc_drop = max(0.0, self.baseline_perf - perf)
        perf_term = self.reward_cfg.lambda_acc * acc_drop

        # Entropy regularization (discourage peaky distributions)
        ent = -float(np.sum(w * np.log(np.clip(w, 1e-12, 1.0))))
        ent_term = -self.reward_cfg.lambda_entropy * ent  # higher entropy -> less penalty (more negative)

        return - (fair_term + perf_term) + ent_term

    def _evaluate(self, w: np.ndarray) -> tuple[float, float]:
        # caching by rounded weights to avoid redundant model fits
        key = tuple(np.round(w, 3).tolist())

        # resample training set according to w
        df_s = resample_by_group(self.df_train, self.g_train, w, self.total_samples, replace=True, rng=self.rng)

        X_train = df_s[list(self.spec.feature_cols)].to_numpy()
        y_train = df_s[self.spec.label_col].to_numpy()

        X_val = self.df_val[list(self.spec.feature_cols)].to_numpy()
        y_val = self.df_val[self.spec.label_col].to_numpy()
        g_val = self.g_val

        model = make_sklearn_model(self.train_cfg.model_name, **self.train_cfg.model_kwargs)
        y_pred, y_score = fit_predict(model, X_train, y_train, X_val)

        # performance metric
        if self.perf_metric == "acc":
            perf = acc_fn(y_val, y_pred)
        else:
            perf = auc_roc(y_val, y_score)

        # disparity metric
        if self.disparity_metric == "dp":
            disp = demographic_parity_diff(y_pred, g_val)
        elif self.disparity_metric == "eo":
            disp = equal_opportunity_diff(y_val, y_pred, g_val)
        elif self.disparity_metric == "abroca":
            # only valid if there are exactly 2 protected groups; treat groups as protected binary
            if self.n_groups != 2:
                # If intersectional groups are used, abroca isn't defined; fall back to EO.
                disp = equal_opportunity_diff(y_val, y_pred, g_val)
            else:
                # y_score must be 1D in binary classification
                score_1d = y_score if np.ndim(y_score) == 1 else (y_score[:,1] if y_score.shape[1]==2 else y_score[:,0])
                disp = abroca_disparity(y_val, score_1d, g_val)
        else:
            raise ValueError(f"Unknown disparity_metric: {self.disparity_metric}")

        out = (float(perf), float(disp))
        if self.cache:
            self._eval_cache[key] = out
        return out
