from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.98
    lr: float = 3e-4
    target_update_every: int = 25


class DQNAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: DQNConfig):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg

        self.q = QNet(obs_dim, act_dim)
        self.q_targ = QNet(obs_dim, act_dim)
        self.q_targ.load_state_dict(self.q.state_dict())

        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, self.act_dim))
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q = self.q(x).squeeze(0).cpu().numpy()
        return int(np.argmax(q))

    def update(self, batch, rng: np.random.Generator) -> float:
        s, a, r, s2, d = batch
        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64)
        r = torch.tensor(r, dtype=torch.float32)
        s2 = torch.tensor(s2, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)

        q_sa = self.q(s).gather(1, a.view(-1,1)).squeeze(1)

        with torch.no_grad():
            q_next = self.q_targ(s2).max(dim=1).values
            target = r + (1.0 - d) * self.cfg.gamma * q_next

        loss = F.smooth_l1_loss(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()
        return float(loss.item())

    def maybe_update_target(self, step: int):
        if step % self.cfg.target_update_every == 0:
            self.q_targ.load_state_dict(self.q.state_dict())
