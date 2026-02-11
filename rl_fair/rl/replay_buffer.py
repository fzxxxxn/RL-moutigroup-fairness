from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self._s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._a = np.zeros((capacity,), dtype=np.int64)
        self._r = np.zeros((capacity,), dtype=np.float32)
        self._s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._d = np.zeros((capacity,), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def push(self, s, a, r, s2, done):
        i = self._ptr
        self._s[i] = s
        self._a[i] = a
        self._r[i] = r
        self._s2[i] = s2
        self._d[i] = 1.0 if done else 0.0
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        n = self._size
        idx = rng.integers(0, n, size=int(batch_size))
        return (
            self._s[idx],
            self._a[idx],
            self._r[idx],
            self._s2[idx],
            self._d[idx],
        )

    @property
    def size(self) -> int:
        return self._size
