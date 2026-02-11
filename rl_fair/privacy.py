from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import hashlib


def stable_hash(text: str, salt: str = "") -> str:
    h = hashlib.sha256((salt + text).encode("utf-8")).hexdigest()
    return h[:12]


@dataclass
class SafeLogger:
    """A minimal logger that avoids printing sensitive column names/values.

    You can still get progress feedback without exposing schema details.
    """
    enabled: bool = True
    prefix: str = "[rl-fair]"

    def log(self, msg: str):
        if self.enabled:
            print(f"{self.prefix} {msg}")

    def log_metrics(self, step: int, perf: float, disparity: float, reward: float):
        if self.enabled:
            print(f"{self.prefix} step={step:04d} perf={perf:.4f} disparity={disparity:.4f} reward={reward:.4f}")
