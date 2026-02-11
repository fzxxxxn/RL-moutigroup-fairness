from __future__ import annotations

from typing import Iterable, List, Sequence, Any
import numpy as np
import pandas as pd


def make_intersectional_groups(df: pd.DataFrame, protected_cols: Sequence[str]) -> np.ndarray:
    """Create intersectional group ids from protected columns.

    Returns an array of group labels encoded as strings, but WITHOUT printing
    or logging the original column names or values.

    Implementation detail:
      group_id = "v1|v2|..." for the values in the given protected_cols.
    """
    if len(protected_cols) == 0:
        raise ValueError("protected_cols must be non-empty")

    # Use astype(str) to handle mixed types safely.
    parts = [df[c].astype(str) for c in protected_cols]
    group = parts[0]
    for p in parts[1:]:
        group = group + "|" + p
    return group.to_numpy()


def encode_groups(groups: np.ndarray) -> tuple[np.ndarray, List[str]]:
    """Encode arbitrary group labels into contiguous integers."""
    uniq = pd.unique(groups)
    mapping = {g: i for i, g in enumerate(uniq)}
    encoded = np.array([mapping[g] for g in groups], dtype=np.int64)
    labels = [str(g) for g in uniq]
    return encoded, labels
