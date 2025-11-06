from typing import Dict
import numpy as np

def composite_reward(acc: float, fairness_mean: float, fair_disp: Dict[str, float], acc_w: float = 0.5) -> float:
    # Higher fairness_mean (BPSN-AUC) and lower dispersion are better
    disp_penalty = (fair_disp.get("range", 0.0) + fair_disp.get("std", 0.0))  # simple penalty
    return acc_w*acc + (1-acc_w)*(fairness_mean - disp_penalty)
