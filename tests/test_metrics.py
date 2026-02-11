import numpy as np
from rl_fair.metrics.classification import demographic_parity_diff, equal_opportunity_diff

def test_dp_eo():
    y_true = np.array([0,0,1,1,0,1])
    y_pred = np.array([0,1,1,1,0,0])
    group = np.array([0,0,0,1,1,1])
    dp = demographic_parity_diff(y_pred, group)
    eo = equal_opportunity_diff(y_true, y_pred, group)
    assert dp >= 0
    assert eo >= 0
