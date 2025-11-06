def reward(task_reward: float, fairness_penalty: float, lam: float = 1.0) -> float:
    return task_reward - lam * fairness_penalty
