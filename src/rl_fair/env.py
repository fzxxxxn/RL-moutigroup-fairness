from dataclasses import dataclass
import numpy as np

@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict

class SimpleFairEnv:
    def __init__(self, n_features=10, n_actions=5, seed=42):
        self.rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.n_actions = n_actions
        self.reset()

    def reset(self):
        self.t = 0
        self.obs = self.rng.normal(size=(self.n_features,)).astype('float32')
        return self.obs

    def step(self, action: int) -> StepResult:
        task_r = self.rng.normal(loc=action*0.1, scale=0.1)
        fairness_penalty = self.rng.random()*0.2
        r = float(task_r - fairness_penalty)
        self.t += 1
        done = self.t >= 100
        self.obs = self.rng.normal(size=(self.n_features,)).astype('float32')
        return StepResult(self.obs, r, done, {'task': float(task_r), 'fair': float(fairness_penalty)})
