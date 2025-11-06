import argparse, os
import numpy as np
from pathlib import Path
from .env import SimpleFairEnv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=1000)
    ap.add_argument('--lam', type=float, default=1.0, help='fairness penalty weight (future use)')
    ap.add_argument('--outdir', type=str, default='runs/demo')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    env = SimpleFairEnv()

    returns = []
    for ep in range(args.episodes):
        env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = np.random.randint(env.n_actions)
            step = env.step(action)
            ep_ret += step.reward
            done = step.done
        returns.append(ep_ret)

    with open(Path(args.outdir)/'train.log', 'w') as f:
        for r in returns:
            f.write(f"{r}\n")
    print(f"[done] episodes={args.episodes} avg_return={np.mean(returns):.3f}")

if __name__ == '__main__':
    main()
