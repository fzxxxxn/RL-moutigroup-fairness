# RL for Multi‑Group Fairness

This repository packages the notebook `RL_FAIR1.ipynb` into a reproducible project for **reinforcement learning (RL) to optimize multi‑group fairness**.

**中文**：本仓库将你的 `.ipynb` 整理为标准 GitHub 结构，支持基于强化学习优化“多组公平性”。

## Features
- Modular code in `src/rl_fair/`:
  - `data.py`: data loading / schema checks
  - `metrics.py`: DP/EO/TPR‑FPR 等跨组公平指标与聚合
  - `env.py`, `rewards.py`: RL 环境与含公平惩罚的奖励
  - `train.py`, `eval.py`: 训练与评估 CLI
- Original notebook preserved in `notebooks/`
- Example config in `configs/demo.yaml`

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Train (toy env)
python -m rl_fair.train --episodes 500 --lam 1.0 --outdir runs/demo

# Evaluate fairness on predictions (CSV columns: y_true,y_pred,group)
python -m rl_fair.eval --pred examples/dummy_preds.csv
```
Python: 3.11.8

## Data
Place raw data in `data/raw/`, then generate processed files in `data/processed/` with columns:
```
feature_*, label, group
```

## Citation
If you use this template:
- Title: RL for Multi‑Group Fairness
- Year: 2025
- Type: GitHub template

## License
MIT
