# RL for Multi-Group Fairness (Pluggable)

A production-ready, **pluggable** framework to optimize **multi-group fairness** with **reinforcement learning (RL)**
as an adaptive, data-centric pre-processing step. Includes a synthetic demo dataset, CLI utilities, and a simple
policy-gradient agent that learns sample weights to jointly improve **multi-group (intersectional) BPSN-AUC equity**
and **predictive accuracy**.

> This open-source implementation is organized and generalized from an academic study on multi-group fairness with RL. See the manuscript for broader context and discussion. 

## Highlights

- **Interfaces-first** design: swap data sources, metrics, rewards, and models via Python entry points or config.
- **Multi-group metrics**: BPSN-AUC per subgroup, dispersion (range/std/var), and composite fairness score.
- **RL pre-processing**: light-weight policy-gradient agent learns per-slice weights; **no inference-time overhead**.
- **Model-agnostic**: use any scikit-learn compatible classifier via config.
- **Reproducible**: pinned requirements, unit tests, synthetic demo, and CI workflow.
- **Privacy-aware**: sensitive features only used during training-time audits / reweighting; not required at inference.

## Quickstart

```bash
# 1) Create env and install
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# 2) Run a complete demo (train baseline, run RL weighting, evaluate fairness)
rlmgf data.validate --schema configs/schema_demo.json --csv data/demo/demo.csv
rlmgf train --config configs/demo.yaml --out runs/baseline
rlmgf rl-train --config configs/demo.yaml --episodes 10 --out runs/rl_demo
rlmgf eval --pred runs/rl_demo/preds.csv --groups data/demo/groups.json --report runs/rl_demo/report.json
```

## Project layout

```
rl-mg-fairness/
├─ src/rlmgf/                # Library (interfaces + reference implementations)
├─ bin/                      # CLI entry wrapper
├─ configs/                  # YAML/JSON configs and schemas
├─ data/demo/                # Synthetic toy dataset + group definition
├─ notebooks/                # (optional) exploration space
├─ tests/                    # Unit tests
└─ docs/                     # Minimal docs
```

## Citation

If you use this repository, please cite the associated study motivating the design and metrics.


## Git & CI Setup

Initialize the repository and push to GitHub:

```bash
git init
git add .
git commit -m "Initial commit: RL multi-group fairness (pluggable)"
git branch -M main
git remote add origin https://github.com/<your-username>/rl-mg-fairness.git
git push -u origin main
```

GitHub Actions will run tests on each push to `main` via `.github/workflows/python-ci.yml`.
