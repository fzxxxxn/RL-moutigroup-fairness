#!/usr/bin/env bash
set -e
python -m rl_fair.train --episodes 500 --lam 1.0 --outdir runs/demo
