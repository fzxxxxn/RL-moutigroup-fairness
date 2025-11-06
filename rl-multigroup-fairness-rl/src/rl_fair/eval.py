import argparse
import pandas as pd
from .metrics import aggregate_fairness

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', type=str, required=True, help='CSV with columns: y_true,y_pred,group')
    ap.add_argument('--w_dp', type=float, default=1.0)
    ap.add_argument('--w_eo', type=float, default=1.0)
    args = ap.parse_args()

    df = pd.read_csv(args.pred)
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    groups = df['group'].values
    score = aggregate_fairness(y_true, y_pred, groups, {'dp': args.w_dp, 'eo': args.w_eo})
    print(f"aggregate_fairness={score:.6f}")

if __name__ == '__main__':
    main()
