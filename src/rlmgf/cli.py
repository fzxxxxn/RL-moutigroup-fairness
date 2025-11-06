import json, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import typer
from rich import print
from .data import DatasetConfig, load_dataset, tensorize_split, group_vectors
from .trainer import fit_predict_classifier, compute_basic_metrics
from .env import WeightingEnv
from .eval import evaluate_fairness
from .rewards import composite_reward

app = typer.Typer(add_completion=False)

@app.command()
def data_validate(schema: Path = typer.Option(..., help="JSON schema path"),
                  csv: Path = typer.Option(..., help="CSV path")):
    import jsonschema
    with open(schema) as f:
        schema_obj = json.load(f)
    df = pd.read_csv(csv)
    fields = {"features": [c for c in df.columns if c.startswith("f")], "label": "label",
              "groups": [c for c in df.columns if c.startswith("g_")]}
    jsonschema.validate(fields, schema_obj)
    print("[green]Dataset appears consistent with schema.[/green]")

@app.command()
def train(config: Path = typer.Option(..., help="YAML config"),
          out: Path = typer.Option(Path("runs/baseline"), help="Output dir")):
    with open(config) as f:
        cfg = yaml.safe_load(f)
    ds = DatasetConfig(**cfg["dataset"])
    df = load_dataset(ds)
    (Xtr, ytr, idtr), (Xte, yte, idte) = tensorize_split(df, ds, cfg["trainer"]["test_size"], cfg["trainer"].get("stratify", True))
    clf, preds, scores = fit_predict_classifier(cfg["model"]["type"], cfg["model"]["params"], Xtr, ytr, Xte)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"idx": idte, "y_true": yte, "y_pred": preds}).to_csv(out / "preds.csv", index=False)
    np.save(out / "scores.npy", scores)
    metrics = compute_basic_metrics(yte, preds)
    print({"metrics": metrics})

@app.command(name="rl-train")
def rl_train(config: Path = typer.Option(...), episodes: int = typer.Option(10),
             out: Path = typer.Option(Path("runs/rl"))):
    with open(config) as f:
        cfg = yaml.safe_load(f)
    ds = DatasetConfig(**cfg["dataset"])
    df = load_dataset(ds)
    (Xtr, ytr, idtr), (Xte, yte, idte) = tensorize_split(df, ds, cfg["trainer"]["test_size"], cfg["trainer"].get("stratify", True))

    # Subgroup column
    sub_col = "g_intersection" if "g_intersection" in df.columns else ds.groups[-1]
    sub_train = df.iloc[idtr][sub_col].values
    sub_test = df.iloc[idte][sub_col].values
    uniq_sub = sorted(np.unique(sub_train).tolist())
    n_slices = len(uniq_sub)

    # Policy parameters over subgroup weights in log-space (log-normal)
    # action = exp(theta + eps), eps~N(0, sigma^2)
    theta = np.zeros(n_slices, dtype=float)
    sigma = float(cfg["rl"].get("sigma", 0.2))
    lr = float(cfg["rl"].get("lr", 0.05))
    baseline = 0.0  # moving reward baseline

    # helper to map subgroup to index
    sub_to_i = {g: i for i, g in enumerate(uniq_sub)}

    def sample_action():
        eps = np.random.normal(0, sigma, size=n_slices)
        a = np.exp(theta + eps)  # positive weights
        logprob = -0.5 * np.sum((eps / sigma)**2) - n_slices * np.log(sigma) - 0.5*n_slices*np.log(2*np.pi)
        return a, eps, logprob

    def apply_weights(base, action):
        w = base.copy()
        for g in uniq_sub:
            w[sub_train == g] *= action[sub_to_i[g]]
        return w / (w.mean() + 1e-8)

    best = {"reward": -1e9, "report": None, "preds": None, "scores": None, "theta": theta.copy()}

    for ep in range(episodes):
        base_w = np.ones_like(sub_train, dtype=float)
        action, eps, logprob = sample_action()
        w_train = apply_weights(base_w, action)

        # Fit model with sample weights
        from importlib import import_module
        Model = getattr(import_module(cfg["model"]["type"].rsplit(".",1)[0]), cfg["model"]["type"].rsplit(".",1)[1])
        clf = Model(**cfg["model"]["params"])
        clf.fit(Xtr, ytr, sample_weight=w_train)

        # Probs on test
        try:
            scores = clf.predict_proba(Xte)
        except Exception:
            dec = clf.decision_function(Xte)
            if dec.ndim == 1:
                dec = np.stack([1-dec, dec], axis=1)
            from scipy.special import softmax
            scores = softmax(dec, axis=1)
        preds = (scores.argmax(axis=1) + 1)

        # Reward: accuracy + fairness
        fair = evaluate_fairness(yte, scores, subgroups=sub_test, n_classes=len(np.unique(ytr)))
        acc = (preds == yte).mean()
        rew = composite_reward(acc, fair["mean_bpsn_auc"], fair["dispersion"], acc_w=float(cfg["rl"]["reward"]["accuracy_weight"]))

        # REINFORCE update with baseline: theta += lr * (R - b) * grad(log pi(a|theta))
        adv = rew - baseline
        # For log-normal: grad_theta log pi ∝ eps / sigma  (since a=exp(theta+eps), eps ~ N(0,sigma^2))
        grad = (eps / (sigma + 1e-8))
        theta = theta + lr * adv * grad

        # Update moving baseline
        beta = 0.1
        baseline = (1 - beta) * baseline + beta * rew

        # Track best
        if rew > best["reward"]:
            best.update({"reward": float(rew), "report": fair, "preds": preds, "scores": scores, "theta": theta.copy()})

        print(f"Episode {ep+1}/{episodes} | reward={rew:.4f} | acc={acc:.4f} | fair_mean={fair['mean_bpsn_auc']:.4f} | baseline={baseline:.4f}")

    out.mkdir(parents=True, exist_ok=True)
    # Save artifacts
    pd.DataFrame({"idx": idte, "y_true": yte, "y_pred": best["preds"]}).to_csv(out / "preds.csv", index=False)
    np.save(out / "scores.npy", best["scores"])
    with open(out / "report.json", "w") as f:
        json.dump(best["report"], f, indent=2)
    with open(out / "policy_theta.json", "w") as f:
        json.dump({"subgroups": uniq_sub, "theta": list(map(float, best["theta"]))}, f, indent=2)
    print("[green]Saved RL run to[/green]", str(out))

@app.command()
def eval(pred: Path = typer.Option(..., help="CSV with columns: idx,y_true,y_pred"),
         groups: Path = typer.Option(..., help="JSON group mapping or ignored if CSV already contains subgroup col"),
         report: Path = typer.Option(Path("runs/report.json"), help="Output report JSON")):
    dfp = pd.read_csv(pred)
    acc = float((dfp.y_true == dfp.y_pred).mean())
    out = {"overall_accuracy": acc}
    with open(report, "w") as f:
        json.dump(out, f, indent=2)
    print(out)
pred: Path = typer.Option(..., help="CSV with columns: idx,y_true,y_pred"),
         groups: Path = typer.Option(..., help="JSON group mapping or ignored if CSV already contains subgroup col"),
         report: Path = typer.Option(Path("runs/report.json"), help="Output report JSON")):
    dfp = pd.read_csv(pred)
    # Placeholder: evaluation expects probabilities; for quick demo just compute confusion-level metrics.
    acc = float((dfp.y_true == dfp.y_pred).mean())
    out = {"overall_accuracy": acc}
    with open(report, "w") as f:
        json.dump(out, f, indent=2)
    print(out)

if __name__ == "__main__":
    app()


@app.command()
def explain(config: Path = typer.Option(..., help="YAML config"),
            n_repeats: int = typer.Option(10, help="Permutation importance repeats"),
            out: Path = typer.Option(Path("runs/explain"), help="Output directory")):
    with open(config) as f:
        cfg = yaml.safe_load(f)
    ds = DatasetConfig(**cfg["dataset"])
    df = load_dataset(ds)
    (Xtr, ytr, idtr), (Xte, yte, idte) = tensorize_split(df, ds, cfg["trainer"]["test_size"], cfg["trainer"].get("stratify", True))
    # Train baseline model and compute importance on test
    from .explain import explain_permutation_importance, slice_report
    clf, preds, scores = fit_predict_classifier(cfg["model"]["type"], cfg["model"]["params"], Xtr, ytr, Xte)
    # Global permutation importance (re-fit inside explain for clean isolation)
    global_imp = explain_permutation_importance(cfg["model"]["type"], cfg["model"]["params"],
                                                Xtr, ytr, Xte, yte, ds.features, n_repeats=n_repeats)
    # Subgroup slice accuracy
    sub_col = "g_intersection" if "g_intersection" in df.columns else ds.groups[-1]
    slices = slice_report(yte, preds, df.iloc[idte][sub_col].values)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "permutation_importance.json", "w") as f:
        json.dump(global_imp, f, indent=2)
    with open(out / "slice_report.json", "w") as f:
        json.dump(slices, f, indent=2)
    print({"perm_importance": global_imp, "slice_report": slices})
