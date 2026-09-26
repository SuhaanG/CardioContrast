"""
compare_runs.py — Paired statistics between two evaluated models.

  python compare_runs.py \
      --a experiments/exp1_baseline_seed42/eval_test_canonical \
      --b experiments/exp4_cardiocontrast_seed42/eval_test_canonical

Unit of analysis = PATIENT (images of one patient are not independent).
For each structure and metric: mean difference (B - A), 95% bootstrap CI,
paired Wilcoxon signed-rank p-value, Holm-corrected across all tests.

To combine seeds, pass several dirs per side (comma-separated); per-patient
values are averaged over seeds first.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["dice", "hd95", "mad", "leakage"]


def load(dirs):
    frames = [pd.read_csv(os.path.join(d, "per_patient.csv")) for d in dirs.split(",")]
    df = pd.concat(frames)
    return df.groupby(["patient", "structure_name"])[METRICS].mean().reset_index()


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference model eval dir(s)")
    ap.add_argument("--b", required=True, help="proposed model eval dir(s)")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--min_patients", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    m = A.merge(B, on=["patient", "structure_name"], suffixes=("_a", "_b"))
    rng = np.random.default_rng(0)
    rows = []
    for s, g in m.groupby("structure_name"):
        for met in METRICS:
            a, b = g[met + "_a"].to_numpy(), g[met + "_b"].to_numpy()
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            d = b - a
            if len(d) < args.min_patients:
                continue
            boot = rng.choice(d, size=(args.n_boot, len(d)), replace=True).mean(1)
            try:
                p = wilcoxon(b, a).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                p = 1.0
            rows.append({"structure": s, "metric": met, "n_patients": len(d),
                         "mean_a": a.mean(), "mean_b": b.mean(), "diff": d.mean(),
                         "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                         "p_wilcoxon": p})
    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("No structure/metric had >= {} paired patients.".format(args.min_patients))
    res["p_holm"] = holm(res["p_wilcoxon"])
    pd.set_option("display.width", 160)
    print(res.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    print("\nLower is better for hd95, mad, leakage. Significant = p_holm < 0.05.")
    if args.out:
        res.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
