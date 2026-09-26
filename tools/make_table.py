"""
tools/make_table.py — Results table (mean ± std over seeds) in Markdown and LaTeX.

  python tools/make_table.py \
      --run "LAVT baseline=experiments/exp1_baseline_seed*" \
      --run "+ decoder CA=experiments/exp2_decoder_ca_seed*" \
      --run "+ contrastive=experiments/exp3_contrastive_seed*" \
      --run "CardioContrast=experiments/exp4_cardiocontrast_seed*" \
      --run "CardioContrast (class emb.)=experiments/exp5_class_embedding_seed*" \
      --run "nnU-Net 2D=experiments/nnunet_2d_fold0" \
      --eval eval_test_canonical

Seed-to-seed std is what reviewers need to judge whether a +0.3 Dice gain is real;
use compare_runs.py for per-patient significance.
"""

import argparse
import glob
import json
import os

import numpy as np

STRUCTS = [("lv_endo", "LV"), ("myocardium", "MYO"), ("left_atrium", "LA")]


def fmt(vals, scale=1.0, digits=3):
    v = np.array([x for x in vals if x is not None], float) * scale
    if len(v) == 0:
        return "–"
    if len(v) == 1:
        return "{:.{d}f}".format(v[0], d=digits)
    return "{:.{d}f} ± {:.{d}f}".format(v.mean(), v.std(ddof=1), d=digits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="Label=glob")
    ap.add_argument("--eval", default="eval_test_canonical")
    args = ap.parse_args()

    header = ["Method", "#seeds"]
    for _, s in STRUCTS:
        header += ["{} Dice".format(s), "{} HD95".format(s)]
    header += ["mean Dice", "EF MAE", "EF r"]
    rows = []
    for spec in args.run:
        label, pattern = spec.split("=", 1)
        sums = []
        for d in sorted(glob.glob(pattern)):
            p = os.path.join(d, args.eval, "summary.json")
            if os.path.isfile(p):
                sums.append(json.load(open(p)))
        if not sums:
            print("WARNING: nothing found for", spec)
            continue
        r = [label, str(len(sums))]
        for key, _ in STRUCTS:
            r.append(fmt([s["segmentation"][key]["all"]["dice"]["mean"] for s in sums]))
            r.append(fmt([s["segmentation"][key]["all"]["hd95"]["mean"] for s in sums], digits=2))
        r.append(fmt([s["mean_dice_over_structures"] for s in sums]))
        ef = [s.get("clinical", {}).get("EF_pred_vs_gtmask", {}) for s in sums]
        r.append(fmt([e.get("mae") for e in ef], digits=2))
        r.append(fmt([e.get("pearson_r") for e in ef]))
        rows.append(r)

    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print("\n% LaTeX")
    print(" & ".join(header) + r" \\ \midrule")
    for r in rows:
        print(" & ".join(x.replace("±", r"$\pm$") for x in r) + r" \\")


if __name__ == "__main__":
    main()
