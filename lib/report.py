"""
lib/report.py — Shared scoring/reporting so CardioContrast, nnU-Net and SAM
baselines are scored by IDENTICAL code (reviewers check this).

Usage:
    rep = Reporter()
    rep.add(patient, view, phase, quality, ref_ef, structure, prompt, pred, gt_full, spacing)
    ...
    summary = rep.write(out_dir, meta={...})
"""

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from data.prompts import STRUCTURES
from lib.metrics import agreement, binary_metrics, ejection_fraction, leakage, \
    simpson_biplane_volume

METRICS = ["dice", "iou", "hd95", "hd", "mad", "leakage"]


def summarize(df):
    out = {}
    for s_name, g in df.groupby("structure_name"):
        out[s_name] = {}
        for phase, gp in list(g.groupby("phase")) + [("all", g)]:
            d = {}
            for m in METRICS:
                v = gp[m].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                d[m] = {"mean": float(v.mean()) if len(v) else None,
                        "std": float(v.std(ddof=1)) if len(v) > 1 else None,
                        "median": float(np.median(v)) if len(v) else None}
            d["n"] = int(len(gp))
            d["n_empty_pred"] = int((gp["empty_case"] == "pred").sum())
            d["n_dice_below_0.7"] = int((gp["dice"] < 0.7).sum())
            out[s_name][phase] = d
        out[s_name]["by_quality"] = {
            q: {"dice_mean": float(gq["dice"].mean()),
                "hd95_mean": float(np.nanmean(gq["hd95"])) if gq["hd95"].notna().any() else None,
                "n": int(len(gq))}
            for q, gq in g.groupby("quality")}
    return out


class Reporter:
    def __init__(self):
        self.rows = []
        self.clin_pred = defaultdict(dict)   # (pt, view, phase) -> {structure: mask}
        self.clin_gt = {}                    # (pt, view, phase) -> (gt_full, spacing)
        self.ref_ef = {}
        self._seen = set()

    def add(self, patient, view, phase, quality, ref_ef, structure, prompt, pred, gt_full,
            spacing):
        pred = pred.astype(bool)
        m = binary_metrics(pred, gt_full == structure, spacing)
        m["leakage"] = leakage(pred, gt_full, structure)
        key = (patient, view, phase)
        self.rows.append({"patient": patient, "view": view, "phase": phase, "quality": quality,
                          "structure": structure, "structure_name": STRUCTURES[structure],
                          "prompt": prompt, **m})
        if (key, structure) not in self._seen:       # first prompt per structure -> EF
            self._seen.add((key, structure))
            self.clin_pred[key][structure] = pred
            self.clin_gt[key] = (gt_full, spacing)
        self.ref_ef[patient] = ref_ef

    def _clinical(self):
        rows = []
        for pt in sorted({k[0] for k in self.clin_gt}):
            need = [(pt, v, ph) for v in ("2CH", "4CH") for ph in ("ED", "ES")]
            if not all(k in self.clin_gt and 1 in self.clin_pred[k] for k in need):
                continue
            vols = {}
            for src in ("pred", "gt"):
                for ph in ("ED", "ES"):
                    k2, k4 = (pt, "2CH", ph), (pt, "4CH", ph)
                    if src == "pred":
                        lv2, lv4 = self.clin_pred[k2][1], self.clin_pred[k4][1]
                        la2, la4 = self.clin_pred[k2].get(3), self.clin_pred[k4].get(3)
                    else:
                        lv2, lv4 = self.clin_gt[k2][0] == 1, self.clin_gt[k4][0] == 1
                        la2, la4 = self.clin_gt[k2][0] == 3, self.clin_gt[k4][0] == 3
                    vols[(src, ph)] = simpson_biplane_volume(
                        lv2, self.clin_gt[k2][1], lv4, self.clin_gt[k4][1], la2, la4)
            rows.append({
                "patient": pt,
                "edv_gt": vols[("gt", "ED")], "esv_gt": vols[("gt", "ES")],
                "edv_pred": vols[("pred", "ED")], "esv_pred": vols[("pred", "ES")],
                "ef_gt": ejection_fraction(vols[("gt", "ED")], vols[("gt", "ES")]),
                "ef_pred": ejection_fraction(vols[("pred", "ED")], vols[("pred", "ES")]),
                "ef_ref_cfg": self.ref_ef.get(pt, np.nan),
            })
        return pd.DataFrame(rows)

    def write(self, out_dir, meta=None, verbose=True):
        os.makedirs(out_dir, exist_ok=True)
        df = pd.DataFrame(self.rows)
        df.to_csv(os.path.join(out_dir, "per_sample.csv"), index=False)
        (df.groupby(["patient", "structure_name"])[METRICS].mean(numeric_only=True)
         .reset_index().to_csv(os.path.join(out_dir, "per_patient.csv"), index=False))
        clin = self._clinical()
        clin.to_csv(os.path.join(out_dir, "clinical.csv"), index=False)

        summary = dict(meta or {})
        summary.update({
            "n_samples": int(len(df)), "n_patients": int(df["patient"].nunique()),
            "segmentation": summarize(df),
            "mean_dice_over_structures": float(df.groupby("structure_name")["dice"].mean().mean()),
        })
        if len(clin):
            summary["clinical"] = {
                "EF_pred_vs_gtmask": agreement(clin["ef_gt"], clin["ef_pred"]),
                "EDV_pred_vs_gtmask": agreement(clin["edv_gt"], clin["edv_pred"]),
                "ESV_pred_vs_gtmask": agreement(clin["esv_gt"], clin["esv_pred"]),
                "SANITY_EF_gtmask_vs_cfg": agreement(clin["ef_ref_cfg"], clin["ef_gt"]),
            }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        if verbose:
            self.print_summary(summary, out_dir)
        return summary

    @staticmethod
    def print_summary(summary, out_dir):
        def f(x):
            return float("nan") if x is None else x
        for s_name, d in summary["segmentation"].items():
            al = d["all"]
            print("{:12s} Dice {:.4f}±{:.4f}  HD95 {:.2f}mm  MAD {:.2f}mm  leakage {:.4f}  "
                  "fails(<0.7) {}".format(s_name, f(al["dice"]["mean"]), f(al["dice"]["std"]),
                                         f(al["hd95"]["mean"]), f(al["mad"]["mean"]),
                                         f(al["leakage"]["mean"]), al["n_dice_below_0.7"]))
        c = summary.get("clinical")
        if c and "pearson_r" in c["EF_pred_vs_gtmask"]:
            e = c["EF_pred_vs_gtmask"]
            print("EF pred vs GT-mask: r={:.3f} bias={:.2f} MAE={:.2f} LoA [{:.1f}, {:.1f}]".format(
                e["pearson_r"], e["bias"], e["mae"], e["loa_low"], e["loa_high"]))
            sc = c["SANITY_EF_gtmask_vs_cfg"]
            if "pearson_r" in sc:
                print("Sanity: Simpson(GT masks) vs CAMUS cfg EF r={:.3f} MAE={:.2f}. If r is "
                      "well below ~0.9, fix the long-axis estimate before reporting EF.".format(
                          sc["pearson_r"], sc["mae"]))
        print("Saved to", out_dir)
