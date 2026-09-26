import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from data.prompts import STRUCTURES
from .metrics import agreement, leakage, per_structure_metrics, simpson_volume


METRICS = ("dice", "iou", "hd95", "hd", "mad", "leakage")
VIEW_NAMES = {"2CH", "4CH"}


def _finite_stats(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"n": 0, "mean": None, "std": None, "median": None}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Reporter:
    def __init__(self):
        self.samples = []
        self._clinical_masks = {}

    def add(self, patient, view, phase, quality, ref_ef, structure, prompt,
            pred, gt_full, spacing):
        if isinstance(structure, str):
            reverse_structures = {name: idx for idx, name in STRUCTURES.items()}
            if structure not in reverse_structures:
                raise ValueError(f"Unknown structure name: {structure}")
            structure_id = reverse_structures[structure]
        else:
            structure_id = int(structure)
        if structure_id not in STRUCTURES:
            raise ValueError(f"Unknown structure id: {structure_id}")

        pred_mask = np.asarray(pred).squeeze() > 0
        full_mask = np.asarray(gt_full).squeeze()
        if pred_mask.shape != full_mask.shape:
            raise ValueError("Prediction and full-resolution ground truth must have equal shapes")
        target = full_mask == structure_id
        values = per_structure_metrics(pred_mask, target, structure_id, spacing=spacing)
        row = {
            "patient": str(patient),
            "view": str(view),
            "phase": str(phase).upper(),
            "quality": str(quality),
            "ref_ef": ref_ef,
            "structure_id": structure_id,
            "structure": STRUCTURES[structure_id],
            "prompt": str(prompt),
            **{metric: values[metric] for metric in ("dice", "iou", "hd95", "hd", "mad")},
            "leakage": leakage(pred_mask, full_mask, structure_id),
            "empty_prediction": bool(not pred_mask.any()),
            "empty_case": bool(values["empty_case"]),
        }
        self.samples.append(row)

        if view in VIEW_NAMES:
            key = (str(patient), str(view), str(phase).upper())
            entry = self._clinical_masks.setdefault(key, {
                "spacing": tuple(map(float, spacing)),
                "ref_ef": ref_ef,
            })
            if structure_id == 1:
                entry["lv_pred"] = pred_mask
                entry["lv_gt"] = target
            elif structure_id == 3:
                entry["la_pred"] = pred_mask
                entry["la_gt"] = full_mask == 3

    def _clinical_rows(self):
        views = defaultdict(dict)
        for (patient, view, phase), fields in self._clinical_masks.items():
            views[(patient, phase)][view] = fields

        patients = sorted({patient for patient, _ in views})
        rows = []
        for patient in patients:
            result = {"patient": patient, "ref_ef_percent": None}
            ed_volumes = {"pred": {}, "gt": {}}
            es_volumes = {"pred": {}, "gt": {}}
            for phase, volume_map in (("ED", ed_volumes), ("ES", es_volumes)):
                phase_views = views.get((patient, phase), {})
                two = phase_views.get("2CH")
                four = phase_views.get("4CH")
                if two and four:
                    for kind in ("pred", "gt"):
                        la_two = two.get(f"la_{kind}")
                        la_four = four.get(f"la_{kind}")
                        volume_map[kind][patient] = simpson_volume(
                            two[f"lv_{kind}"], four[f"lv_{kind}"],
                            la_two, la_four,
                            two["spacing"], four["spacing"],
                        )
                candidates = [v.get("ref_ef") for v in phase_views.values() if v.get("ref_ef") is not None]
                if candidates:
                    result["ref_ef_percent"] = float(candidates[0])

            for label, values in (("edv", ed_volumes), ("esv", es_volumes)):
                result[f"pred_{label}_ml"] = values["pred"].get(patient)
                result[f"gt_{label}_ml"] = values["gt"].get(patient)
            for kind in ("pred", "gt"):
                edv = ed_volumes[kind].get(patient)
                esv = es_volumes[kind].get(patient)
                result[f"{kind}_ef_percent"] = (
                    100.0 * (edv - esv) / edv if edv is not None and edv > 0 and esv is not None else None
                )
            rows.append(result)
        return rows

    def write(self, out_dir, meta=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.samples:
            raise ValueError("No samples have been added to the report")

        patient_groups = defaultdict(list)
        for row in self.samples:
            patient_groups[(row["patient"], row["structure_id"])].append(row)
        per_patient = []
        for (patient, structure_id), rows in sorted(patient_groups.items()):
            summary = {
                "patient": patient,
                "structure_id": structure_id,
                "structure": STRUCTURES[structure_id],
                "n_samples": len(rows),
            }
            for metric in METRICS:
                summary[metric] = _finite_stats([row[metric] for row in rows])["mean"]
            per_patient.append(summary)

        clinical = self._clinical_rows()
        _write_csv(out_dir / "per_sample.csv", self.samples)
        _write_csv(out_dir / "per_patient.csv", per_patient)
        _write_csv(out_dir / "clinical.csv", clinical)

        summary = {"meta": meta or {}, "structures": {}, "quality": {}, "clinical": {}}
        for structure_id, structure_name in STRUCTURES.items():
            structure_rows = [row for row in self.samples if row["structure_id"] == structure_id]
            phases = {}
            for phase in ("ED", "ES", "all"):
                rows = structure_rows if phase == "all" else [r for r in structure_rows if r["phase"] == phase]
                phases[phase] = {
                    "n_samples": len(rows),
                    "metrics": {metric: _finite_stats([r[metric] for r in rows]) for metric in METRICS},
                    "dice_below_0_7": sum(r["dice"] < 0.7 for r in rows),
                    "empty_predictions": sum(r["empty_prediction"] for r in rows),
                }
            summary["structures"][structure_name] = phases

        for quality in sorted({row["quality"] for row in self.samples}):
            rows = [row for row in self.samples if row["quality"] == quality]
            summary["quality"][quality] = {
                "n_samples": len(rows),
                "metrics": {metric: _finite_stats([r[metric] for r in rows]) for metric in METRICS},
            }

        for metric in ("pred_edv_ml", "gt_edv_ml", "pred_esv_ml", "gt_esv_ml", "pred_ef_percent", "gt_ef_percent"):
            values = [row[metric] for row in clinical if row.get(metric) is not None]
            summary["clinical"][metric] = _finite_stats(values)
        for metric, x_key, y_key in (
            ("edv_pred_vs_gt", "pred_edv_ml", "gt_edv_ml"),
            ("esv_pred_vs_gt", "pred_esv_ml", "gt_esv_ml"),
            ("ef_pred_vs_gt", "pred_ef_percent", "gt_ef_percent"),
            ("gt_ef_vs_cfg", "gt_ef_percent", "ref_ef_percent"),
            ("pred_ef_vs_cfg", "pred_ef_percent", "ref_ef_percent"),
        ):
            summary["clinical"][metric] = agreement(
                [row.get(x_key, np.nan) for row in clinical],
                [row.get(y_key, np.nan) for row in clinical],
            )

        safe_summary = _json_safe(summary)
        with (out_dir / "summary.json").open("w", encoding="utf-8") as output:
            json.dump(safe_summary, output, indent=2, sort_keys=True, allow_nan=False)
        return safe_summary
