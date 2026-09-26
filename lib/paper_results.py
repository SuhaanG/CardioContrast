import numpy as np

from .metrics import bootstrap_confidence_interval


METRICS = ("dice", "iou", "hd95", "hd", "mad", "leakage")


def aggregate_patient_metrics(frame_rows, n_bootstrap=2000, seed=42):
    grouped = {}
    for row in frame_rows:
        key = (str(row["patient"]), int(row["structure_id"]))
        grouped.setdefault(key, []).append(row)

    patient_rows = []
    for (patient, structure_id), rows in sorted(grouped.items()):
        patient_row = {
            "patient": patient,
            "structure_id": structure_id,
            "structure": rows[0].get("structure", str(structure_id)),
            "n_frames": len(rows),
        }
        for metric in METRICS:
            values = np.asarray([row.get(metric, np.nan) for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            patient_row[metric] = float(np.mean(finite)) if finite.size else float("nan")
        patient_rows.append(patient_row)

    structures = sorted({row["structure_id"] for row in patient_rows})
    structure_summaries = {}
    for structure_id in structures:
        rows = [row for row in patient_rows if row["structure_id"] == structure_id]
        metric_summaries = {}
        for metric_index, metric in enumerate(METRICS):
            values = [row[metric] for row in rows if np.isfinite(row[metric])]
            metric_summaries[metric] = bootstrap_confidence_interval(
                values,
                n_bootstrap=n_bootstrap,
                seed=seed + structure_id * 100 + metric_index,
            )
        structure_summaries[str(structure_id)] = {
            "structure_id": structure_id,
            "structure": rows[0].get("structure", str(structure_id)),
            "n_patients": len(rows),
            "metrics": metric_summaries,
        }

    return {
        "patient_metrics": patient_rows,
        "structure_summary": structure_summaries,
    }