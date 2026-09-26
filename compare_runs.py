import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


METRICS = ("dice", "hd95", "mad", "leakage")


def _expand_inputs(values):
    expanded = []
    for value in values:
        matches = glob.glob(value, recursive=True)
        expanded.extend(Path(match) for match in matches)
    return sorted(set(path.resolve() for path in expanded))


def _find_patient_tables(paths, split="test"):
    expected_eval_dir = f"eval_{split}_canonical"
    tables = []
    for path in paths:
        if path.is_file() and path.name == "per_patient.csv":
            if path.parent.name == expected_eval_dir:
                tables.append(path)
        elif path.is_dir():
            for table in path.rglob("per_patient.csv"):
                if table.parent.name == expected_eval_dir:
                    tables.append(table)
    return sorted(set(tables))


def _load_patient_metrics(paths, split="test"):
    values = defaultdict(list)
    for table in _find_patient_tables(paths, split):
        with table.open("r", newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                key = (str(row["patient"]), str(row["structure_id"]))
                for metric in METRICS:
                    try:
                        value = float(row[metric])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if np.isfinite(value):
                        values[(key, metric)].append(value)
    return {key: float(np.mean(entries)) for key, entries in values.items() if entries}


def _holm_adjust(p_values):
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def compare_runs(a_paths, b_paths, min_patients=10, n_bootstrap=10000, seed=42, split="test"):
    if min_patients < 1 or n_bootstrap < 1:
        raise ValueError("min_patients and n_bootstrap must be positive")
    if split not in {"test", "val"}:
        raise ValueError("split must be test or val")
    a_values = _load_patient_metrics(_expand_inputs(a_paths), split)
    b_values = _load_patient_metrics(_expand_inputs(b_paths), split)
    comparison_keys = sorted({key for key, _ in a_values} | {key for key, _ in b_values})
    raw_results = []
    for structure_id in sorted({key[1] for key in comparison_keys}):
        for metric in METRICS:
            paired = [
                patient for patient, current_structure in comparison_keys
                if current_structure == structure_id
                and ((patient, current_structure), metric) in a_values
                and ((patient, current_structure), metric) in b_values
            ]
            if len(paired) < min_patients:
                continue
            left = np.asarray([a_values[((patient, structure_id), metric)] for patient in paired])
            right = np.asarray([b_values[((patient, structure_id), metric)] for patient in paired])
            differences = right - left
            rng = np.random.default_rng([seed, int(structure_id) if structure_id.isdigit() else 0, METRICS.index(metric)])
            bootstrap_means = np.mean(
                rng.choice(differences, size=(n_bootstrap, len(differences)), replace=True), axis=1
            )
            if np.allclose(differences, 0.0):
                p_value = 1.0
            else:
                p_value = float(wilcoxon(left, right, alternative="two-sided").pvalue)
            raw_results.append({
                "structure_id": structure_id,
                "metric": metric,
                "n_patients": len(paired),
                "mean_difference_b_minus_a": float(np.mean(differences)),
                "ci95_lower": float(np.quantile(bootstrap_means, 0.025)),
                "ci95_upper": float(np.quantile(bootstrap_means, 0.975)),
                "wilcoxon_p": p_value,
            })
    adjusted = _holm_adjust([result["wilcoxon_p"] for result in raw_results])
    for result, p_adjusted in zip(raw_results, adjusted):
        result["holm_p"] = p_adjusted
    return {
        "split": split,
        "prompt_set": "canonical",
        "a_tables": [str(path) for path in _find_patient_tables(_expand_inputs(a_paths), split)],
        "b_tables": [str(path) for path in _find_patient_tables(_expand_inputs(b_paths), split)],
        "minimum_patients": min_patients,
        "results": raw_results,
        "status": "ok" if raw_results else "no comparisons met min_patients",
    }


def main():
    parser = argparse.ArgumentParser(description="Paired patient-level comparison of evaluated run directories.")
    parser.add_argument("--a", nargs="+", required=True, help="Run directories, evaluation directories, or glob patterns")
    parser.add_argument("--b", nargs="+", required=True, help="Run directories, evaluation directories, or glob patterns")
    parser.add_argument("--min_patients", type=int, default=10)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = compare_runs(
        args.a, args.b, args.min_patients, args.bootstrap_samples, args.seed, args.split
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
