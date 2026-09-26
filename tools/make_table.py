import argparse
import csv
import glob
from pathlib import Path

import numpy as np


STRUCTURES = ("lv_endo", "myocardium", "left_atrium")


def _read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _evaluation_dirs(run_dir):
    run_dir = Path(run_dir)
    return sorted(
        path.parent
        for path in run_dir.rglob("per_patient.csv")
        if any(part == "eval_test_canonical" for part in path.parts)
    )


def _number(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _seed_summary(eval_dir):
    patient_rows = _read_rows(eval_dir / "per_patient.csv")
    clinical_rows = _read_rows(eval_dir / "clinical.csv") if (eval_dir / "clinical.csv").is_file() else []
    result = {structure: {} for structure in STRUCTURES}
    for structure in STRUCTURES:
        rows = [row for row in patient_rows if row.get("structure") == structure]
        for metric in ("dice", "hd95"):
            values = [_number(row, metric) for row in rows]
            values = [value for value in values if np.isfinite(value)]
            result[structure][metric] = float(np.mean(values)) if values else float("nan")
    structure_dice = [result[name]["dice"] for name in STRUCTURES]
    result["mean_dice"] = float(np.nanmean(structure_dice)) if np.isfinite(structure_dice).any() else float("nan")
    ef_pairs = [
        (_number(row, "pred_ef_percent"), _number(row, "gt_ef_percent"))
        for row in clinical_rows
    ]
    ef_pairs = [(pred, target) for pred, target in ef_pairs if np.isfinite(pred) and np.isfinite(target)]
    result["ef_mae"] = float(np.mean([abs(pred - target) for pred, target in ef_pairs])) if ef_pairs else float("nan")
    result["ef_r"] = float(np.corrcoef(
        [pair[0] for pair in ef_pairs], [pair[1] for pair in ef_pairs]
    )[0, 1]) if len(ef_pairs) >= 2 and np.std([pair[0] for pair in ef_pairs]) > 0 and np.std([pair[1] for pair in ef_pairs]) > 0 else float("nan")
    return result


def _format_mean_std(values, digits=3):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return "NA"
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def make_tables(specifications):
    groups = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"Expected Label=glob, got {specification!r}")
        label, pattern = specification.split("=", 1)
        run_dirs = [Path(value) for value in glob.glob(pattern, recursive=True) if Path(value).is_dir()]
        runs = []
        for run_dir in run_dirs:
            for eval_dir in _evaluation_dirs(run_dir):
                runs.append(_seed_summary(eval_dir))
        if runs:
            groups[label] = runs

    if not groups:
        raise ValueError("No matching test/canonical evaluation summaries were found")

    columns = ["Label"]
    for structure in STRUCTURES:
        columns.extend((f"{structure} Dice", f"{structure} HD95"))
    columns.extend(("Mean Dice", "EF MAE", "EF r"))
    markdown = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    latex_rows = []
    for label, runs in groups.items():
        cells = [label]
        for structure in STRUCTURES:
            cells.append(_format_mean_std([run[structure]["dice"] for run in runs]))
            cells.append(_format_mean_std([run[structure]["hd95"] for run in runs]))
        for metric in ("mean_dice", "ef_mae", "ef_r"):
            cells.append(_format_mean_std([run[metric] for run in runs]))
        markdown.append("| " + " | ".join(cells) + " |")
        latex_rows.append(" & ".join(cells) + r" \\")

    latex_columns = [column.replace("_", r"\_") for column in columns]
    latex = "\\begin{tabular}{l" + "c" * (len(columns) - 1) + "}\n"
    latex += " \\toprule\n" + " & ".join(latex_columns) + r" \\ \midrule" + "\n"
    latex += "\n".join(latex_rows) + "\n\\bottomrule\n\\end{tabular}"
    return "\n".join(markdown), latex


def main():
    parser = argparse.ArgumentParser(description="Build paper tables from evaluated run directories.")
    parser.add_argument("--run", action="append", required=True, help="Label=glob; repeat for multiple conditions")
    args = parser.parse_args()
    markdown, latex = make_tables(args.run)
    print("Markdown:\n" + markdown)
    print("\nLaTeX:\n" + latex)


if __name__ == "__main__":
    main()
