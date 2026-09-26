import argparse
import json
from pathlib import Path

import numpy as np


def parse_metric_line(line):
    try:
        if "Mean IoU:" in line:
            return float(line.split("Mean IoU:")[-1].strip())
        if "Overall IoU:" in line:
            return float(line.split("Overall IoU:")[-1].strip())
    except Exception:
        pass
    return None


def summarize_log_file(log_path: str):
    text = Path(log_path).read_text(errors="ignore")
    lines = text.splitlines()
    result = {
        "log": log_path,
        "mean_iou": None,
        "overall_iou": None,
        "best_model": None,
        "contrastive_fired_pct": None,
    }
    for line in lines:
        if "Mean IoU:" in line:
            result["mean_iou"] = parse_metric_line(line)
        if "Overall IoU:" in line:
            result["overall_iou"] = parse_metric_line(line)
        if "Saved best model" in line and "Overall IoU" in line:
            result["best_model"] = line.strip()
        if "contrastive fired" in line and "%" in line:
            try:
                result["contrastive_fired_pct"] = float(line.split("contrastive fired")[-1].split("%", 1)[0].strip())
            except Exception:
                pass
    return result


def summarize_logs(root: str):
    root_path = Path(root)
    payload = []
    for log_file in sorted(root_path.glob("**/*.log")):
        payload.append(summarize_log_file(str(log_file)))
    for log_file in sorted(root_path.glob("**/*.txt")):
        if log_file.name.endswith(".txt") and not any(p.name == log_file.name for p in root_path.glob("**/*.log")):
            payload.append(summarize_log_file(str(log_file)))
    return payload


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment logs into a paper-friendly summary.")
    parser.add_argument("--root", default="logs", help="Root directory containing log files")
    args = parser.parse_args()
    payload = summarize_logs(args.root)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
