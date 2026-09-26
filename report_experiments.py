import json
import os
from pathlib import Path


def summarize_logs(root: str):
    root_path = Path(root)
    summary = []
    for log_file in sorted(root_path.glob("**/*.log")):
        lines = log_file.read_text(errors="ignore").splitlines()
        stats = {
            "file": str(log_file),
            "mean_iou": None,
            "overall_iou": None,
            "best_epoch": None,
        }
        for line in lines:
            if "Mean IoU:" in line:
                try:
                    stats["mean_iou"] = float(line.split("Mean IoU:")[-1].strip())
                except Exception:
                    pass
            if "Overall IoU:" in line:
                try:
                    stats["overall_iou"] = float(line.split("Overall IoU:")[-1].strip())
                except Exception:
                    pass
            if "Saved best model" in line and "Overall IoU" in line:
                try:
                    stats["best_epoch"] = line
                except Exception:
                    pass
        summary.append(stats)
    return summary


def main():
    root = os.environ.get("CC_LOG_ROOT", "logs")
    payload = summarize_logs(root)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
