import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import config


DEFAULT_EXPERIMENT_ORDER = [
    "exp1_baseline",
    "exp2_decoder_ca",
    "exp3_contrastive",
    "exp4_cardiocontrast",
]


def build_env_for_preset(preset_name: str) -> Dict[str, str]:
    env = os.environ.copy()
    if preset_name == "exp1_baseline":
        env["CC_DECODE_WITH_LANG"] = "0"
        env["CC_CONTRASTIVE_WEIGHT"] = "0.0"
        env["CC_EPOCHS"] = str(config.EPOCHS)
        return env

    preset = config.PRESETS[preset_name]
    env["CC_DECODE_WITH_LANG"] = "1" if bool(preset.get("decode_with_lang", False)) else "0"
    env["CC_CONTRASTIVE_WEIGHT"] = str(float(preset.get("contrastive_weight", 0.0)))
    env["CC_EPOCHS"] = str(config.EPOCHS)
    return env


def build_command_for_preset(preset_name: str) -> List[str]:
    if preset_name == "exp1_baseline":
        return [sys.executable, "train_camus.py"]
    return [sys.executable, "train_camus_contrastive.py"]


def run_experiments(experiment_order=None, dry_run=False):
    if experiment_order is None:
        experiment_order = DEFAULT_EXPERIMENT_ORDER
    results = []
    manifest_root = Path(config.CC_OUTPUT_ROOT) / "paper"
    manifest_root.mkdir(parents=True, exist_ok=True)
    for name in experiment_order:
        if name not in config.PRESETS and name != "exp1_baseline":
            raise KeyError(f"Unknown preset: {name}")
        command = build_command_for_preset(name)
        env = build_env_for_preset(name)
        summary = {
            "experiment": name,
            "command": " ".join(command),
            "decode_with_lang": bool(int(env.get("CC_DECODE_WITH_LANG", "0"))),
            "contrastive_weight": float(env.get("CC_CONTRASTIVE_WEIGHT", "0.0")),
            "val_split_used_for_selection": True,
            "test_split_reserved_for_final_report": True,
            "split_policy": {
                "train": "patient-level",
                "val": "for model selection only",
                "test": "held back for final reporting",
            },
            "env": env,
        }
        if dry_run:
            print(f"[dry-run] {name}: {' '.join(command)}")
            print(f"  decode_with_lang={summary['decode_with_lang']} contrastive_weight={summary['contrastive_weight']}")
            print("  split_policy: val for selection, test reserved for final report")
        else:
            print(f"[run] starting {name} ...")
            completed = subprocess.run(command, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))
            if completed.returncode != 0:
                print(f"[error] {name} failed with exit code {completed.returncode}")
                summary["status"] = "failed"
                summary["returncode"] = completed.returncode
                results.append(summary)
                manifest_path = manifest_root / f"{name}_manifest.json"
                manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
                break
            summary["status"] = "ok"
            summary["returncode"] = 0
            summary["checkpoint_dir"] = str(Path(config.CC_OUTPUT_ROOT) / "checkpoints")
            summary["best_checkpoint"] = str(Path(config.CC_OUTPUT_ROOT) / "checkpoints" / ("model_best_exp4_cardiocontrast.pth" if name == "exp4_cardiocontrast" else "model_best_exp2_decoder_ca.pth" if name == "exp2_decoder_ca" else "model_best_exp3_contrastive_only.pth" if name == "exp3_contrastive" else "model_best_camus.pth"))
            results.append(summary)
            manifest_path = manifest_root / f"{name}_manifest.json"
            manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print()
    if results:
        summary_path = manifest_root / "ablation_manifest.json"
        summary_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CardioContrast ablation suite in a reproducible order.")
    parser.add_argument("--run", action="store_true", help="Execute the configured experiments in order.")
    parser.add_argument("--preset", action="append", default=[], help="Run a specific preset name. Can be passed more than once.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the commands without executing them.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.preset:
        experiment_order = args.preset
    else:
        experiment_order = DEFAULT_EXPERIMENT_ORDER

    should_run = args.run or not args.dry_run
    run_experiments(experiment_order=experiment_order, dry_run=not should_run)


if __name__ == "__main__":
    main()
