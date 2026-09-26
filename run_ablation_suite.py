import argparse
import os
import subprocess
import sys
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
    for name in experiment_order:
        if name not in config.PRESETS and name != "exp1_baseline":
            raise KeyError(f"Unknown preset: {name}")
        command = build_command_for_preset(name)
        env = build_env_for_preset(name)
        summary = {
            "experiment": name,
            "command": " ".join(command),
            "decode_with_lang": env.get("CC_DECODE_WITH_LANG", "0"),
            "contrastive_weight": env.get("CC_CONTRASTIVE_WEIGHT", "0.0"),
            "env": env,
        }
        if dry_run:
            print(f"[dry-run] {name}: {' '.join(command)}")
            print(f"  decode_with_lang={summary['decode_with_lang']} contrastive_weight={summary['contrastive_weight']}")
        else:
            print(f"[run] starting {name} ...")
            completed = subprocess.run(command, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))
            if completed.returncode != 0:
                print(f"[error] {name} failed with exit code {completed.returncode}")
                results.append({"experiment": name, "status": "failed", "returncode": completed.returncode})
                break
            results.append({"experiment": name, "status": "ok", "returncode": 0})
        print()
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
