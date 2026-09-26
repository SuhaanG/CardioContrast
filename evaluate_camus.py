import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from data.dataset_camus import CAMUSDataset
from data.prompts import STRUCTURES
from lib import segmentation
from lib.metrics import leakage, per_structure_metrics
from lib.paper_results import aggregate_patient_metrics


EXPERIMENTS = {
    "exp1_baseline": {"decode_with_lang": False, "contrastive_weight": 0.0},
    "exp2_decoder_ca": {"decode_with_lang": True, "contrastive_weight": 0.0},
    "exp3_contrastive": {"decode_with_lang": False, "contrastive_weight": 0.1},
    "exp4_cardiocontrast": {"decode_with_lang": True, "contrastive_weight": 0.1},
}


def build_model_args(decode_with_lang, runtime):
    return SimpleNamespace(
        model="lavt_one",
        swin_type=runtime.get("swin_type", config.SWIN_TYPE),
        decode_with_lang=decode_with_lang,
        mha="",
        fusion_drop=0.0,
        window12=runtime.get("window12", True),
        img_size=runtime.get("img_size", config.IMG_SIZE),
        bert_tokenizer=runtime.get("bert_path", config.BERT_PATH),
        ck_bert=runtime.get("bert_path", config.BERT_PATH),
        bert_trainable_layers=runtime.get("bert_trainable_layers", config.BERT_TRAINABLE_LAYERS),
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write an empty result table: {path}")
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_test_set(experiment, checkpoint_path, output_dir, device=None, n_bootstrap=2000):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}; choose from {sorted(EXPERIMENTS)}")

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint_hash = sha256_file(checkpoint_path)
    condition = EXPERIMENTS[experiment]
    decode_with_lang = condition["decode_with_lang"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_experiment = checkpoint.get("experiment")
    if saved_experiment is not None and saved_experiment != experiment:
        raise ValueError(f"Checkpoint is for {saved_experiment}, not {experiment}.")
    training_runtime = checkpoint.get("runtime", {})
    if not isinstance(training_runtime, dict):
        raise ValueError("Checkpoint runtime metadata must be a dictionary.")

    dataset = CAMUSDataset(
        data_dir=config.CAMUS_DATA_DIR,
        split="test",
        img_size=training_runtime.get("img_size", config.IMG_SIZE),
        seed=training_runtime.get("seed", config.SEED),
        use_language=decode_with_lang,
        bert_tokenizer=training_runtime.get("bert_path", config.BERT_PATH),
        spacing_unit=training_runtime.get("spacing_unit") or config.SPACING_UNIT,
    )
    if not len(dataset):
        raise ValueError("CAMUS test split is empty; check CAMUS_DATA_DIR and database_split files.")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        pin_memory=str(device).startswith("cuda"))

    model = segmentation.lavt_one(
        pretrained="",
        args=build_model_args(decode_with_lang, training_runtime),
    )
    saved_condition = checkpoint.get("decode_with_lang")
    if saved_condition is not None and bool(saved_condition) != decode_with_lang:
        raise ValueError("Checkpoint language-conditioning metadata does not match --experiment.")
        saved_weight = checkpoint.get("contrastive_weight")
        if saved_weight is not None and not np.isclose(float(saved_weight), condition["contrastive_weight"]):
            raise ValueError("Checkpoint contrastive-weight metadata does not match --experiment.")
    state_dict = checkpoint.get("model")
    if state_dict is None:
        raise ValueError("Checkpoint must contain a 'model' state dictionary.")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    frame_rows = []
    cached_mask_path = None
    cached_mask = None
    cached_spacing = None
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            sample = dataset.samples[index]
            record = sample["record"]
            mask_path = record["mask_path"]
            if mask_path != cached_mask_path:
                cached_mask = dataset._load_nifti_2d(mask_path)
                cached_spacing = dataset.get_pixel_spacing(
                    mask_path,
                    assumed_unit=training_runtime.get("spacing_unit") or config.SPACING_UNIT,
                )
                cached_mask_path = mask_path

            image = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attn_mask"].to(device, non_blocking=True)
            logits = model(
                image,
                input_ids,
                l_mask=attention_mask,
                decode_with_lang=decode_with_lang,
            )
            logits = F.interpolate(
                logits,
                size=tuple(cached_mask.shape),
                mode="bilinear",
                align_corners=True,
            )
            prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            structure_id = int(sample["structure"])
            target = (cached_mask == structure_id).astype(np.uint8)
            metrics = per_structure_metrics(
                prediction, target, structure_id=structure_id, spacing=cached_spacing
            )
            frame_rows.append({
                "patient": record["patient"],
                "view": record["view"],
                "phase": record["phase"],
                "structure_id": structure_id,
                "structure": STRUCTURES[structure_id],
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "hd95": metrics["hd95"],
                "hd": metrics["hd"],
                "mad": metrics["mad"],
                "leakage": leakage(prediction, cached_mask, structure_id),
                "empty_case": metrics["empty_case"],
            })

    aggregated = aggregate_patient_metrics(frame_rows, n_bootstrap=n_bootstrap, seed=config.SEED)
    if output_dir is None:
        output_dir = Path(config.CC_OUTPUT_ROOT) / "paper" / "test_eval" / experiment / checkpoint_hash[:12]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "frame_metrics.csv", frame_rows)
    write_csv(output_dir / "patient_metrics.csv", aggregated["patient_metrics"])

    runtime = config.describe_runtime()
    runtime.update(training_runtime)
    runtime["decode_with_lang"] = decode_with_lang
    runtime["contrastive_weight"] = condition["contrastive_weight"]
    summary = {
        "experiment": experiment,
        "split": "test",
        "test_used_for_model_selection": False,
        "confidence_intervals_clustered_by": "patient",
        "distance_unit": "mm",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "selection_metric": checkpoint.get("selection_metric"),
        "best_validation_overall_iou_percent": checkpoint.get("best_validation_overall_iou_percent"),
        "evaluation_git_hash": config.git_hash(),
        "runtime": runtime,
        "n_frames": len(frame_rows),
        "n_patients": len({row["patient"] for row in frame_rows}),
        "n_bootstrap": int(n_bootstrap),
        "structure_summary": aggregated["structure_summary"],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as output:
        json.dump(json_safe(summary), output, indent=2, sort_keys=True, allow_nan=False)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one selected CardioContrast checkpoint on the held-out CAMUS test split."
    )
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--checkpoint", required=True, help="Selected checkpoint; selection must be based on validation only.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")

    result = evaluate_test_set(
        args.experiment,
        args.checkpoint,
        args.output_dir,
        device=args.device,
        n_bootstrap=args.bootstrap_samples,
    )
    print(json.dumps(json_safe(result["structure_summary"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()