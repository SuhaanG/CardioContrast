import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from data.dataset_camus import CAMUSDataset
from lib import segmentation
from lib.report import Reporter


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def evaluate_run(run_dir, checkpoint_name="best.pth", split="test",
                 prompt_set="canonical", save_preds=False, device=None):
    if split not in {"test", "val"}:
        raise ValueError("split must be 'test' or 'val'")
    if prompt_set not in {"canonical", "heldout"}:
        raise ValueError("prompt_set must be 'canonical' or 'heldout'")
    run_dir = Path(run_dir).resolve()
    args_path = run_dir / "args.json"
    checkpoint_path = run_dir / checkpoint_name
    if not args_path.is_file():
        raise FileNotFoundError(f"Training args not found: {args_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    with args_path.open("r", encoding="utf-8") as source:
        run_args = json.load(source)
    if run_args.get("text_encoder") == "embedding" and prompt_set == "heldout":
        raise ValueError("Held-out text prompts are not defined for the class-embedding encoder")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = segmentation.build_model(
        swin_type=run_args["swin_type"],
        pretrained_swin="",
        window_size=run_args["window_size"],
        text_encoder=run_args["text_encoder"],
        bert_path=run_args["bert_path"],
        bert_trainable_layers=run_args["bert_trainable_layers"],
        decode_with_lang=run_args["decode_with_lang"],
        embed_tokens=run_args["embed_tokens"],
        warn_random_init=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    dataset = CAMUSDataset(
        data_dir=run_args["data_dir"],
        split=split,
        img_size=run_args["img_size"],
        prompt_mode="fixed",
        eval_prompt_set=prompt_set,
        seed=run_args["seed"],
        use_language=run_args["text_encoder"] == "bert",
        bert_tokenizer=run_args["bert_path"],
        spacing_unit=config.SPACING_UNIT,
        augment=False,
    )
    if not len(dataset):
        raise ValueError(f"CAMUS {split} split is empty")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    reporter = Reporter()
    output_dir = run_dir / f"eval_{split}_{prompt_set}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_image_idx = None
    cached_full_mask = None
    cached_spacing = None

    with torch.inference_mode():
        for index, batch in enumerate(loader):
            sample = dataset.samples[index]
            record = sample["record"]
            image_idx = int(sample["image_idx"])
            if image_idx != cached_image_idx:
                cached_full_mask, cached_spacing = dataset.load_full_res(image_idx)
                cached_image_idx = image_idx

            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            class_ids = batch["class_id"].to(device)
            logits = model(image, input_ids, attn_mask, class_ids)
            logits = F.interpolate(
                logits.float(),
                size=tuple(cached_full_mask.shape),
                mode="bilinear",
                align_corners=False,
            )
            probability = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
            prediction = probability >= 0.5
            structure_id = int(sample["structure"])
            prompt = dataset._resolve_prompt(
                structure_id, image_idx, sample["prompt_k"]
            )
            reporter.add(
                patient=record["patient"],
                view=record["view"],
                phase=record["phase"],
                quality=record["quality"],
                ref_ef=float("nan") if record["ref_ef"] is None else record["ref_ef"],
                structure=structure_id,
                prompt=prompt,
                pred=prediction,
                gt_full=cached_full_mask,
                spacing=cached_spacing,
            )
            if save_preds:
                name = "_".join((
                    _safe_name(record["patient"]), _safe_name(record["view"]),
                    _safe_name(record["phase"]), str(structure_id),
                    str(sample["prompt_k"]),
                ))
                np.savez_compressed(
                    output_dir / f"{name}.npz",
                    prediction=prediction.astype(np.uint8),
                    probability=probability.astype(np.float32),
                    spacing=np.asarray(cached_spacing, dtype=np.float32),
                )

    summary = reporter.write(output_dir, meta={
        "run_dir": str(run_dir),
        "split": split,
        "prompt_set": prompt_set,
        "test_used_for_model_selection": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_val_mean_dice": checkpoint.get("best_val_mean_dice"),
        "training_git_hash": run_args.get("git_hash"),
        "evaluation_git_hash": config.git_hash(),
        "save_predictions": bool(save_preds),
    })
    return output_dir, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate a CardioContrast run using the shared Reporter.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--ckpt", default="best.pth")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--prompt_set", choices=("canonical", "heldout"), default="canonical")
    parser.add_argument("--save_preds", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    output_dir, summary = evaluate_run(
        args.run_dir, args.ckpt, args.split, args.prompt_set,
        args.save_preds, args.device,
    )
    print(f"Evaluation written to {output_dir}")
    print(json.dumps(summary["clinical"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
