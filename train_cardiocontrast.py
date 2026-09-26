import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from data.dataset_camus import CAMUSDataset
from data.samplers import GroupedStructureBatchSampler
from lib import segmentation
from lib.contrastive import ContrastiveAnatomicalLoss, weighted_pool


DEFAULT_PRESET = "exp4_cardiocontrast"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a reproducible CardioContrast CAMUS experiment.")
    parser.add_argument("--preset", choices=sorted(config.PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--swin_type", choices=("tiny", "small", "base", "large"), default=None)
    parser.add_argument("--pretrained_swin", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--bert_path", default=None)
    parser.add_argument("--bert_trainable_layers", type=int, default=None)
    parser.add_argument("--embed_tokens", type=int, default=None)
    parser.add_argument("--decode_with_lang", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--contrastive_weight", type=float, default=None)
    parser.add_argument("--contrastive_tau", type=float, default=None)
    parser.add_argument("--pool_region", choices=("pred", "gt", "union"), default=None)
    parser.add_argument("--text_encoder", choices=("bert", "embedding"), default=None)
    parser.add_argument("--prompt_mode", choices=("fixed", "paraphrase"), default=None)
    parser.add_argument("--images_per_batch", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def resolve_args(parsed):
    preset = config.PRESETS[parsed.preset]
    resolved = {
        "exp_name": preset.get("exp_name", parsed.preset),
        "seed": config.SEED,
        "epochs": config.EPOCHS,
        "img_size": config.IMG_SIZE,
        "window_size": config.WINDOW_SIZE,
        "swin_type": config.SWIN_TYPE,
        "pretrained_swin": config.PRETRAINED_SWIN,
        "data_dir": config.CAMUS_DATA_DIR,
        "output_root": config.CC_OUTPUT_ROOT,
        "bert_path": config.BERT_PATH,
        "bert_trainable_layers": config.BERT_TRAINABLE_LAYERS,
        "embed_tokens": config.EMBED_TOKENS,
        "decode_with_lang": bool(preset.get("decode_with_lang", False)),
        "contrastive_weight": float(preset.get("contrastive_weight", 0.0)),
        "contrastive_tau": config.CONTRASTIVE_TAU,
        "pool_region": preset.get("contrastive_pool_region", "pred"),
        "text_encoder": preset.get("text_encoder", "bert"),
        "prompt_mode": preset.get("prompt_mode", "fixed"),
        "images_per_batch": config.IMAGES_PER_BATCH,
        "grad_accum_steps": config.GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": config.LR,
        "weight_decay": config.WEIGHT_DECAY,
        "max_images": parsed.max_images,
        "workers": parsed.workers,
        "device": parsed.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "resume": parsed.resume,
    }
    for key in tuple(resolved):
        value = getattr(parsed, key, None)
        if value is not None:
            resolved[key] = value
    if resolved["prompt_mode"] == "paraphrase" and resolved["decode_with_lang"] is False:
        raise ValueError("Paraphrase prompts require language conditioning")
    if resolved["images_per_batch"] < 1 or resolved["grad_accum_steps"] < 1:
        raise ValueError("images_per_batch and grad_accum_steps must be positive")
    if resolved["max_images"] is not None and resolved["max_images"] < 1:
        raise ValueError("max_images must be positive")
    return resolved


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def segmentation_loss(logits, target):
    target = target.long()
    cross_entropy = F.cross_entropy(logits, target)
    foreground_probability = torch.softmax(logits, dim=1)[:, 1]
    target_float = (target == 1).to(foreground_probability.dtype)
    intersection = (foreground_probability * target_float).sum(dim=(-2, -1))
    denominator = foreground_probability.sum(dim=(-2, -1)) + target_float.sum(dim=(-2, -1))
    soft_dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return cross_entropy + (1.0 - soft_dice).mean()


def build_optimizer(model, contrastive_module, args):
    groups = {"decay": [], "no_decay": [], "gates": []}
    seen = set()
    named_parameters = list(model.named_parameters()) + [
        (f"contrastive.{name}", parameter)
        for name, parameter in contrastive_module.named_parameters()
    ]
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        identifier = id(parameter)
        if identifier in seen:
            raise AssertionError(f"Trainable parameter appears more than once: {name}")
        seen.add(identifier)
        lowered = name.lower()
        if "gate" in lowered:
            group_name = "gates"
        elif (
            name.endswith(".bias")
            or "norm" in lowered
            or "relative_position_bias_table" in lowered
            or lowered.endswith(".table")
        ):
            group_name = "no_decay"
        else:
            group_name = "decay"
        groups[group_name].append(parameter)

    expected = {id(parameter) for _, parameter in named_parameters if parameter.requires_grad}
    if seen != expected:
        raise AssertionError("Optimizer parameter partition omitted or duplicated trainable parameters")

    parameter_groups = []
    for name, parameters in groups.items():
        if not parameters:
            continue
        parameter_groups.append({
            "params": parameters,
            "lr": args["learning_rate"] * (100.0 if name == "gates" else 1.0),
            "weight_decay": args["weight_decay"] if name == "decay" else 0.0,
            "group_name": name,
        })
    if not parameter_groups:
        raise ValueError("Model has no trainable parameters")
    return torch.optim.AdamW(parameter_groups)


def save_checkpoint(path, model, contrastive_module, optimizer, scheduler, epoch,
                    best_score, args):
    torch.save({
        "model": model.state_dict(),
        "contrastive_module": contrastive_module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_mean_dice": best_score,
        "args": args,
    }, path)


def evaluate(model, data_loader, device, decode_with_lang):
    model.eval()
    dice_by_structure = {1: [], 2: [], 3: []}
    intersection_sum = 0
    union_sum = 0
    prompt_features = {}
    with torch.inference_mode():
        for batch in data_loader:
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            class_ids = batch["class_id"].to(device)
            target = batch["target"].to(device).long()
            logits, features = model(
                image, input_ids, attn_mask, class_ids, return_features=True
            )
            prediction = logits.argmax(dim=1)
            for index in range(target.shape[0]):
                pred = prediction[index] == 1
                truth = target[index] == 1
                intersection = torch.logical_and(pred, truth).sum().item()
                union = torch.logical_or(pred, truth).sum().item()
                dice = 1.0 if union == 0 else 2.0 * intersection / (pred.sum().item() + truth.sum().item())
                structure_id = int(batch["structure"][index])
                dice_by_structure[structure_id].append(dice)
                intersection_sum += intersection
                union_sum += union

                pooled = weighted_pool(
                    features[index:index + 1],
                    batch["union"][index:index + 1].to(device),
                    detach=True,
                )[0]
                image_id = int(batch["image_idx"][index])
                prompt_features.setdefault(image_id, []).append(pooled)

    structure_means = {
        structure_id: float(np.mean(values)) if values else float("nan")
        for structure_id, values in dice_by_structure.items()
    }
    pair_cosines = []
    for vectors in prompt_features.values():
        for left in range(len(vectors)):
            for right in range(left + 1, len(vectors)):
                pair_cosines.append(F.cosine_similarity(vectors[left], vectors[right], dim=0).item())
    return {
        "val_mean_dice": float(np.mean(list(structure_means.values()))),
        "val_dice_by_structure": structure_means,
        "val_overall_iou": float(intersection_sum / union_sum) if union_sum else 1.0,
        "val_prompt_feature_cos": float(np.mean(pair_cosines)) if pair_cosines else float("nan"),
    }


def train(args):
    set_seed(args["seed"])
    device = torch.device(args["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    run_dir = Path(args["output_root"]) / f"{args['exp_name']}_seed{args['seed']}"
    if run_dir.exists() and args["resume"] is None and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory already contains files: {run_dir}; use --resume or a new seed")
    run_dir.mkdir(parents=True, exist_ok=True)
    args["git_hash"] = config.git_hash()
    args["bf16_autocast"] = use_bf16
    with (run_dir / "args.json").open("w", encoding="utf-8") as output:
        json.dump(args, output, indent=2, sort_keys=True)

    use_language = args["text_encoder"] == "bert"
    train_dataset = CAMUSDataset(
        args["data_dir"], split="train", img_size=args["img_size"],
        prompt_mode=args["prompt_mode"], seed=args["seed"],
        use_language=use_language, bert_tokenizer=args["bert_path"],
        spacing_unit=config.SPACING_UNIT, max_images=args["max_images"],
    )
    val_dataset = CAMUSDataset(
        args["data_dir"], split="val", img_size=args["img_size"], seed=args["seed"],
        use_language=use_language, bert_tokenizer=args["bert_path"],
        spacing_unit=config.SPACING_UNIT, augment=False, max_images=args["max_images"],
    )
    if not train_dataset.max_images or not val_dataset.max_images:
        raise ValueError("Train and validation splits must both contain at least one image")

    batch_sampler = GroupedStructureBatchSampler(
        train_dataset, args["images_per_batch"], shuffle=True, seed=args["seed"]
    )
    if len(batch_sampler) == 0:
        raise ValueError("Training set is too small for images_per_batch")
    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler, num_workers=args["workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args["workers"],
        pin_memory=device.type == "cuda",
    )

    model = segmentation.build_model(
        swin_type=args["swin_type"],
        pretrained_swin=args["pretrained_swin"],
        window_size=args["window_size"],
        text_encoder=args["text_encoder"],
        bert_path=args["bert_path"],
        bert_trainable_layers=args["bert_trainable_layers"],
        decode_with_lang=args["decode_with_lang"],
        embed_tokens=args["embed_tokens"],
    ).to(device)
    contrastive_module = ContrastiveAnatomicalLoss(
        segmentation.decoder_hidden_size(args["swin_type"]),
        proj_hidden_dim=segmentation.decoder_hidden_size(args["swin_type"]),
        proj_out_dim=128,
        tau=args["contrastive_tau"],
    ).to(device)
    optimizer = build_optimizer(model, contrastive_module, args)
    total_steps = math.ceil(len(train_loader) / args["grad_accum_steps"]) * args["epochs"]
    warmup_steps = min(500, total_steps // 10)

    def schedule(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress) ** 0.9

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start_epoch = 0
    best_score = -float("inf")
    if args["resume"]:
        checkpoint = torch.load(args["resume"], map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        contrastive_module.load_state_dict(checkpoint["contrastive_module"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_val_mean_dice"])

    log_path = run_dir / "log.csv"
    log_fields = [
        "epoch", "train_ce_dice_loss", "train_contrastive_loss", "train_negative_cosine",
        "train_contrastive_anchors", "val_mean_dice", "val_overall_iou",
        "val_prompt_feature_cos", "val_dice_lv_endo", "val_dice_myocardium",
        "val_dice_left_atrium", "gate_values",
    ]
    for epoch in range(start_epoch, args["epochs"]):
        model.train()
        contrastive_module.train()
        train_dataset.set_epoch(epoch)
        batch_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        seg_sum = cont_sum = neg_cos_sum = 0.0
        anchor_sum = 0
        optimizer_steps = 0

        for step, batch in enumerate(train_loader):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True).long()
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn_mask = batch["attn_mask"].to(device, non_blocking=True)
            class_ids = batch["class_id"].to(device, non_blocking=True)
            image_ids = batch["image_idx"].to(device, non_blocking=True)
            structure_ids = batch["structure"].to(device, non_blocking=True)
            union_masks = batch["union"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits, features = model(
                    image, input_ids, attn_mask, class_ids, return_features=True
                )
                loss_seg = segmentation_loss(logits.float(), target)
                loss_cont, neg_cos, n_anchors = contrastive_module(
                    features.float(), logits.float(), image_ids, structure_ids,
                    pool_region=args["pool_region"], gt_masks=target,
                    union_masks=union_masks,
                )
                total_loss = loss_seg + args["contrastive_weight"] * loss_cont

            accumulation_count = min(
                args["grad_accum_steps"],
                len(train_loader) - (step // args["grad_accum_steps"]) * args["grad_accum_steps"],
            )
            (total_loss / accumulation_count).backward()
            boundary = (step + 1) % args["grad_accum_steps"] == 0 or step + 1 == len(train_loader)
            if boundary:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for group in optimizer.param_groups for parameter in group["params"]],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            seg_sum += float(loss_seg.detach())
            cont_sum += float(loss_cont.detach())
            neg_cos_sum += float(neg_cos.detach())
            anchor_sum += int(n_anchors)

        validation = evaluate(model, val_loader, device, args["decode_with_lang"])
        gate_values = model.classifier.gate_values()
        mean_seg = seg_sum / max(1, len(train_loader))
        mean_cont = cont_sum / max(1, len(train_loader))
        mean_neg_cos = neg_cos_sum / max(1, len(train_loader))
        row = {
            "epoch": epoch,
            "train_ce_dice_loss": mean_seg,
            "train_contrastive_loss": mean_cont,
            "train_negative_cosine": mean_neg_cos,
            "train_contrastive_anchors": anchor_sum,
            "val_mean_dice": validation["val_mean_dice"],
            "val_overall_iou": validation["val_overall_iou"],
            "val_prompt_feature_cos": validation["val_prompt_feature_cos"],
            "val_dice_lv_endo": validation["val_dice_by_structure"][1],
            "val_dice_myocardium": validation["val_dice_by_structure"][2],
            "val_dice_left_atrium": validation["val_dice_by_structure"][3],
            "gate_values": json.dumps(gate_values),
        }
        with log_path.open("a", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=log_fields)
            if output.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

        save_checkpoint(run_dir / "last.pth", model, contrastive_module, optimizer,
                        scheduler, epoch, best_score, args)
        if validation["val_mean_dice"] > best_score:
            best_score = validation["val_mean_dice"]
            save_checkpoint(run_dir / "best.pth", model, contrastive_module, optimizer,
                            scheduler, epoch, best_score, args)
        print(
            f"epoch={epoch + 1}/{args['epochs']} "
            f"train={mean_seg:.4f} cont={mean_cont:.4f} "
            f"val_dice={validation['val_mean_dice']:.4f} "
            f"val_iou={validation['val_overall_iou']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.3g} steps={optimizer_steps}",
            flush=True,
        )
    return run_dir


def main():
    args = resolve_args(parse_args())
    run_dir = train(args)
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
