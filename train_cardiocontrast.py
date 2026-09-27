"""
train_cardiocontrast.py — One training script for every experiment.

Examples
--------
  python train_cardiocontrast.py --preset exp1_baseline        --seed 42
  python train_cardiocontrast.py --preset exp4_cardiocontrast  --seed 42
  python train_cardiocontrast.py --preset exp4_cardiocontrast  --seed 43
  python train_cardiocontrast.py --preset exp4_cardiocontrast  --contrastive_weight 0.3 --exp_name lambda0.3

Outputs go to <output_root>/<exp_name>_seed<seed>/:
  args.json, log.csv, best.pth (selected on VAL mean Dice), last.pth

What changed vs. train_camus.py / train_camus_contrastive.py
------------------------------------------------------------
  * Model selection on a VALIDATION split carved from the training patients.
    The test patients are never seen here.
  * All experiments (including the baseline) use the same grouped sampler,
    augmentation, loss and schedule -> clean ablation.
  * Loss = cross-entropy + soft Dice (unweighted). The old inverse-frequency
    weights (0.59 / 3.41) were computed from the union of all three structures
    (~15% foreground), but each training target is ONE structure (~5%), so the
    weights did not match the task.
  * Selection metric = mean per-structure Dice on val (overall IoU is dominated
    by large structures and hides the myocardium).
  * bf16 autocast on GPUs that support it; no per-step gc.collect() /
    empty_cache() (they only slowed training).
  * Single-GPU. DataParallel split a 3-sample batch 2+1 across GPUs, giving
    BatchNorm a batch of 1 on one card.
  * Every trainable parameter is checked to be in exactly one optimizer group.
  * CLI flags + presets instead of hand-editing config.py between runs.
"""

import argparse
import csv
import datetime
import json
import os
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
from data.dataset_camus import CAMUSDataset
from data.prompts import STRUCTURES
from data.samplers import GroupedStructureBatchSampler
from lib.contrastive import ContrastiveAnatomicalLoss, weighted_pool
from lib.segmentation import build_model, decoder_hidden_size


# ----------------------------------------------------------------------- args
def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=sorted(config.PRESETS), default=None)
    p.add_argument("--exp_name", default=None)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--data_dir", default=config.CAMUS_DATA_DIR)
    p.add_argument("--output_root", default=config.OUTPUT_ROOT)
    p.add_argument("--pretrained_swin", default=config.PRETRAINED_SWIN)
    p.add_argument("--bert_path", default=config.BERT_PATH)
    p.add_argument("--swin_type", default=config.SWIN_TYPE)
    p.add_argument("--window_size", type=int, default=config.WINDOW_SIZE)
    p.add_argument("--img_size", type=int, default=config.IMG_SIZE)

    p.add_argument("--decode_with_lang", type=int, default=None)
    p.add_argument("--contrastive_weight", type=float, default=None)
    p.add_argument("--tau", type=float, default=config.CONTRASTIVE_TAU)
    p.add_argument("--pool_region", default=None, choices=["pred", "gt", "union"])
    p.add_argument("--proj_head", default=None, choices=["none", "mlp"],
                   help="none = loss on decoder features (v2, default); mlp = v1 ablation")
    p.add_argument("--no_detach_pool", action="store_true")
    p.add_argument("--text_encoder", default=None, choices=["bert", "embedding"])
    p.add_argument("--bert_trainable_layers", type=int, default=config.BERT_TRAINABLE_LAYERS)
    p.add_argument("--embed_tokens", type=int, default=8)
    p.add_argument("--prompt_mode", default=None, choices=["fixed", "paraphrase"])

    p.add_argument("--loss", default="ce_dice", choices=["ce", "ce_dice"])
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--images_per_batch", type=int, default=config.IMAGES_PER_BATCH)
    p.add_argument("--accum", type=int, default=config.GRAD_ACCUM_STEPS)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--warmup_steps", type=int, default=config.WARMUP_STEPS)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--print_freq", type=int, default=100)
    p.add_argument("--resume", action="store_true", help="resume from last.pth if present")
    p.add_argument("--max_images", type=int, default=None, help="smoke test only")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    # Resolve: explicit flag > preset > default
    defaults = dict(decode_with_lang=1, contrastive_weight=0.0, pool_region="union",
                    proj_head="none", text_encoder="bert", prompt_mode="fixed")
    preset = config.PRESETS[args.preset] if args.preset else {}
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, preset.get(k, v))
    args.decode_with_lang = bool(args.decode_with_lang)
    if args.exp_name is None:
        args.exp_name = args.preset or "custom"
    return args


# ---------------------------------------------------------------------- utils
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def seg_loss(logits, target, kind="ce_dice"):
    logits = logits.float()
    ce = F.cross_entropy(logits, target)
    if kind == "ce":
        return ce
    prob = torch.softmax(logits, dim=1)[:, 1]
    tgt = target.float()
    inter = (prob * tgt).sum(dim=(1, 2))
    denom = prob.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    return ce + dice.mean()


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def build_optimizer(model, contrastive_module, lr, wd, gate_lr_mult=10.0):
    no_decay, decay, gates = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".gate"):
            # Zero-initialised tanh gates get 10x the base LR so they can open.
            # (100x let a gate jump from 0.01 to 0.49 within one epoch and
            # destabilised training once the contrastive loss acted on the decoder.)
            gates.append(p)
        elif (p.ndim <= 1 or "norm" in name or "relative_position_bias_table" in name
                or "absolute_pos_embed" in name or "text_encoder.table" in name):
            no_decay.append(p)
        else:
            decay.append(p)
    for name, p in contrastive_module.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)

    groups = [{"params": decay, "weight_decay": wd, "lr": lr},
              {"params": no_decay, "weight_decay": 0.0, "lr": lr},
              {"params": gates, "weight_decay": 0.0, "lr": lr * gate_lr_mult}]
    # Coverage check: every trainable parameter exactly once.
    seen = [id(p) for g in groups for p in g["params"]]
    assert len(seen) == len(set(seen)), "parameter in two optimizer groups"
    trainable = {id(p) for p in list(model.parameters()) + list(contrastive_module.parameters())
                 if p.requires_grad}
    assert trainable == set(seen), "trainable parameter missing from optimizer"
    n = sum(p.numel() for g in groups for p in g["params"])
    print("[optim] {:.1f}M trainable parameters".format(n / 1e6), flush=True)
    return torch.optim.AdamW([g for g in groups if g["params"]], lr=lr)


# ------------------------------------------------------------------- training
def train_one_epoch(model, cont, optimizer, scheduler, loader, epoch, args, amp_dtype):
    model.train()
    cont.train()
    stats = {"seg": 0.0, "cont": 0.0, "neg_cos": [], "n": 0}
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader)

    for i, batch in enumerate(loader):
        b = to_device(batch, args.device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, feats = model(b["image"], b["input_ids"], b["attn_mask"], b["class_id"],
                                  return_features=True)
        loss_seg = seg_loss(logits, b["target"], args.loss)
        loss_cont, neg_cos, _ = cont(feats, logits, b["image_idx"], b["structure"],
                                     target=b["target"], union=b["union"])
        loss = loss_seg + args.contrastive_weight * loss_cont
        (loss / args.accum).backward()

        if (i + 1) % args.accum == 0 or (i + 1) == n_batches:
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        stats["seg"] += loss_seg.item()
        stats["cont"] += float(loss_cont.item())
        if neg_cos == neg_cos:
            stats["neg_cos"].append(neg_cos)
        stats["n"] += 1
        if i % args.print_freq == 0:
            print("Epoch [{}] step [{}/{}] seg {:.4f} cont {:.4f} negcos {:.3f} lr {:.2e}".format(
                epoch, i, n_batches, loss_seg.item(), float(loss_cont.item()), neg_cos,
                optimizer.param_groups[0]["lr"]), flush=True)

    n = max(1, stats["n"])
    return {"train_seg": stats["seg"] / n, "train_cont": stats["cont"] / n,
            "train_neg_cos": float(np.mean(stats["neg_cos"])) if stats["neg_cos"] else float("nan")}


@torch.no_grad()
def validate(model, loader, args, amp_dtype):
    """Fast validation at network resolution. Final numbers come from evaluate.py."""
    model.eval()
    dices = {s: [] for s in STRUCTURES}
    cum_i = cum_u = 0
    raw_cos = []
    for batch in loader:
        b = to_device(batch, args.device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, feats = model(b["image"], b["input_ids"], b["attn_mask"], b["class_id"],
                                  return_features=True)
        # Head-free separation diagnostic (comparable across ALL experiments, incl.
        # the baseline): cosine between decoder features of different prompts of
        # the same image, pooled over the SAME pixels (union of structures).
        pooled = F.normalize(weighted_pool(feats.float(), b["union"].unsqueeze(1)), dim=1)
        same = b["image_idx"][:, None] == b["image_idx"][None, :]
        diff = b["structure"][:, None] != b["structure"][None, :]
        pair = same & diff
        if pair.any():
            raw_cos.append((pooled @ pooled.t())[pair].mean().item())
        pred = logits.float().argmax(1)
        tgt = b["target"]
        inter = (pred * tgt).sum(dim=(1, 2)).float()
        ps, gs = pred.sum(dim=(1, 2)).float(), tgt.sum(dim=(1, 2)).float()
        d = torch.where(ps + gs > 0, 2 * inter / (ps + gs).clamp(min=1), torch.ones_like(inter))
        for s_id, dv in zip(b["structure"].tolist(), d.tolist()):
            dices[s_id].append(dv)
        cum_i += inter.sum().item()
        cum_u += (ps + gs - inter).sum().item()
    out = {"val_dice_" + STRUCTURES[s]: float(np.mean(v)) for s, v in dices.items()}
    out["val_mean_dice"] = float(np.mean([out["val_dice_" + STRUCTURES[s]] for s in STRUCTURES]))
    out["val_overall_iou"] = 100.0 * cum_i / max(1, cum_u)
    out["val_prompt_feature_cos"] = float(np.mean(raw_cos)) if raw_cos else float("nan")
    return out


def main(argv=None):
    args = get_args(argv)
    set_seed(args.seed)
    run_dir = os.path.join(args.output_root, "{}_seed{}".format(args.exp_name, args.seed))
    os.makedirs(run_dir, exist_ok=True)
    args.git_hash = git_hash()
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 70)
    print("Experiment      : {}  (seed {})".format(args.exp_name, args.seed))
    print("Decoder CA      : {}".format("ON" if args.decode_with_lang else "OFF"))
    print("Contrastive     : {}".format(
        "ON (w={}, tau={}, pool={}, head={}, detach={})".format(
            args.contrastive_weight, args.tau, args.pool_region, args.proj_head,
            not args.no_detach_pool)
        if args.contrastive_weight > 0 else "OFF"))
    print("Text encoder    : {}   prompts: {}".format(args.text_encoder, args.prompt_mode))
    print("Output dir      : {}".format(run_dir))
    print("=" * 70, flush=True)

    use_cuda = args.device.startswith("cuda")
    amp_dtype = None
    if use_cuda and not args.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
        if amp_dtype is None:
            print("[amp] bf16 not supported on this GPU -> fp32", flush=True)
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    train_ds = CAMUSDataset(args.data_dir, "train", args.img_size, args.bert_path,
                            augment=not args.no_augment, prompt_mode=args.prompt_mode,
                            seed=args.seed, max_images=args.max_images)
    val_ds = CAMUSDataset(args.data_dir, "val", args.img_size, args.bert_path,
                          augment=False, seed=args.seed, max_images=args.max_images)
    print("[data] train images {} (samples {}), val images {} (samples {})".format(
        len(train_ds.images), len(train_ds), len(val_ds.images), len(val_ds)), flush=True)

    sampler = GroupedStructureBatchSampler(train_ds, args.images_per_batch, shuffle=True,
                                           seed=args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=use_cuda)
    val_loader = torch.utils.data.DataLoader(      # batch of 6 = 2 whole images
        val_ds, batch_size=6, shuffle=False, num_workers=args.num_workers, pin_memory=use_cuda)

    model = build_model(args.swin_type, args.pretrained_swin, args.window_size,
                        args.text_encoder, args.bert_path, args.bert_trainable_layers,
                        args.decode_with_lang, args.embed_tokens).to(args.device)
    hid = decoder_hidden_size(args.swin_type)
    cont = ContrastiveAnatomicalLoss(hid, hid, 128, args.tau, args.pool_region,
                                     detach_weights=not args.no_detach_pool,
                                     proj_head=args.proj_head).to(args.device)

    optimizer = build_optimizer(model, cont, args.lr, args.weight_decay)
    steps_per_epoch = (len(train_loader) + args.accum - 1) // args.accum
    total_steps = steps_per_epoch * args.epochs
    warm = min(args.warmup_steps, max(1, total_steps // 10))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return max(0.0, 1 - (step - warm) / max(1, total_steps - warm)) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch, best = 0, -1.0
    last_path = os.path.join(run_dir, "last.pth")
    if args.resume and os.path.isfile(last_path):
        ck = torch.load(last_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        cont.load_state_dict(ck["contrastive_module"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch, best = ck["epoch"] + 1, ck["best_val"]
        print("[resume] from epoch {} (best val {:.4f})".format(start_epoch, best), flush=True)

    log_path = os.path.join(run_dir, "log.csv")
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_ds.set_epoch(epoch)
        sampler.set_epoch(epoch)
        tr = train_one_epoch(model, cont, optimizer, scheduler, train_loader, epoch, args,
                             amp_dtype)
        va = validate(model, val_loader, args, amp_dtype)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **tr, **va}
        if args.decode_with_lang:
            row.update({"gate_" + k: v for k, v in model.classifier.gate_values().items()})
        print("Epoch [{}] ".format(epoch) + "  ".join(
            "{} {:.4g}".format(k, v) for k, v in row.items() if isinstance(v, float)), flush=True)

        new_file = not os.path.isfile(log_path)
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                w.writeheader()
            w.writerow(row)

        state = {"model": model.state_dict(), "contrastive_module": cont.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "epoch": epoch, "best_val": best, "args": vars(args)}
        if va["val_mean_dice"] > best:
            best = va["val_mean_dice"]
            state["best_val"] = best
            torch.save({k: state[k] for k in ("model", "contrastive_module", "epoch", "args")}
                       | {"val": va}, os.path.join(run_dir, "best.pth"))
            print("  -> new best val mean Dice {:.4f} (saved best.pth)".format(best), flush=True)
        torch.save(state, last_path)

    print("Training complete in {}. Best val mean Dice {:.4f}".format(
        datetime.timedelta(seconds=int(time.time() - t0)), best), flush=True)
    print("Next: python evaluate.py --run_dir {}".format(run_dir), flush=True)


if __name__ == "__main__":
    main()
