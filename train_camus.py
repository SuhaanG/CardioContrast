# train_camus.py — Multi-GPU training of LAVT on the CAMUS dataset.
# Runs on 2x RTX A5000 using gradient accumulation and DataParallel.
#
# Usage:
#   python train_camus.py 2>&1 | tee logs/baseline_run.txt
import os
import time
import datetime
import gc
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from lib import segmentation
import config


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_model_args():
    return SimpleNamespace(
        model="lavt_one",
        swin_type=config.SWIN_TYPE,
        decode_with_lang=False,
        mha="",
        fusion_drop=0.0,
        window12=config.WINDOW_SIZE == 12 or "window12" in config.PRETRAINED_SWIN.lower(),
        img_size=config.IMG_SIZE,
        bert_tokenizer=config.BERT_PATH,
        ck_bert=config.BERT_PATH,
        bert_trainable_layers=config.BERT_TRAINABLE_LAYERS,
    )


def criterion(output, target):
    # Inverse-frequency class weights for background/foreground.
    # Foreground (structure) occupies ~15% of image area on average across
    # all three CAMUS structures; background ~85%.
    # Weights = 1/freq, normalized to sum to 2: w_bg=0.59, w_fg=3.41.
    weight = torch.FloatTensor([0.59, 3.41]).to(output.device)
    return nn.functional.cross_entropy(output, target, weight=weight)


def IoU(pred, gt):
    pred = pred.argmax(1)
    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection
    if intersection == 0 or union == 0:
        return 0, intersection, union
    return float(intersection) / float(union), intersection, union


def train_one_epoch(model, optimizer, data_loader, lr_scheduler, epoch, print_freq=150):
    model.train()
    running_loss = 0.0
    n_batches    = 0
    optimizer.zero_grad()

    for i, data in enumerate(data_loader):
        image = data["image"]
        target = data["target"].long()
        sentences = data["input_ids"]
        attentions = data["attn_mask"]
        image      = image.cuda(non_blocking=True)
        target     = target.cuda(non_blocking=True)
        sentences  = sentences.cuda(non_blocking=True)
        attentions = attentions.cuda(non_blocking=True)

        sentences  = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        output = model(image, sentences, l_mask=attentions)
        loss   = criterion(output, target)

        total_loss  = loss
        scaled_loss = total_loss / config.GRADIENT_ACCUMULATION_STEPS
        scaled_loss.backward()

        if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(data_loader):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        running_loss += total_loss.item()
        n_batches    += 1

        if i % print_freq == 0:
            print("Epoch [{}] step [{}/{}] loss {:.4f} lr {:.6f}".format(
                epoch, i, len(data_loader), total_loss.item(),
                optimizer.param_groups[0]['lr']), flush=True)

        del image, target, sentences, attentions, loss, total_loss, scaled_loss, output, data
        gc.collect()
        torch.cuda.empty_cache()

    print("Epoch [{}] average loss: {:.4f}".format(
        epoch, running_loss / max(1, n_batches)), flush=True)


def evaluate(model, data_loader):
    model.eval()
    acc_ious  = 0
    total_its = 0
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total   = 0
    mean_IoU    = []

    with torch.no_grad():
        for data in data_loader:
            total_its += 1
            image = data["image"]
            target = data["target"].long()
            sentences = data["input_ids"]
            attentions = data["attn_mask"]
            image      = image.cuda(non_blocking=True)
            target     = target.cuda(non_blocking=True)
            sentences  = sentences.cuda(non_blocking=True)
            attentions = attentions.cuda(non_blocking=True)

            sentences  = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            output = model(image, sentences, l_mask=attentions)

            iou, I, U = IoU(output, target)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n in range(len(eval_seg_iou_list)):
                seg_correct[n] += (iou >= eval_seg_iou_list[n])
            seg_total += 1

    mIoU        = np.mean(np.array(mean_IoU)) if mean_IoU else 0.0
    overall_IoU = (100. * cum_I / cum_U) if cum_U > 0 else 0.0
    print("Mean IoU: {:.2f}".format(mIoU * 100), flush=True)
    print("Overall IoU: {:.2f}".format(overall_IoU), flush=True)
    return mIoU * 100, overall_IoU


def main():
    config.initialize_environment()
    runtime = config.describe_runtime()
    print("[config] Runtime summary: {}".format(runtime), flush=True)
    set_seed(config.SEED)

    assert torch.cuda.is_available(), "CUDA GPU required. Run on the lab machine."

    model_args = build_model_args()
    checkpoint_runtime = config.describe_runtime()
    checkpoint_runtime.update({
        "decode_with_lang": False,
        "contrastive_weight": 0.0,
        "window12": model_args.window12,
        "window_size": 12 if model_args.window12 else config.WINDOW_SIZE,
    })
    from data.dataset_camus import CAMUSDataset
    train_ds = CAMUSDataset(
        data_dir=config.CAMUS_DATA_DIR,
        split="train",
        img_size=config.IMG_SIZE,
        seed=config.SEED,
        use_language=False,
    )
    val_ds = CAMUSDataset(
        data_dir=config.CAMUS_DATA_DIR,
        split="val",
        img_size=config.IMG_SIZE,
        seed=config.SEED,
        use_language=False,
    )
    print("CAMUS examples — train: {}  val: {}".format(len(train_ds), len(val_ds)), flush=True)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "CAMUS train or validation split is empty. Check CAMUS_DATA_DIR and "
            "the database_split subgroup files."
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4)

    print("Building model: {}".format(model_args.model), flush=True)
    model = segmentation.__dict__[model_args.model](
        pretrained=config.PRETRAINED_SWIN, args=model_args)

    if len(config.GPU_IDS) > 1:
        print("Activating DataParallel on GPUs: {}".format(config.GPU_IDS))
        model = nn.DataParallel(model, device_ids=config.GPU_IDS)

    model.cuda()
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    backbone_no_decay = []
    backbone_decay    = []
    for name, m in raw_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    params_to_optimize = [
        {'params': backbone_no_decay, 'weight_decay': 0.0},
        {'params': backbone_decay},
        {"params": [p for p in raw_model.classifier.parameters() if p.requires_grad]},
    ]
    if raw_model.text_encoder is not None:
        text_parameters = [p for p in raw_model.text_encoder.parameters() if p.requires_grad]
        if text_parameters:
            params_to_optimize.append({"params": text_parameters})

    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    total_steps  = (len(train_loader) // config.GRADIENT_ACCUMULATION_STEPS) * config.EPOCHS
    warmup_steps = 500
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, (1 - (current_step - warmup_steps) / max(1, total_steps - warmup_steps)) ** 0.9)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_oIoU  = -1.0
    start_time = time.time()

    for epoch in range(config.EPOCHS):
        train_one_epoch(model, optimizer, train_loader, lr_scheduler,
                        epoch, print_freq=150)
        mIoU, overall_IoU = evaluate(model, val_loader)

        if overall_IoU > best_oIoU:
            best_oIoU = overall_IoU
            save_path = os.path.join(config.CHECKPOINT_DIR, "model_best_camus.pth")
            torch.save({
                "model": raw_model.state_dict(),
                "epoch": epoch,
                "experiment": "exp1_baseline",
                "decode_with_lang": False,
                "contrastive_weight": 0.0,
                "selection_metric": "validation_overall_iou_percent",
                "best_validation_overall_iou_percent": best_oIoU,
                "runtime": checkpoint_runtime,
            }, save_path)
            print("Saved best model (Overall IoU {:.2f}) -> {}".format(
                overall_IoU, save_path), flush=True)

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print("Training complete. Total time {}".format(total_time), flush=True)


if __name__ == "__main__":
    main()
