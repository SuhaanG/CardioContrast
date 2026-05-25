"""
train_camus.py — Single-GPU training of LAVT on the CAMUS dataset.

Clean single-GPU adaptation of LAVT's train.py. Reuses LAVT's model, loss,
and training-loop logic; removes the multi-GPU DistributedDataParallel
machinery so it runs on one A6000. LAVT's original train.py is left untouched.

All settings come from config.py. Run on the A6000 machine:
    python train_camus.py
"""

import os
import time
import datetime
import gc
import operator
from functools import reduce
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from lib import segmentation
import transforms as T
import config


def build_model_args():
    return SimpleNamespace(
        model="lavt_one",
        swin_type=config.SWIN_TYPE,
        mha="",
        fusion_drop=0.0,
        window12=True,
        img_size=config.IMG_SIZE,
        bert_tokenizer="bert-base-uncased",
        ck_bert="bert-base-uncased",
    )


def get_transform(img_size):
    return T.Compose([
        T.Resize(img_size, img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def criterion(output, target):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return nn.functional.cross_entropy(output, target, weight=weight)


def IoU(pred, gt):
    pred = pred.argmax(1)
    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection
    if intersection == 0 or union == 0:
        return 0, intersection, union
    return float(intersection) / float(union), intersection, union


def train_one_epoch(model, optimizer, data_loader, lr_scheduler, epoch, print_freq):
    model.train()
    running_loss = 0.0
    n_batches = 0
    for i, data in enumerate(data_loader):
        image, target, sentences, attentions = data
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        sentences = sentences.cuda(non_blocking=True)
        attentions = attentions.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        output = model(image, sentences, l_mask=attentions)

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        running_loss += loss.item()
        n_batches += 1

        if i % print_freq == 0:
            print("Epoch [{}] step [{}/{}] loss {:.4f} lr {:.6f}".format(
                epoch, i, len(data_loader), loss.item(),
                optimizer.param_groups[0]['lr']), flush=True)

        del image, target, sentences, attentions, loss, output, data
        gc.collect()
        torch.cuda.empty_cache()

    print("Epoch [{}] average loss: {:.4f}".format(
        epoch, running_loss / max(1, n_batches)), flush=True)


def evaluate(model, data_loader):
    model.eval()
    acc_ious = 0
    total_its = 0
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for data in data_loader:
            total_its += 1
            image, target, sentences, attentions = data
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            sentences = sentences.cuda(non_blocking=True)
            attentions = attentions.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
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

    mIoU = np.mean(np.array(mean_IoU)) if mean_IoU else 0.0
    overall_IoU = (100. * cum_I / cum_U) if cum_U > 0 else 0.0
    print("Mean IoU: {:.2f}".format(mIoU * 100), flush=True)
    print("Overall IoU: {:.2f}".format(overall_IoU), flush=True)
    return mIoU * 100, overall_IoU


def main():
    assert torch.cuda.is_available(), "CUDA GPU required (run on the A6000 machine)."

    model_args = build_model_args()
    transform = get_transform(config.IMG_SIZE)

    from data.dataset_camus import CAMUSDataset
    full_dataset = CAMUSDataset(
        data_dir=config.CAMUS_DATA_DIR,
        bert_tokenizer=model_args.bert_tokenizer,
        image_transforms=transform,
    )
    print("Total CAMUS examples: {}".format(len(full_dataset)), flush=True)

    n_total = len(full_dataset)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))
    print("Train: {}  Val: {}".format(n_train, n_val), flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4)

    print("Building model: {}".format(model_args.model), flush=True)
    model = segmentation.__dict__[model_args.model](
        pretrained=config.PRETRAINED_SWIN, args=model_args)
    model.cuda()

    backbone_no_decay = []
    backbone_decay = []
    for name, m in model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    params_to_optimize = [
        {'params': backbone_no_decay, 'weight_decay': 0.0},
        {'params': backbone_decay},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
        {"params": reduce(operator.concat,
                          [[p for p in model.text_encoder.encoder.layer[i].parameters()
                            if p.requires_grad] for i in range(10)])},
    ]

    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(train_loader) * config.EPOCHS)) ** 0.9)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_oIoU = -1.0
    start_time = time.time()

    for epoch in range(config.EPOCHS):
        train_one_epoch(model, optimizer, train_loader, lr_scheduler,
                        epoch, print_freq=10)
        mIoU, overall_IoU = evaluate(model, val_loader)
        if overall_IoU > best_oIoU:
            best_oIoU = overall_IoU
            save_path = os.path.join(config.CHECKPOINT_DIR, "model_best_camus.pth")
            torch.save({'model': model.state_dict(), 'epoch': epoch}, save_path)
            print("Saved new best model (Overall IoU {:.2f}) -> {}".format(
                overall_IoU, save_path), flush=True)

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print("Training complete. Total time {}".format(total_time), flush=True)


if __name__ == "__main__":
    main()
