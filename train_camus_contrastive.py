# train_camus_contrastive.py — CardioContrast full method training.
# Run AFTER confirming train_camus.py (baseline) works correctly.
#
# Before running: set CONTRASTIVE_WEIGHT > 0.0 in config.py (e.g. 0.1)
#
# Usage:
#   python train_camus_contrastive.py 2>&1 | tee logs/contrastive_run.txt

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
from lib.contrastive import ContrastiveAnatomicalLoss
import transforms as T
import config

# Swin-Base decoder hidden_size = c4_dims // 2 = 1024 // 2 = 512
DECODER_HIDDEN_SIZE = 512


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_model_args():
    return SimpleNamespace(
        model="lavt_one", swin_type=config.SWIN_TYPE,
        mha="", fusion_drop=0.0, window12=True,
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
    weight = torch.FloatTensor([0.9, 1.1]).to(output.device)
    return nn.functional.cross_entropy(output, target, weight=weight)


def IoU(pred, gt):
    pred = pred.argmax(1)
    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection
    if intersection == 0 or union == 0:
        return 0, intersection, union
    return float(intersection) / float(union), intersection, union


def train_one_epoch(model, contrastive_module, optimizer, data_loader,
                    lr_scheduler, epoch, print_freq):
    model.train()
    contrastive_module.train()
    running_loss_seg = running_loss_cont = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for i, data in enumerate(data_loader):
        image, target, sentences, attentions, image_ids, structure_ids = data
        image         = image.cuda(non_blocking=True)
        target        = target.cuda(non_blocking=True)
        sentences     = sentences.cuda(non_blocking=True)
        attentions    = attentions.cuda(non_blocking=True)
        image_ids     = image_ids.cuda(non_blocking=True)
        structure_ids = structure_ids.cuda(non_blocking=True)
        sentences     = sentences.squeeze(1)
        attentions    = attentions.squeeze(1)

        logits, pre_logit_features = model(
            image, sentences, l_mask=attentions, return_features=True)

        loss_seg  = criterion(logits, target)
        loss_cont = contrastive_module(
            pre_logit_features, logits, image_ids, structure_ids)
        total_loss  = loss_seg + config.CONTRASTIVE_WEIGHT * loss_cont
        scaled_loss = total_loss / config.GRADIENT_ACCUMULATION_STEPS
        scaled_loss.backward()

        if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(data_loader):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        running_loss_seg  += loss_seg.item()
        running_loss_cont += loss_cont.item()
        n_batches += 1

        if i % print_freq == 0:
            print("Epoch [{}] step [{}/{}] seg {:.4f} cont {:.4f} lr {:.6f}".format(
                epoch, i, len(data_loader),
                loss_seg.item(), loss_cont.item(),
                optimizer.param_groups[0]['lr']), flush=True)

        del (image, target, sentences, attentions, image_ids, structure_ids,
             logits, pre_logit_features, loss_seg, loss_cont, total_loss,
             scaled_loss, data)
        gc.collect()
        torch.cuda.empty_cache()

    print("Epoch [{}] avg seg {:.4f} avg cont {:.4f}".format(
        epoch,
        running_loss_seg  / max(1, n_batches),
        running_loss_cont / max(1, n_batches)), flush=True)


def evaluate(model, data_loader):
    model.eval()
    acc_ious = total_its = 0
    cum_I = cum_U = 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    mean_IoU = []

    with torch.no_grad():
        for data in data_loader:
            total_its += 1
            image, target, sentences, attentions = data[0], data[1], data[2], data[3]
            image      = image.cuda(non_blocking=True)
            target     = target.cuda(non_blocking=True)
            sentences  = sentences.cuda(non_blocking=True)
            attentions = attentions.cuda(non_blocking=True)
            sentences  = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            output     = model(image, sentences, l_mask=attentions)
            iou, I, U  = IoU(output, target)
            acc_ious  += iou
            mean_IoU.append(iou)
            cum_I += I; cum_U += U
            for n in range(len(eval_seg_iou_list)):
                seg_correct[n] += (iou >= eval_seg_iou_list[n])

    mIoU        = np.mean(np.array(mean_IoU)) if mean_IoU else 0.0
    overall_IoU = (100. * cum_I / cum_U) if cum_U > 0 else 0.0
    print("Mean IoU: {:.2f}".format(mIoU * 100), flush=True)
    print("Overall IoU: {:.2f}".format(overall_IoU), flush=True)
    return mIoU * 100, overall_IoU


def main():
    config.initialize_environment()
    set_seed(config.SEED)
    assert torch.cuda.is_available(), "CUDA GPU required."
    assert config.CONTRASTIVE_WEIGHT > 0.0, (
        "Set CONTRASTIVE_WEIGHT > 0.0 in config.py before running this script.")

    model_args = build_model_args()
    transform  = get_transform(config.IMG_SIZE)

    from data.dataset_camus_contrastive import CAMUSDatasetContrastive
    full_dataset = CAMUSDatasetContrastive(
        data_dir=config.CAMUS_DATA_DIR,
        bert_tokenizer=model_args.bert_tokenizer,
        image_transforms=transform,
    )
    print("Total CAMUS examples: {}".format(len(full_dataset)), flush=True)
    if len(full_dataset) == 0:
        raise ValueError("No CAMUS samples found. Check CAMUS_DATA_DIR in config.py.")

    n_total = len(full_dataset)
    n_val   = int(0.2 * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.SEED))
    print("Train: {}  Val: {}".format(n_train, n_val), flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4)

    model = segmentation.__dict__[model_args.model](
        pretrained=config.PRETRAINED_SWIN, args=model_args)

    contrastive_module = ContrastiveAnatomicalLoss(
        in_dim=DECODER_HIDDEN_SIZE,
        proj_hidden_dim=DECODER_HIDDEN_SIZE,
        proj_out_dim=128,
        tau=0.07,
    )

    if len(config.GPU_IDS) > 1:
        model = nn.DataParallel(model, device_ids=config.GPU_IDS)

    model.cuda()
    contrastive_module.cuda()
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
        {"params": reduce(operator.concat,
                          [[p for p in raw_model.text_encoder.encoder.layer[i].parameters()
                            if p.requires_grad] for i in range(10)])},
        {"params": contrastive_module.parameters()},
    ]

    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    total_steps = (len(train_loader) // config.GRADIENT_ACCUMULATION_STEPS) * config.EPOCHS
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: (1 - x / total_steps) ** 0.9)

    best_oIoU  = -1.0
    start_time = time.time()
    print("Contrastive weight: {}".format(config.CONTRASTIVE_WEIGHT), flush=True)

    for epoch in range(config.EPOCHS):
        train_one_epoch(model, contrastive_module, optimizer,
                        train_loader, lr_scheduler, epoch, print_freq=10)
        mIoU, overall_IoU = evaluate(model, val_loader)

        if overall_IoU > best_oIoU:
            best_oIoU = overall_IoU
            save_path = os.path.join(
                config.CHECKPOINT_DIR, "model_best_camus_contrastive.pth")
            torch.save({
                'model':              raw_model.state_dict(),
                'contrastive_module': contrastive_module.state_dict(),
                'epoch':              epoch,
                'contrastive_weight': config.CONTRASTIVE_WEIGHT,
            }, save_path)
            print("Saved best model (Overall IoU {:.2f}) -> {}".format(
                overall_IoU, save_path), flush=True)

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print("Training complete. Total time {}".format(total_time), flush=True)


if __name__ == "__main__":
    main()
