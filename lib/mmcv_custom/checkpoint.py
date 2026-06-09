# lib/mmcv_custom/checkpoint.py
# mmcv dependency removed -- replaced with stdlib + torch equivalents.

import os
import os.path as osp
import time
import torch
import torch.distributed as dist


def _get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def mkdir_or_exist(dir_name):
    os.makedirs(dir_name, exist_ok=True)


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
    if not osp.isfile(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")

    checkpoint = torch.load(filename, map_location=map_location or "cpu")

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if logger:
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")

    return checkpoint


def save_checkpoint(model, filename, optimizer=None, meta=None):
    mkdir_or_exist(osp.dirname(filename))
    meta = meta or {}
    meta.update(time=time.asctime())

    checkpoint = {
        "meta": meta,
        "state_dict": model.state_dict(),
    }
    if optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()

    torch.save(checkpoint, filename)
