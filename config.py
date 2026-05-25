# config.py — all paths and hyperparameters in one place.
# Mohammed: edit the PATHS section below to match the lab machine.
# Nothing else needs to be edited to run training.

import os
import torch

# =====================================================================
# 1. PATHS — EDIT THESE TO MATCH YOUR MACHINE
# =====================================================================
# Set CAMUS_DATA_DIR to the full absolute path of your database_nifti folder.
# Example: "/data/ssd/CAMUS_public/database_nifti"
# DO NOT use a relative path like "./data/CAMUS" — it will not find the data.
CAMUS_DATA_DIR = "/path/to/CAMUS_public/database_nifti"

ECHONET_DATA_DIR = "/path/to/EchoNet"
PRETRAINED_SWIN  = "./pretrained_weights/swin_base_patch4_window12_384_22k.pth"
CHECKPOINT_DIR   = "./experiments/checkpoints"
OUTPUT_DIR       = "./experiments/outputs"
LOG_DIR          = "./logs"

# =====================================================================
# 2. COMPUTE CONFIGURATION
# =====================================================================
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
GPU_IDS = [0, 1] if torch.cuda.device_count() > 1 else [0]

# =====================================================================
# 3. TRAINING HYPERPARAMETERS
# =====================================================================
# BATCH_SIZE = 2 with GRADIENT_ACCUMULATION_STEPS = 4 gives an
# effective batch size of 8, safe for 24 GB A4000 cards.
# If you still see an out-of-memory error, lower BATCH_SIZE to 1.
BATCH_SIZE                  = 2
GRADIENT_ACCUMULATION_STEPS = 4

LR           = 0.00005
WEIGHT_DECAY = 1e-2
EPOCHS       = 40
IMG_SIZE     = 480
SWIN_TYPE    = "base"

# =====================================================================
# 4. METHOD HYPERPARAMETERS
# =====================================================================
# CONTRASTIVE_WEIGHT: 0.0 = baseline (contrastive loss OFF).
# Set > 0.0 to enable CardioContrast contrastive loss (future experiment).
CONTRASTIVE_WEIGHT = 0.0

# USE_MULTI_STAGE_ATTN: future flag for decoder cross-attention.
# Currently NOT active in the training script — do not change.
USE_MULTI_STAGE_ATTN = False

# =====================================================================
# ENVIRONMENT INITIALIZER — called automatically at training startup
# =====================================================================
def initialize_environment():
    for d in [CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
    print("[*] Device        : {}".format(DEVICE))
    print("[*] GPUs           : {}".format(GPU_IDS))
    print("[*] Batch size     : {} (physical) x {} (accumulation) = {} effective".format(
        BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
        BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
    print("[*] Contrastive    : {}".format(
        "ON (weight={})".format(CONTRASTIVE_WEIGHT)
        if CONTRASTIVE_WEIGHT > 0 else "OFF (baseline)"))
    print("[*] CAMUS data dir : {}".format(CAMUS_DATA_DIR))

if __name__ == "__main__":
    initialize_environment()
