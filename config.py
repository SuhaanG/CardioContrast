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

# Random seed — controls train/val split AND all model random ops.
# Do NOT change this for any reported experiment.
SEED = 42

# =====================================================================
# 4. METHOD HYPERPARAMETERS
# =====================================================================
# CONTRASTIVE_WEIGHT controls the ablation:
#
#   Experiment 1 — Baseline (LAVT, no contributions):
#     Use train_camus.py with CONTRASTIVE_WEIGHT = 0.0 (default)
#
#   Experiment 2 — Decoder cross-attention only (Contribution 1):
#     Use train_camus_contrastive.py with CONTRASTIVE_WEIGHT = 0.0
#
#   Experiment 3 — Full CardioContrast (both contributions):
#     Use train_camus_contrastive.py with CONTRASTIVE_WEIGHT = 0.1
#
# Comparing Exp 1 vs 2 isolates the decoder cross-attention contribution.
# Comparing Exp 2 vs 3 isolates the contrastive loss contribution.
# Comparing Exp 1 vs 3 shows the full combined improvement.
CONTRASTIVE_WEIGHT = 0.0

# CONTRASTIVE_TAU: temperature for the contrastive repulsion loss.
# Fixed at 0.07 following SimCLR. Ablate if reviewers ask about sensitivity.
CONTRASTIVE_TAU = 0.07

# USE_MULTI_STAGE_ATTN: decoder cross-attention is now IMPLEMENTED.
# This flag documents the state of the method — do not change it manually.
# Controlled by which script you run:
#   train_camus.py             -> decode_with_lang=False (baseline, CA inactive)
#   train_camus_contrastive.py -> decode_with_lang=True  (CardioContrast, CA active)
USE_MULTI_STAGE_ATTN = True

# =====================================================================
# ENVIRONMENT INITIALIZER — called automatically at training startup
# =====================================================================
def initialize_environment():
    for d in [CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
    print("[*] Device         : {}".format(DEVICE))
    print("[*] GPUs            : {}".format(GPU_IDS))
    print("[*] Effective batch : {} physical x {} accumulation = {}".format(
        BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
        BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
    print("[*] Contrastive     : {}".format(
        "ON (weight={}, tau={})".format(CONTRASTIVE_WEIGHT, CONTRASTIVE_TAU)
        if CONTRASTIVE_WEIGHT > 0 else "OFF (baseline or decoder-CA-only)"))
    print("[*] CAMUS data dir  : {}".format(CAMUS_DATA_DIR))

if __name__ == "__main__":
    initialize_environment()
