# config.py — all paths and hyperparameters in one place.
# Mohammed: edit the PATHS section below to match the lab machine.
# Nothing else needs to be edited to run training.

import os
import torch

# =====================================================================
# 1. PATHS — EDIT THESE TO MATCH YOUR MACHINE
# =====================================================================
CAMUS_DATA_DIR   = "/path/to/CAMUS_public/database_nifti"
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
BATCH_SIZE                  = 4
GRADIENT_ACCUMULATION_STEPS = 4
LR           = 0.00005
WEIGHT_DECAY = 1e-2
EPOCHS       = 40
IMG_SIZE     = 480
SWIN_TYPE    = "base"
SEED         = 42

# =====================================================================
# 4. ABLATION CONTROL
# =====================================================================
# Four experiment conditions for the full ablation table:
#
#   Exp 1 - Baseline (plain LAVT, no contributions):
#     Script: train_camus.py
#     Settings: (this script ignores both flags below)
#
#   Exp 2 - Decoder cross-attention only (Contribution 1 alone):
#     Script: train_camus_contrastive.py
#     Settings: DECODE_WITH_LANG=True, CONTRASTIVE_WEIGHT=0.0
#
#   Exp 3 - Contrastive loss only (Contribution 2 alone):
#     Script: train_camus_contrastive.py
#     Settings: DECODE_WITH_LANG=False, CONTRASTIVE_WEIGHT=0.1
#
#   Exp 4 - Full CardioContrast (both contributions):
#     Script: train_camus_contrastive.py
#     Settings: DECODE_WITH_LANG=True, CONTRASTIVE_WEIGHT=0.1
#
# Comparisons:
#   Exp 1 vs Exp 2 = isolates decoder CA contribution alone
#   Exp 1 vs Exp 3 = isolates contrastive loss contribution alone
#   Exp 2 vs Exp 4 = adds contrastive on top of CA
#   Exp 3 vs Exp 4 = adds CA on top of contrastive
#   Exp 1 vs Exp 4 = full CardioContrast improvement over baseline

# DECODE_WITH_LANG: decoder cross-attention ablation switch (Contribution 1)
# True  = language injected into decoder at every refinement stage
# False = baseline decoder, no language conditioning in decoder
DECODE_WITH_LANG = True

# CONTRASTIVE_WEIGHT: contrastive loss ablation switch (Contribution 2)
# 0.0 = contrastive loss OFF
# 0.1 = contrastive loss ON (recommended value)
CONTRASTIVE_WEIGHT = 0.0

# CONTRASTIVE_TAU: temperature for the contrastive repulsion loss.
# Fixed at 0.07 following SimCLR. Ablate if reviewers ask.
CONTRASTIVE_TAU = 0.07

# =====================================================================
# ENVIRONMENT INITIALIZER
# =====================================================================
def initialize_environment():
    for d in [CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
    print("[*] Device          : {}".format(DEVICE))
    print("[*] GPUs             : {}".format(GPU_IDS))
    print("[*] Effective batch  : {} x {} = {}".format(
        BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
        BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
    print("[*] Contrastive      : {}".format(
        "ON (weight={}, tau={})".format(CONTRASTIVE_WEIGHT, CONTRASTIVE_TAU)
        if CONTRASTIVE_WEIGHT > 0 else "OFF"))
    print("[*] Decoder CA       : {}".format(
        "ON" if DECODE_WITH_LANG else "OFF (baseline decoder)"))
    print("[*] CAMUS data dir   : {}".format(CAMUS_DATA_DIR))

if __name__ == "__main__":
    initialize_environment()
