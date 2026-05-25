# config.py — all paths and hyperparameters in one place.
# Tailored for the shared 2x RTX A4000 (24GB) workstation layout.

import os
import torch

# =====================================================================
# 1. PATHS & DIRECTORIES (SSD Storage-Aware)
# =====================================================================
# Hardcoded to stream from local storage paths instead of caching to shared 128GB RAM
CAMUS_DATA_DIR = "./data/CAMUS"            # Where CAMUS lives on your local SSD
ECHONET_DATA_DIR = "./data/EchoNet"        # Prepped placeholder for your generalization dataset
PRETRAINED_SWIN = "./pretrained_weights/swin_base_patch4_window12_384_22k.pth"
CHECKPOINT_DIR = "./experiments/checkpoints"
OUTPUT_DIR = "./experiments/outputs"

# =====================================================================
# 2. COMPUTE CONFIGURATION (Dual-GPU Distribution Setup)
# =====================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Explicitly handles multi-GPU tracking across your 2x A4000 setup
GPU_IDS = [0, 1] if torch.cuda.device_count() > 1 else [0]

# =====================================================================
# 3. TRAINING & MEMORY-SAFE HYPERPARAMETERS (VRAM OOM Mitigation)
# =====================================================================
# Crucial Change: Setting the physical batch size to 8 on a 24GB A4000 with Swin-Base
# at 480x480 resolution will trigger an instant OOM error.
# We set physical BATCH_SIZE = 2 and use 4 GRADIENT_ACCUMULATION_STEPS.
# Mathematical result: 2 (samples) * 4 (accumulations) = Effective Batch Size of 8.
BATCH_SIZE = 2  
GRADIENT_ACCUMULATION_STEPS = 4  

LR = 0.00005
WEIGHT_DECAY = 1e-2
EPOCHS = 40
IMG_SIZE = 480
SWIN_TYPE = "base"

# =====================================================================
# 4. METHOD HYPERPARAMETERS (Core Algorithmic Ablation Switches)
# =====================================================================
# 0.0 = contrastive loss OFF (baseline). >0.0 = ON. 
# This serves as the switch your independent sweep scripts will toggle.
CONTRASTIVE_WEIGHT = 0.0
USE_MULTI_STAGE_ATTN = True  # Activates text-prompt injection at the decoder levels

# =====================================================================
# ENVIRONMENT INITIALIZER HOOK
# =====================================================================
def initialize_environment():
    """Systematically builds directories so experiments do not conflict on the SSD."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] Workspace Initialized on Device: {DEVICE}")
    print(f"[*] Target GPUs: {GPU_IDS} | Available Memory Setup: 2x 24GB")
    print(f"[*] Batch Execution Strategy: Physical Batch={BATCH_SIZE} | Accumulation Steps={GRADIENT_ACCUMULATION_STEPS}")
    print(f"[*] Current Operational Architecture: Contrastive Weight Flag = {CONTRASTIVE_WEIGHT}")

if __name__ == "__main__":
    initialize_environment()
