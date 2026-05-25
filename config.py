# config.py — all paths and hyperparameters in one place.
# Your friend edits the PATHS section to match the A6000 machine.
# Nothing else in the codebase should need editing to run.

# ---- PATHS (set these on the machine that runs training) ----
CAMUS_DATA_DIR = "/path/to/CAMUS"          # where CAMUS lives on the lab machine
PRETRAINED_SWIN = "./pretrained_weights/swin_base_patch4_window12_384_22k.pth"
CHECKPOINT_DIR = "./checkpoints"

# ---- TRAINING HYPERPARAMETERS ----
BATCH_SIZE = 8
LR = 0.00005
WEIGHT_DECAY = 1e-2
EPOCHS = 40
IMG_SIZE = 480
SWIN_TYPE = "base"

# ---- METHOD HYPERPARAMETERS (used later) ----
# 0.0 = contrastive loss OFF (baseline). >0.0 = ON. This is your core ablation switch.
CONTRASTIVE_WEIGHT = 0.0