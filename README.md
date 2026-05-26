# CardioContrast

Language-guided echocardiographic segmentation built on LAVT, trained on
the CAMUS cardiac ultrasound dataset. Single-node dual-GPU training on
2x RTX A4000 (24GB each).

This README is written for Mohammed, who runs the experiments on the lab
machine. Suhaan (the author) writes all the code and pushes updates to
GitHub. Mohammed pulls updates and runs training at his convenience,
interleaved with his own experiments.

---

## Overview
1. One-time setup (environment, data, weights)
2. Edit config.py with your paths
3. Run experiments in order (baseline first, then contrastive)
4. Send back logs and checkpoints after each run

---

## One-Time Setup

### 1. Clone the repo
git clone https://github.com/SuhaanG/CardioContrast.git
cd CardioContrast

### 2. Set up the environment
First install PyTorch matching your CUDA version from https://pytorch.org.
Then install remaining dependencies:
pip install -r requirements.txt
If you see CUDA errors at startup, it is a PyTorch/CUDA version mismatch.
Send the full error to Suhaan.

### 3. Verify the GPUs are available
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
Must print "CUDA available: True" and "GPUs: 2". If not, fix the
PyTorch/CUDA install before continuing.

### 4. Download the CAMUS dataset
Register (free) and download from:
https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8

Locate the database_nifti folder inside the download. It contains folders
named patient0001, patient0002, etc. Note the FULL absolute path to this
folder (e.g. /data/ssd/CAMUS_public/database_nifti). You need it in step 6.

NOTE: the data loader reads files from disk on demand. It does NOT load
the full dataset into RAM. Memory usage during training stays low.

### 5. Download the Swin Transformer pretrained weights
mkdir -p pretrained_weights
Download this exact file and place it inside pretrained_weights/:
https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth

### 6. Edit config.py
Open config.py and set CAMUS_DATA_DIR to the full absolute path of your
database_nifti folder. This is the only line you need to change for the
baseline run:
```python
CAMUS_DATA_DIR = "/full/absolute/path/to/CAMUS_public/database_nifti"
```
IMPORTANT: use a full absolute path, not a relative path like ./data/CAMUS.
Leave all other settings as they are unless Suhaan says otherwise.

MEMORY NOTE: BATCH_SIZE = 2 with GRADIENT_ACCUMULATION_STEPS = 4 gives an
effective batch size of 8, calibrated for 24GB A4000 cards. If you see an
out-of-memory error, lower BATCH_SIZE to 1 in config.py.

---

## Running Experiments

Run experiments IN ORDER. Do not skip ahead.

### Step A: Create directories (do this once)
mkdir -p logs experiments/checkpoints

### Step B: Sanity check (do this before any full run)
Temporarily set EPOCHS = 1 in config.py, then:
python train_camus.py 2>&1 | tee logs/sanity_run.txt
Confirm you see:
- "[*] Device", "[*] GPUs", "[*] Effective batch" lines from startup
- "Total CAMUS examples: 6000"
- "Train: 4800  Val: 1200"
- "Building model: lavt_one"
- Per-step lines: "Epoch [0] step [0/...] loss X.XXXX"
- After the epoch: "Mean IoU: X.XX" and "Overall IoU: X.XX"

Set EPOCHS back to 40 once confirmed.
NOTE: if you see multiprocessing errors, set num_workers=0 in train_camus.py.

### Experiment 1: Baseline run (LAVT, no contrastive loss)
config.py: CONTRASTIVE_WEIGHT = 0.0 (default, do not change)
python train_camus.py 2>&1 | tee logs/baseline_run.txt
Checkpoint: experiments/checkpoints/model_best_camus.pth

### Experiment 2: CardioContrast run (with contrastive loss)
BEFORE RUNNING: open config.py and change:
```python
CONTRASTIVE_WEIGHT = 0.1
```
Then run:
python train_camus_contrastive.py 2>&1 | tee logs/contrastive_run.txt
Checkpoint: experiments/checkpoints/model_best_camus_contrastive.pth

The log prints "contrastive fired X% of steps" each epoch.
This should be close to 100%. If it shows 0%, stop and message Suhaan.

### Future experiments
Suhaan will push additional scripts and message you when ready.
Always git pull before starting a new experiment.

---

## After Each Run: Send Back
- The log file (e.g. logs/baseline_run.txt)
- The checkpoint from experiments/checkpoints/
  (large file — use Google Drive, NOT GitHub)

---

## Getting Updates
git pull

---

## If Something Breaks
Send Suhaan:
- The COMPLETE error message (all of it, not just the last line)
- Which experiment and which step failed
- The log file if training had already started

Suhaan will fix it, push, and let you know. Then git pull and retry.
