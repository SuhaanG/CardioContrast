# CardioContrast

Language-guided echocardiographic segmentation built on LAVT, trained on
the CAMUS cardiac ultrasound dataset. Single-node multi-GPU training.

This README is written for Mohammed, who runs the experiments on the lab
machine. Suhaan (the author) writes all the code and pushes updates to
GitHub. Mohammed pulls updates and runs training at his convenience,
interleaved with his own experiments.

---

## Overview
1. One-time setup (environment, data, weights)
2. Edit config.py with your paths
3. Run a 1-epoch sanity check
4. Run full training
5. Send back the log and checkpoint

---

## One-Time Setup

### 1. Clone the repo
git clone https://github.com/SuhaanG/CardioContrast.git
cd CardioContrast

### 2. Set up the environment
Use Python 3.9+. Install dependencies:
pip install torch torchvision
pip install nibabel numpy pillow
pip install requests tqdm regex sacremoses sentencepiece filelock packaging tokenizers
IMPORTANT: install a PyTorch build that matches the lab machine's CUDA
version. Get the exact command from https://pytorch.org. If you see CUDA
errors at startup, it is a version mismatch — send the full error to Suhaan.

### 3. Verify the GPU is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
Must print "CUDA available: True". If False, fix the PyTorch/CUDA install
before continuing.

### 4. Download the CAMUS dataset
Register (free) and download from:
https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8

Locate the database_nifti folder inside the download. It contains folders
named patient0001, patient0002, etc. Note the full path to database_nifti.

NOTE: the data loader reads files from disk on demand (streaming), it does
NOT load the full dataset into RAM. Memory usage during training is low.

### 5. Download the Swin Transformer pretrained weights
mkdir -p pretrained_weights
Download this exact file and place it in pretrained_weights/:
https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth

---

## Before Each Run: Edit config.py

Open config.py and set the two paths to match this machine:
```python
CAMUS_DATA_DIR = "/full/path/to/CAMUS_public/database_nifti"
PRETRAINED_SWIN = "./pretrained_weights/swin_base_patch4_window12_384_22k.pth"
```
Leave all other settings as they are unless instructed otherwise by Suhaan.

BATCH SIZE NOTE: default is 4, safe for 24GB A4000 cards. If you see an
out-of-memory error, lower BATCH_SIZE to 2 in config.py.

---

## Running Experiments

Each experiment is a separate run. Run them one at a time at your
convenience — there is no dependency between runs, and each saves its own
checkpoint. Always save the log for each run.

### Sanity check (do this first, ever)
Temporarily set EPOCHS = 1 in config.py, then:
python train_camus.py 2>&1 | tee logs/sanity_run.txt
Confirm you see:
- "Total CAMUS examples: 6000"
- "Train: 4800  Val: 1200"
- "Building model: lavt_one"
- Per-step loss lines
- "Mean IoU" and "Overall IoU" after the epoch

Set EPOCHS back to 40 once confirmed.

### Full baseline run (first real experiment)
mkdir -p logs
python train_camus.py 2>&1 | tee logs/baseline_run.txt

### Future experiments (Suhaan will push these)
Additional scripts will appear in the repo as the project progresses
(e.g., train_camus_contrastive.py). Each will have its own log name.
Suhaan will message you when a new script is ready to run.

---

## After Each Run: Send Back

- The log file (e.g., logs/baseline_run.txt)
- The checkpoint: checkpoints/model_best_camus.pth
  (large file — send via Google Drive, NOT GitHub)

---

## Getting Updates

When Suhaan pushes new code, pull before running:
git pull

## If Something Breaks

Send Suhaan:
- The COMPLETE error (all of it, not just the last line)
- Which step failed
- The log file if training started before the error

Suhaan will fix it and push. Then git pull and retry.
