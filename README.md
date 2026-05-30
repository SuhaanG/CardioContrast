# CardioContrast

Language-guided echocardiographic segmentation with two contributions over LAVT:
1. Multi-stage decoder cross-attention — language injected at every decoder stage
2. Contrastive anatomical repulsion loss — penalizes similar decoder activations
   when different structure prompts are applied to the same image

Training dataset: CAMUS (3 cardiac structures).
Generalization test: EchoNet-Dynamic (held-out, never used for training).
Hardware: 2x RTX A4000 (24GB each).

This README is written for Mohammed. Suhaan writes all code and pushes to
GitHub. Mohammed pulls and runs experiments at his convenience.

---

## Overview
1. One-time setup (environment, data, weights)
2. Edit config.py with your paths
3. Run four experiments IN ORDER
4. Send back logs and checkpoints after each run

---

## One-Time Setup

### 1. Clone the repo
git clone https://github.com/SuhaanG/CardioContrast.git
cd CardioContrast

### 2. Set up the environment
Install PyTorch matching your CUDA version from https://pytorch.org, then:
pip install -r requirements.txt
If you see CUDA errors at startup it is a PyTorch/CUDA version mismatch.
Send the full error to Suhaan.

### 3. Verify GPUs
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
Must print "CUDA: True" and "GPUs: 2".

### 4. Download the CAMUS dataset
Register (free) and download from:
https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8

Locate the database_nifti folder (contains patient0001, patient0002, ...).
Note its FULL absolute path — you need it in step 6.

NOTE: the data loader streams from disk on demand. It does NOT load the
full dataset into RAM.

### 5. Download Swin pretrained weights
mkdir -p pretrained_weights
Download this exact file into pretrained_weights/:
https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth

### 6. Edit config.py
Set CAMUS_DATA_DIR to the full absolute path of your database_nifti folder.
This is the ONLY line you need to change for all four experiments:
```python
CAMUS_DATA_DIR = "/full/absolute/path/to/CAMUS_public/database_nifti"
```
Use a full absolute path. Do NOT use a relative path like ./data/CAMUS.

MEMORY NOTE: BATCH_SIZE=2 with GRADIENT_ACCUMULATION_STEPS=4 gives effective
batch size 8, calibrated for 24GB A4000 cards. If out-of-memory, lower
BATCH_SIZE to 1 in config.py.

---

## Four Experiments (run IN ORDER, do not skip ahead)

These four runs produce the complete ablation table for the paper.
Each experiment changes exactly one thing so the contribution of each
component can be measured independently.

| Exp | Script                       | DECODE_WITH_LANG | CONTRASTIVE_WEIGHT | Checkpoint saved                        |
|-----|------------------------------|------------------|--------------------|-----------------------------------------|
| 1   | train_camus.py               | —                | —                  | model_best_camus.pth                    |
| 2   | train_camus_contrastive.py   | True             | 0.0                | model_best_exp2_decoder_ca.pth          |
| 3   | train_camus_contrastive.py   | False            | 0.1                | model_best_exp3_contrastive_only.pth    |
| 4   | train_camus_contrastive.py   | True             | 0.1                | model_best_exp4_cardiocontrast.pth      |

Each experiment saves to its own checkpoint file. No run can overwrite another.

---

### Step A: Create directories (do this once before any experiment)
mkdir -p logs experiments/checkpoints

### Step B: Sanity check (do this before starting any full run, ever)
Set EPOCHS=1 in config.py temporarily, then:
python train_camus.py 2>&1 | tee logs/sanity_run.txt
Confirm you see ALL of these:
- "[*] Device", "[*] GPUs", "[*] Effective batch" lines
- "Total CAMUS examples: 6000"
- "Train: 4800  Val: 1200"
- "Building model: lavt_one"
- Per-step: "Epoch [0] step [0/...] loss X.XXXX"
- End of epoch: "Mean IoU: X.XX" and "Overall IoU: X.XX"

Set EPOCHS back to 40 after confirming. If multiprocessing errors appear,
set num_workers=0 in train_camus.py and retry.

---

### Experiment 1: Baseline (plain LAVT, no CardioContrast contributions)
config.py: no changes needed (defaults are correct)
python train_camus.py 2>&1 | tee logs/exp1_baseline.txt
Checkpoint: experiments/checkpoints/model_best_camus.pth

---

### Experiment 2: Decoder Cross-Attention Only (Contribution 1 alone)
In config.py set:
```python
DECODE_WITH_LANG   = True
CONTRASTIVE_WEIGHT = 0.0
```
Then run:
python train_camus_contrastive.py 2>&1 | tee logs/exp2_decoder_ca.txt
Checkpoint: experiments/checkpoints/model_best_exp2_decoder_ca.pth

The startup log should print:
  "Mode: Experiment 2 - Decoder cross-attention only"
If it prints something different, stop and message Suhaan.

---

### Experiment 3: Contrastive Loss Only (Contribution 2 alone)
In config.py set:
```python
DECODE_WITH_LANG   = False
CONTRASTIVE_WEIGHT = 0.1
```
Then run:
python train_camus_contrastive.py 2>&1 | tee logs/exp3_contrastive_only.txt
Checkpoint: experiments/checkpoints/model_best_exp3_contrastive_only.pth

The startup log should print:
  "Mode: Experiment 3 - Contrastive loss only"
Also watch for: "contrastive fired X% of steps" each epoch.
This should be close to 100%. If it shows 0%, stop and message Suhaan.

---

### Experiment 4: Full CardioContrast (both contributions)
In config.py set:
```python
DECODE_WITH_LANG   = True
CONTRASTIVE_WEIGHT = 0.1
```
Then run:
python train_camus_contrastive.py 2>&1 | tee logs/exp4_cardiocontrast.txt
Checkpoint: experiments/checkpoints/model_best_exp4_cardiocontrast.pth

The startup log should print:
  "Mode: Experiment 4 - Full CardioContrast (both contributions)"
Also watch for: "contrastive fired X% of steps" — should be close to 100%.

---

## After Each Run: Send Back
- The log file for that experiment (logs/exp1_baseline.txt, etc.)
- The checkpoint file from experiments/checkpoints/
  (large file — send via Google Drive, NOT GitHub)

Send each run's results immediately after it finishes.
Do not wait for all four to complete before sending.
This lets Suhaan check things are working before the next run starts.

---

## Getting Updates
git pull
Always git pull before starting a new experiment.

---

## If Something Breaks
Send Suhaan:
- The COMPLETE error message (all of it, not just the last line)
- Which experiment failed and which step
- The log file if training had already started

Suhaan will fix it, push, and let you know. Then git pull and retry.
