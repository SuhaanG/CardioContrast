# CardioContrast

Language-guided echocardiographic segmentation. Two contributions over the
LAVT baseline:
  1. Multi-stage decoder cross-attention — language injected at every decoder
     refinement stage, not just the encoder.
  2. Contrastive anatomical repulsion loss — penalizes similar decoder
     activations when different structure prompts are applied to the same image.

Training on CAMUS (3 cardiac structures). Generalization test on EchoNet-Dynamic.
Single-node dual-GPU: 2x RTX A4000 (24GB each).

This README is written for Mohammed, who runs the experiments on the lab machine.
Suhaan writes all the code and pushes updates. Mohammed pulls and runs at his
convenience, interleaved with his own work.

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

### 3. Verify GPUs
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
Must print "CUDA: True" and "GPUs: 2".

### 4. Download the CAMUS dataset
Register (free) and download from:
https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8

Locate the database_nifti folder (contains patient0001, patient0002, ...).
Note its FULL absolute path.

### 5. Download Swin pretrained weights
mkdir -p pretrained_weights
Download into pretrained_weights/:
https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth

### 6. Edit config.py
Set CAMUS_DATA_DIR to the full absolute path of your database_nifti folder:
```python
CAMUS_DATA_DIR = "/full/absolute/path/to/CAMUS_public/database_nifti"
```
This is the ONLY line you need to change. Leave everything else unless
Suhaan says otherwise.

MEMORY NOTE: BATCH_SIZE=2 with GRADIENT_ACCUMULATION_STEPS=4 gives effective
batch size 8, calibrated for 24GB A4000 cards. If out-of-memory, lower
BATCH_SIZE to 1.

---

## Four Experiments (run IN ORDER, do not skip ahead)

These four runs produce the complete ablation table for the paper.
Each experiment isolates one variable so the contribution of each
component can be measured independently.

| Exp | What is active | Script | DECODE_WITH_LANG | CONTRASTIVE_WEIGHT |
|-----|---------------|--------|------------------|--------------------|
| 1   | Nothing (baseline) | train_camus.py | — | — |
| 2   | Decoder CA only | train_camus_contrastive.py | True | 0.0 |
| 3   | Contrastive only | train_camus_contrastive.py | False | 0.1 |
| 4   | Both (CardioContrast) | train_camus_contrastive.py | True | 0.1 |

### Step A: Create directories (once)
mkdir -p logs experiments/checkpoints

### Step B: Sanity check (before any full run)
Set EPOCHS=1 in config.py temporarily, then:
python train_camus.py 2>&1 | tee logs/sanity_run.txt
Confirm you see: "Total CAMUS examples: 6000", "Train: 4800  Val: 1200",
per-step loss lines, and "Mean IoU / Overall IoU" at epoch end.
Set EPOCHS back to 40 after confirming.

---

### Experiment 1: Baseline (plain LAVT, no CardioContrast contributions)
No config changes needed (defaults are correct).
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
Checkpoint: experiments/checkpoints/model_best_camus_contrastive.pth

The startup log should print: "Mode: Experiment 2 - Decoder cross-attention only"

---

### Experiment 3: Contrastive Loss Only (Contribution 2 alone)
In config.py set:
```python
DECODE_WITH_LANG   = False
CONTRASTIVE_WEIGHT = 0.1
```
Then run:
python train_camus_contrastive.py 2>&1 | tee logs/exp3_contrastive_only.txt
Checkpoint: experiments/checkpoints/model_best_camus_contrastive.pth

The startup log should print: "Mode: Experiment 3 - Contrastive loss only"
Also watch for: "contrastive fired X% of steps" — should be close to 100%.
If it shows 0%, stop and message Suhaan.

---

### Experiment 4: Full CardioContrast (both contributions)
In config.py set:
```python
DECODE_WITH_LANG   = True
CONTRASTIVE_WEIGHT = 0.1
```
Then run:
python train_camus_contrastive.py 2>&1 | tee logs/exp4_cardiocontrast.txt
Checkpoint: experiments/checkpoints/model_best_camus_contrastive.pth

The startup log should print: "Mode: Experiment 4 - Full CardioContrast"
Also watch for: "contrastive fired X% of steps" — should be close to 100%.

---

## After Each Run: Send Back
- The log file (logs/exp1_baseline.txt, exp2_decoder_ca.txt, etc.)
- The checkpoint from experiments/checkpoints/
  (large file — send via Google Drive, NOT GitHub)
- Send each run immediately after it finishes, not all at once.

---

## Getting Updates
git pull
Always pull before starting a new experiment.

---

## If Something Breaks
Send Suhaan:
- The COMPLETE error message (all of it, not just the last line)
- Which experiment and which step failed
- The log file if training had already started

Suhaan will fix it, push, and let you know. Then git pull and retry.
