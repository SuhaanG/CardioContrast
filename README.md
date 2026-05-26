# CardioContrast

Language-guided echocardiographic segmentation. Two contributions over the
LAVT baseline:
  1. Multi-stage decoder cross-attention — language injected at every decoder
     refinement stage, not just the encoder.
  2. Contrastive anatomical repulsion loss — penalizes similar decoder
     activations when different structure prompts are applied to the same image.

Training on CAMUS (3 cardiac structures). Generalization test on EchoNet-Dynamic.
Single-node dual-GPU: 2x RTX A4000 (24GB each).

This README is written for Mohammed, who runs the experiments on the lab
machine. Suhaan writes all the code and pushes updates. Mohammed pulls
and runs at his convenience, interleaved with his own work.

---

## Overview
1. One-time setup (environment, data, weights)
2. Edit config.py with your paths
3. Run three experiments IN ORDER
4. Send back logs and checkpoints after each run

---

## One-Time Setup

### 1. Clone the repo
git clone https://github.com/SuhaanG/CardioContrast.git
cd CardioContrast

### 2. Set up the environment
First install PyTorch matching your CUDA version from https://pytorch.org.
Then:
pip install -r requirements.txt
If you see CUDA errors at startup, it is a PyTorch/CUDA version mismatch.
Send the full error to Suhaan.

### 3. Verify the GPUs are available
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
Must print "CUDA available: True" and "GPUs: 2".

### 4. Download the CAMUS dataset
Register (free) and download from:
https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8

Locate the database_nifti folder (contains patient0001, patient0002, ...).
Note its FULL absolute path — you need it in step 6.

NOTE: the data loader streams from disk on demand. It does NOT load the
full dataset into RAM.

### 5. Download the Swin Transformer pretrained weights
mkdir -p pretrained_weights
Download and place inside pretrained_weights/:
https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth

### 6. Edit config.py
Set CAMUS_DATA_DIR to the full absolute path of your database_nifti folder:
```python
CAMUS_DATA_DIR = "/full/absolute/path/to/CAMUS_public/database_nifti"
```
This is the ONLY line you need to change. Leave everything else unless
Suhaan tells you otherwise.

MEMORY NOTE: BATCH_SIZE=2 with GRADIENT_ACCUMULATION_STEPS=4 gives an
effective batch size of 8, calibrated for 24GB A4000 cards. If you see
an out-of-memory error, lower BATCH_SIZE to 1 in config.py.

---

## Three Experiments (run IN ORDER, do not skip ahead)

These three runs produce the core ablation table of the paper:
- Exp 1 vs 2 isolates the decoder cross-attention contribution
- Exp 2 vs 3 isolates the contrastive loss contribution
- Exp 1 vs 3 shows the full CardioContrast improvement

### Step A: Create directories (once)
mkdir -p logs experiments/checkpoints

### Step B: Sanity check (do before any full run)
Set EPOCHS = 1 in config.py temporarily, then:
python train_camus.py 2>&1 | tee logs/sanity_run.txt
Confirm you see:
- "[*] Device", "[*] GPUs", "[*] Effective batch" startup lines
- "Total CAMUS examples: 6000"
- "Train: 4800  Val: 1200"
- "Building model: lavt_one"
- Per-step: "Epoch [0] step [0/...] loss X.XXXX"
- End of epoch: "Mean IoU: X.XX" and "Overall IoU: X.XX"

Set EPOCHS back to 40 once confirmed.
NOTE: if you see multiprocessing errors, set num_workers=0 in train_camus.py.

---

### Experiment 1: Baseline (LAVT — no CardioContrast contributions)
config.py: CONTRASTIVE_WEIGHT = 0.0 (default, do not change)
python train_camus.py 2>&1 | tee logs/exp1_baseline.txt
Checkpoint: experiments/checkpoints/model_best_camus.pth

What this is: plain LAVT adapted to CAMUS. Language goes to the encoder
only. No decoder cross-attention. No contrastive loss. This is the
control group everything else is compared against.

---

### Experiment 2: Decoder Cross-Attention Only (Contribution 1 alone)
In config.py, set:
```python
CONTRASTIVE_WEIGHT = 0.0
```
Then run:
python train_camus_contrastive.py 2>&1 | tee logs/exp2_decoder_ca.txt
Checkpoint: experiments/checkpoints/model_best_camus_contrastive.pth

What this is: adds multi-stage decoder cross-attention (language injected
at 3 decoder stages) but WITHOUT the contrastive loss. Comparing this to
Experiment 1 isolates the decoder cross-attention contribution alone.

NOTE: even though CONTRASTIVE_WEIGHT=0.0, this script activates
decode_with_lang=True, so the decoder cross-attention IS active.

---

### Experiment 3: Full CardioContrast (both contributions)
In config.py, set:
```python
CONTRASTIVE_WEIGHT = 0.1
```
Then run:
python train_camus_contrastive.py 2>&1 | tee logs/exp3_cardiocontrast.txt
Checkpoint: experiments/checkpoints/model_best_camus_contrastive.pth

What this is: adds BOTH decoder cross-attention AND the contrastive
anatomical repulsion loss. This is the full CardioContrast method.
Comparing to Experiment 2 isolates the contrastive loss contribution.

The log will print "contrastive fired X% of steps" each epoch.
This should be close to 100%. If it shows 0%, stop and message Suhaan.

---

## After Each Run: Send Back
- The log file (logs/exp1_baseline.txt, exp2_decoder_ca.txt, or
  exp3_cardiocontrast.txt)
- The checkpoint from experiments/checkpoints/
  (large file — send via Google Drive, NOT GitHub)

Please send each run's results as soon as it finishes, not all at once
at the end. This lets Suhaan check things are working before the next run.

---

## Getting Updates
git pull
Always pull before starting a new experiment.

---

## If Something Breaks
Send Suhaan:
- The COMPLETE error message (all of it, not just the last line)
- Which experiment failed and which step
- The log file if training had already started

Suhaan will fix it, push, and let you know. Then git pull and retry.
