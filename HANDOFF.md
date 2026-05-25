# CardioContrast — Training Handoff Instructions

This document tells you (the person running training on the A6000) exactly
how to set up and run CardioContrast. The author wrote and verified all the
code and the data pipeline; your job is to run training on the GPU and send
back the results.

---

## What you need to do (overview)
1. Clone this repo onto the A6000 machine
2. Set up the Python environment
3. Download the CAMUS dataset and the Swin pretrained weights
4. Edit `config.py` to point at where you put them
5. Run `python train_camus.py`
6. Send back the training log and the saved checkpoint

---

## 1. Clone the repo
git clone <this repo URL>
cd CardioContrast

## 2. Set up the environment
Create a fresh conda or venv environment with Python 3.9+ and install:
pip install torch torchvision
pip install nibabel numpy pillow
pip install requests tqdm regex sacremoses sentencepiece filelock packaging tokenizers
NOTE ON VERSIONS: LAVT was originally built for PyTorch 1.7.1, but that is too
old for a modern A6000. Please install a recent PyTorch build that matches the
machine's CUDA version (see https://pytorch.org for the right command). If you
hit import or CUDA errors at startup, they are almost certainly a
version/CUDA-compatibility issue — please copy the full error and send it back.

## 3. Download the data and weights (these are NOT in the repo)

### a) CAMUS dataset
Download from the Human Heart Project platform (register, free).
Locate the `database_nifti` folder. It contains patient folders (patient0001, ...).

### b) Swin Transformer pretrained weights
Download this exact file (needed to initialize the model):
https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
Then:
mkdir -p pretrained_weights
move the downloaded .pth into pretrained_weights/

## 4. Edit config.py
Open `config.py` and set these two paths to match THIS machine:
```python
CAMUS_DATA_DIR = "/full/path/to/CAMUS_public/database_nifti"
PRETRAINED_SWIN = "./pretrained_weights/swin_base_patch4_window12_384_22k.pth"
```
Leave the other settings as they are for the first run.

## 5. Run training
python train_camus.py
The first run will download the BERT tokenizer/model (bert-base-uncased)
automatically — this is expected and only happens once.

### What you should see if it's working
- "Total CAMUS examples: 6000"
- "Train: 4800  Val: 1200"
- "Building model: lavt_one"
- Then per-step lines like: "Epoch [0] step [0/600] loss 0.xxxx lr 0.0000xx"
- After each epoch: "Mean IoU" and "Overall IoU" numbers
- A saved file: checkpoints/model_best_camus.pth

## 6. Send back
- The full terminal output / log (copy it to a text file)
- The saved checkpoint: checkpoints/model_best_camus.pth
  (this is large — send via Google Drive or similar, NOT GitHub)

---

## If something breaks
This is expected on a first run of research code in a new environment.
Please send back:
- The FULL error message (all of it, not just the last line)
- Which step it failed at (setup, data loading, model building, or training)
The author will fix it and push an update for you to pull.

## Quick sanity check before full training (recommended)
Set EPOCHS = 1 in config.py for a first quick pass to confirm one epoch
completes, then set it back to 40 for the real run.
