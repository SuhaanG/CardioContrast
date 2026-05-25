# CardioContrast

Language-guided echocardiographic segmentation. This repository adapts LAVT
(Language-Aware Vision Transformer) to the CAMUS cardiac ultrasound dataset,
training on a single GPU.

This README is a complete step-by-step guide for setting up and running
training on the A6000 machine. The data pipeline has already been written and
verified; the steps below are everything needed to run it on the GPU.

---

## Overview of steps
1. Clone the repository
2. Set up the Python environment
3. Verify the GPU is available
4. Download the CAMUS dataset
5. Download the Swin Transformer pretrained weights
6. Edit config.py with your paths
7. Do a quick 1-epoch sanity run
8. Run full training
9. Send back the log and the trained checkpoint

---

## 1. Clone the repository
git clone <REPO_URL>
cd CardioContrast
Replace <REPO_URL> with the link to this GitHub repository.

## 2. Set up the Python environment
Use Python 3.9 or newer. Create a fresh environment, then install packages:
pip install torch torchvision
pip install nibabel numpy pillow
pip install requests tqdm regex sacremoses sentencepiece filelock packaging tokenizers

IMPORTANT — PyTorch version:
LAVT was originally written for PyTorch 1.7.1, which is too old for a modern
A6000. Install a current PyTorch build that matches this machine's CUDA
version. Get the exact install command from https://pytorch.org (select your
CUDA version). If you see CUDA or import errors at startup, they are almost
certainly a version mismatch — copy the FULL error and send it back.

## 3. Verify the GPU is available
Before going further, confirm PyTorch can see the GPU:
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
This must print "CUDA available: True". If it prints False, the PyTorch/CUDA
install is wrong — fix this before continuing.

## 4. Download the CAMUS dataset
Download from the Human Heart Project platform (free registration required):
https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8

Download the CAMUS_public collection. Inside it, find the `database_nifti`
folder — it contains patient folders (patient0001, patient0002, ...). Note the
full path to this `database_nifti` folder; you will need it in step 6.

## 5. Download the Swin Transformer pretrained weights
The model needs these weights to initialize. Download this exact file:
https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth

Then place it in a pretrained_weights folder inside the repo:
mkdir -p pretrained_weights
Move the downloaded .pth file into pretrained_weights/.

## 6. Edit config.py
Open config.py and set these two paths to match THIS machine:
```python
CAMUS_DATA_DIR = "/full/path/to/CAMUS_public/database_nifti"
PRETRAINED_SWIN = "./pretrained_weights/swin_base_patch4_window12_384_22k.pth"
```
Leave all other settings as they are for now.

## 7. Quick 1-epoch sanity run (do this first)
Before a long run, confirm one epoch completes. In config.py temporarily set:
```python
EPOCHS = 1
```
Then run:
python train_camus.py 2>&1 | tee training_log.txt
The `2>&1 | tee training_log.txt` part saves all output to training_log.txt
while also showing it on screen. The first run will automatically download the
BERT model (bert-base-uncased) — this is expected and happens once.

What you should see if it is working:
- "Total CAMUS examples: 6000"
- "Train: 4800  Val: 1200"
- "Building model: lavt_one"
- Per-step lines: "Epoch [0] step [0/600] loss 0.xxxx lr 0.0000xx"
- After the epoch: "Mean IoU" and "Overall IoU" numbers
- A saved file: checkpoints/model_best_camus.pth

If startup hangs for more than ~15 minutes with no output at all, stop it and
send the log.

## 8. Run full training
Once the 1-epoch run completes successfully, set EPOCHS back to 40 in
config.py and run again:
python train_camus.py 2>&1 | tee training_log.txt

## 9. Send back
- training_log.txt (the full log)
- checkpoints/model_best_camus.pth (the trained model — this is large, send via
  Google Drive or similar, NOT GitHub)

---

## If something breaks
This is normal on a first run of research code in a new environment. Send back:
- The COMPLETE error message (all of it, not just the last line)
- Which step it failed at (environment, GPU check, data loading, model build,
  or training)
The fix will be pushed to the repo; then pull the update and re-run:
git pull
