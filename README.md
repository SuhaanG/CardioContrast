# CardioContrast

CardioContrast is a language-guided echocardiographic segmentation framework for the CAMUS benchmark. The project combines multi-stage language-conditioned decoding with a contrastive anatomical repulsion objective to improve cardiac structure segmentation from text prompts.

## Highlights

- Patient-level CAMUS split protocol with strict train/val/test separation
- Prompt-bank utilities for canonical, held-out, and training phrase sets
- Multi-stage decoder cross-attention conditioning for structure-aware decoding
- Contrastive anatomical repulsion loss for same-image, different-structure separation
- Native medical metrics for Dice, IoU, distance-based agreement, and cardiac volume/error proxies
- CPU-only regression suite for protocol and model sanity checks

## Project status

This repository is structured for publication-quality experimentation and reproducible scientific validation. The core protocol and model interface are being kept configuration-driven, with all major data paths and training settings exposed through the config layer rather than hard-coded to any local machine or cloud environment.

## Repository layout

- `config.py` — environment-driven configuration, presets, and experiment metadata
- `data/` — dataset indexing, prompt banks, and patient split logic
- `lib/` — model blocks, contrastive loss, metrics, and decoder utilities
- `tests/test_core.py` — lightweight regression suite for protocol and model invariants
- `run_ablation_suite.py` — reproducible ablation runner across the main experimental presets
- `report_experiments.py` — summary script to aggregate experiment logs into a paper-friendly report
- `train_camus.py` — CAMUS training entry point
- `train_camus_contrastive.py` — contrastive and language-conditioned experiments

## Setup

### 1. Clone

```bash
git clone https://github.com/SuhaanG/CardioContrast.git
cd CardioContrast
```

### 2. Create a Python environment

Install PyTorch for your platform first, then install the project dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure paths

All data paths are kept configurable via environment variables and `config.py`. For example:

```bash
export CAMUS_DATA_DIR="/absolute/path/to/database_nifti"
export CC_OUTPUT_ROOT="/absolute/path/to/outputs"
```

The project also supports these settings in `config.py`:

- `CAMUS_DATA_DIR`
- `ECHONET_DATA_DIR`
- `PRETRAINED_SWIN`
- `BERT_PATH`
- `CC_OUTPUT_ROOT`

## Data protocol

The codebase uses a patient-aware protocol for CAMUS.

- Train/val/test patients are defined at the patient level, not the image level.
- Prompt sets are separated into canonical and held-out banks to reduce phrase leakage.
- Split and prompt logic are enforced in the `data/` package and validated by the core test suite.

## Validation

Run the project regression suite before trusting a change:

```bash
python tests/test_core.py
```

This checks the important protocol and model contract invariants, including:

- split disjointness
- prompt bank separation
- contrastive pooling and loss behavior
- decoder gate-zero equivalence
- metric sanity checks

## Training

The recommended workflow is to define a preset in `config.py` and then run the relevant training script:

```bash
python train_camus.py
```

or

```bash
python train_camus_contrastive.py
```

For a reproducible paper-style ablation sweep, use the built-in runner:

```bash
python run_ablation_suite.py --run
```

and to collect a summary of logged results:

```bash
python report_experiments.py
```

The repository includes experiment preset definitions for baseline and contrastive variants in `config.PRESETS`.

## Citation and reuse

This project is intended for research reproduction and extension. If you use it in a study, please retain the project attribution and document any modifications made to the training protocol, model configuration, or data splits.

## Notes

- Keep all file paths configurable and avoid local hard-coded absolute paths.
- Do not mix train/val/test data by patient ID.
- Use validation split selection and reserve the test split for final reporting.
- Any experimental changes should be validated against the regression suite before training runs are treated as publication evidence.
