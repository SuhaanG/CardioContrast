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
- `evaluate_camus.py` — final held-out test evaluation and paper result export
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
- `CC_SPACING_UNIT` — only needed when NIfTI headers do not declare spatial units; set this only after verifying the source data units (`mm`, `meter`, or `micron`)

## Data protocol

The codebase uses a patient-aware protocol for CAMUS.

- Official `database_split` subgroup files are used when present. Otherwise, the explicit fallback is patients 001–400 for training, 401–450 for validation, and 451–500 for test.
- Training entry points construct separate train and validation datasets from this shared split policy; the test cohort is not used for model selection.
- Prompt sets are separated into canonical and held-out banks to reduce phrase leakage.
- Split and prompt logic are enforced in the `data/` package and validated by the core test suite.

The ablation runner supports the four implemented conditions (`exp1_baseline`, `exp2_decoder_ca`, `exp3_contrastive`, and `exp4_cardiocontrast`). Other entries in `config.PRESETS` are experimental ideas and are rejected by the runner until their behavior is implemented end to end.

## Validation

Run the project regression suite before trusting a change:

```bash
python tests/test_core.py
```

This checks CPU-level protocol and model contract invariants, including:

- split disjointness
- dataset membership at patient split boundaries
- prompt bank separation
- tokenizer-free baseline and contrastive-only paths
- safe experiment manifest contents and supported preset validation
- contrastive pooling and loss behavior
- decoder gate-zero equivalence
- metric sanity checks

The suite does not establish training reproducibility, CUDA compatibility, or benchmark performance. Those require a complete CAMUS dataset and the configured training environment; no performance claim should be made without recorded runs and final test-set evaluation.

## Training

Set the data, pretrained-weight, and output paths through environment variables before launching a run. For the baseline:

```bash
python train_camus.py
```

or

```bash
python train_camus_contrastive.py
```

Inspect the four implemented ablations without starting training:

```bash
python run_ablation_suite.py --dry-run
```

To execute the sweep:

```bash
python run_ablation_suite.py --run
```

and to collect a summary of logged results:

```bash
python report_experiments.py
```

The runner records the seed, runtime configuration, selected non-secret environment settings, and split policy in JSON manifests under `CC_OUTPUT_ROOT/paper`. After selecting a checkpoint using validation only, run final held-out evaluation:

```bash
python evaluate_camus.py --experiment exp4_cardiocontrast --checkpoint /path/to/selected_checkpoint.pth
```

The evaluator always uses the test split and writes frame-level and patient-level CSV tables plus a JSON summary with patient-clustered bootstrap confidence intervals. Its metrics use original-resolution masks and NIfTI in-plane spacing converted to millimeters. If headers have unknown units, evaluation stops unless `CC_SPACING_UNIT` is explicitly set. Do not use these test results for further model or hyperparameter selection.

## Citation and reuse

This project is intended for research reproduction and extension. If you use it in a study, please retain the project attribution and document any modifications made to the training protocol, model configuration, or data splits.

## Notes

- Keep all file paths configurable and avoid local hard-coded absolute paths.
- Do not mix train/val/test data by patient ID.
- Use validation split selection and reserve the test split for final reporting.
- Any experimental changes should be validated against the regression suite before training runs are treated as publication evidence.
