# CardioContrast

CardioContrast is a language-guided CAMUS echocardiographic segmentation research pipeline. The active model combines a multimodal Swin backbone, a three-stage gated decoder cross-attention module, and a same-image/different-structure contrastive objective.

## Setup

Install a PyTorch build appropriate for the target CPU/CUDA system, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Configure paths with environment variables; no machine-specific paths are embedded in the training workflow:

```bash
CAMUS_DATA_DIR=/path/to/database_nifti
PRETRAINED_SWIN=/path/to/swin_checkpoint.pth
BERT_PATH=bert-base-uncased
CC_OUTPUT_ROOT=/path/to/outputs
```

On Windows PowerShell, use `$env:CAMUS_DATA_DIR = "..."` syntax. If a CAMUS NIfTI header omits spatial units, verify the source units before setting `CC_SPACING_UNIT` to `mm`, `meter`, or `micron`; physical distance metrics intentionally reject unknown units.

## Data protocol

The dataset uses official `database_split/subgroup_{training,validation,testing}.txt` files when available. Otherwise the fallback is patients 001–400 for train, 401–450 for validation, and 451–500 for test. Training and checkpoint selection use train and validation only; test is reserved for evaluation and baseline scoring.

Each image is represented by three contiguous structure prompts. All experiments use the same deterministic grouped batch sampler, which keeps each image's three structures together. Training augmentation is synchronized across prompts and seeded by `(seed, epoch, image_idx)`. Canonical, paraphrase-training, and held-out prompt banks are kept separate.

## Presets

| Preset | Decoder cross-attention | Contrastive weight | Text/prompt/pooling change |
| --- | --- | ---: | --- |
| `exp1_baseline` | Off | 0.0 | Standard BERT-conditioned backbone |
| `exp2_decoder_ca` | On | 0.0 | Decoder cross-attention |
| `exp3_contrastive` | Off | 0.1 | Contrastive objective |
| `exp4_cardiocontrast` | On | 0.1 | Full method |
| `exp5_class_embedding` | On | 0.1 | Learned class-token encoder |
| `exp6_paraphrase` | On | 0.1 | Training paraphrase bank |
| `exp7_union_pool` | On | 0.1 | Union-mask contrastive pooling |

Explicit CLI flags override preset values.

## Smoke and training

Run the CPU regression suite:

```bash
python tests/test_core.py
```

Create a disposable CAMUS-style NIfTI fixture and run the CPU smoke path without remote BERT assets:

```bash
python tests/make_synthetic_camus.py --output-dir /tmp/cardio-smoke
python train_cardiocontrast.py --preset exp5_class_embedding --epochs 1 --max_images 12 --swin_type tiny --img_size 192 --window_size 6 --device cpu --data_dir /tmp/cardio-smoke/database_nifti --output_root /tmp/cardio-smoke/runs
python evaluate.py --run_dir /tmp/cardio-smoke/runs/exp5_class_embedding_seed42 --split val --device cpu
```

For PowerShell, replace `/tmp/cardio-smoke` with a writable Windows path. This checks data loading, forward/backward, checkpoint writing/reloading, validation, and reporting, but uses random Swin initialization and makes no performance claim. Real experiments should use the configured pretrained weights. Training writes `<CC_OUTPUT_ROOT>/<exp_name>_seed<seed>/args.json`, `log.csv`, `best.pth`, and `last.pth`. Resume with `--resume <run_dir>/last.pth`.

Inspect the seven experiment commands without launching runs:

```bash
python run_ablation_suite.py --dry-run
```

The recommended experimental order is:

1. Select the contrastive weight from `{0.1, 0.3, 1.0}` using validation only.
2. Run `exp1`–`exp5` with seeds 42, 43, and 44.
3. Run `exp6` and `exp7` with the preselected weight and seeds.
4. Export and train the nnU-Net baseline, then evaluate its multi-label predictions.
5. Evaluate SAM/MedSAM prompt baselines and run EchoNet-Dynamic LV evaluation.
6. Run paired statistics and generate paper tables.

For a lambda sweep, use a separate `CC_OUTPUT_ROOT` per weight so runs cannot overwrite each other, evaluate each candidate with `evaluate.py --split val`, then compare the validation summaries with `compare_runs.py --split val` before freezing the selected weight.

## Final evaluation

Choose a checkpoint using validation only, then evaluate it once on test:

```bash
python evaluate.py --run_dir outputs/cardiocontrast/exp4_cardiocontrast_seed42 --ckpt best.pth --split test --prompt_set canonical
```

Outputs are written under `eval_<split>_<prompt_set>/`: `per_sample.csv`, `per_patient.csv`, `clinical.csv`, and `summary.json`. Metrics are computed at native resolution; surface distances use millimeters from NIfTI spacing. Bootstrap intervals resample patients, not correlated frames. `--save_preds` writes compressed per-sample masks and probabilities. Do not use test results for model, prompt, or hyperparameter selection.

## Baselines and analysis

Export the patient-disjoint CAMUS files for nnU-Net v2:

```bash
python tools/export_nnunet.py --output_root /path/to/nnUNet_raw --dataset_id 501
nnUNetv2_plan_and_preprocess -d 501 -c 2d
nnUNetv2_train 501 2d 0
nnUNetv2_predict -i /path/to/nnUNet_raw/Dataset501_CardioContrast/imagesTs -o /path/to/nnunet_predictions -d 501 -c 2d -f 0
python tools/eval_nnunet.py --pred_dir /path/to/nnunet_predictions
```

SAM requires the optional `segment-anything` package and a compatible checkpoint. MedSAM is evaluated with box prompts only; standard SAM also supports point prompts:

```bash
python baselines/sam_prompt_eval.py --weights /path/to/sam_checkpoint.pth --variant sam --prompt_type box
python baselines/sam_prompt_eval.py --weights /path/to/medsam_checkpoint.pth --variant medsam --prompt_type box
```

EchoNet-Dynamic requires its separately downloaded videos and CSV files:

```bash
python evaluate_echonet.py --run_dir outputs/cardiocontrast/exp4_cardiocontrast_seed42 --orientation_check
```

It reports only LV prompt Dice and HD95 in pixels; overlays for all three prompts are qualitative. For patient-paired comparisons and seed-summary tables:

```bash
python compare_runs.py --a outputs/cardiocontrast/exp1_baseline_seed* --b outputs/cardiocontrast/exp4_cardiocontrast_seed* --min_patients 10
python tools/make_table.py --run "Baseline=outputs/cardiocontrast/exp1_baseline_seed*" --run "CardioContrast=outputs/cardiocontrast/exp4_cardiocontrast_seed*"
```

The nnU-Net and SAM packages/checkpoints, CAMUS/EchoNet datasets, CUDA, and the final per-run predictions are external prerequisites; scripts cannot establish baseline performance without them.

## Pre-reporting gates

- Run `python tests/test_core.py` and `python -m pip install pyflakes` followed by `python -m pyflakes` over the active Python files.
- Complete the synthetic CAMUS-style NIfTI CPU smoke run, then evaluate its validation split and run the comparison tooling.
- Verify the GT-mask Simpson EF sanity correlation against CAMUS CFG EF is approximately `r >= 0.9` before making clinical claims.
- Record three seeds, patient-level confidence intervals, paired Wilcoxon tests, and Holm-corrected p-values.
- Confirm the held-out test cohort has been evaluated once only.

No benchmark numbers are asserted by this repository alone. Dataset-backed, multi-seed, external-baseline, and EchoNet results must be run and archived before submission.