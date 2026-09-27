# CardioContrast

Language-guided echocardiographic segmentation (CAMUS: LV endocardium, myocardium, left atrium), built on LAVT, with two additions:

1. **Multi-stage decoder cross-attention**: language is re-injected at every decoder stage through a gated residual.
2. **Contrastive anatomical repulsion loss**: decoder features for different structure prompts on the same image, pooled over the same pixels (union of the three structures), are pushed apart. The loss acts on the decoder features directly (no projection head); see `lib/contrastive.py` for why the v1 projection-head version was dropped.

Out-of-distribution test: EchoNet-Dynamic (never used for training; LV labels only).

---

## What changed in this revision (and why)

| Area | Old behaviour | Now | Why it matters |
|---|---|---|---|
| Split | Patients 451–500 were used as "val" **and** for checkpoint selection | Patient-level train / val / test (`data/splits.py`); uses the official `database_split/*.txt` if present, otherwise 1–400 / 401–450 / 451–500 | The old numbers were picked on the test set, so they were optimistically biased |
| Metric | Overall IoU at 352×352 | Dice, IoU, HD95, HD, MAD **in mm at native resolution**, per structure × ED/ES × image quality, plus cross-structure **leakage** | Matches CAMUS/TMI reporting; overall IoU hides the myocardium |
| Clinical | none | EDV / ESV / EF (biplane Simpson) with r, bias, MAE, limits of agreement | The paper's motivation is LVEF |
| Ablation | Baseline used random batches; the other runs used grouped batches | Same sampler, loss, augmentation and schedule for every run | Otherwise Exp1 vs Exp2–4 is confounded |
| Loss | CE with weights 0.59/3.41 computed from the union of all structures | CE + soft Dice | The weights did not match the per-prompt binary task (~5% foreground, not ~15%) |
| Contrastive | Sum of softplus(sim/τ), τ = 0.07 (≈27 per anchor at init); pooling weights not detached; BatchNorm on 3 samples | τ·softplus(sim/τ) averaged over negatives; detached weights; LayerNorm; pooling region `pred` / `gt` / `union` | The loss was ~2× the segmentation loss at init; the mask could be reshaped to satisfy the loss; BN on 3 vectors is noise |
| Decoder CA | Double Q/K/V projection; features replaced by `out_proj(LN(S+Z))` | One MHA with kdim/vdim; `S + tanh(g)·W_out(MHA(LN(S),L,L))`, g=0 at init (higher LR for g) | Starts exactly as the baseline decoder; gate values show how much language each stage uses |
| Language ablation | none | `--text_encoder embedding` (learned class lookup) and paraphrase training / held-out paraphrase testing | Tests whether *language*, rather than a class index, does the work |
| Baselines | none | nnU-Net export/eval with the same split; SAM / MedSAM box and point prompts | The first two things a reviewer will ask for |
| Stats | single run | `compare_runs.py`: per-patient paired Wilcoxon + bootstrap CI + Holm; `tools/make_table.py`: mean ± std over seeds | |
| Engineering | Hand-edit `config.py` per run; DataParallel with batch 3; `gc.collect()` every step; dead RefCOCO code | CLI presets, bf16 autocast, resume, optimizer coverage check, CPU unit tests | |

**Existing checkpoints are incompatible with this revision.** Retrain everything; no old number is reportable.

---

## Setup

```bash
git clone https://github.com/SuhaanG/CardioContrast.git && cd CardioContrast
pip install -r requirements.txt          # install torch/torchvision for your CUDA first
mkdir -p pretrained_weights
wget -P pretrained_weights https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
export CAMUS_DATA_DIR=/content/CAMUS_public/CAMUS_public/database_nifti   # or edit config.py
python tests/test_core.py                 # must print ALL TESTS PASSED
```

Check the split printout at the start of training. It says whether the official `database_split` files were found.

## Smoke test (≈2 minutes, run before any full job)

```bash
python train_cardiocontrast.py --preset exp4_cardiocontrast --epochs 1 --max_images 12 --exp_name smoke
```

## Experiments

Every run: `python train_cardiocontrast.py --preset <name> --seed <s>` → `experiments/<name>_seed<s>/`, then `python evaluate.py --run_dir experiments/<name>_seed<s>`.

| Preset | Decoder CA | Contrastive | Text | Purpose |
|---|---|---|---|---|
| `exp1_baseline` | off | off | BERT | LAVT baseline |
| `exp2_decoder_ca` | on | off | BERT | Contribution 1 alone |
| `exp3_contrastive` | off | on | BERT | Contribution 2 alone |
| `exp4_cardiocontrast` | on | on | BERT | Full method |
| `exp5_class_embedding` | on | on | learned class table | **Does language matter?** |
| `exp6_paraphrase` | on | on | BERT, paraphrase prompts | Language generalisation (evaluate with `--prompt_set heldout`) |
| `exp7_v1_projhead` | on | on (v1: projection head + predicted-mask pooling) | BERT | Ablation: the original loss formulation, which collapses to ~0 early in training |

Order:

1. **λ on validation first:** run `exp4_cardiocontrast` with `--contrastive_weight 0.1`, `0.3` and `1.0` (`--exp_name lam0.1` etc.) and pick the best by `val_mean_dice` in `log.csv`. Use that λ for exp3–exp7 (`--contrastive_weight X`). Never pick λ from test results.
2. Run exp1–exp5 with seeds 42, 43, 44.
3. Run exp6 and exp7 (one seed first; add seeds if they matter for the story).
4. Baselines: nnU-Net (`tools/export_nnunet.py`, instructions in the file header) and SAM/MedSAM (`baselines/sam_prompt_eval.py`).
5. EchoNet-Dynamic: `python evaluate_echonet.py --run_dir <run> --orientation_check` first, then the real run.
6. Tables and statistics:

```bash
python tools/make_table.py --run "LAVT=experiments/exp1_baseline_seed*" \
    --run "CardioContrast=experiments/exp4_cardiocontrast_seed*" \
    --run "Class-embedding=experiments/exp5_class_embedding_seed*" \
    --run "nnU-Net 2D=experiments/nnunet_2d_fold0"
python compare_runs.py --a experiments/exp1_baseline_seed42/eval_test_canonical,experiments/exp1_baseline_seed43/eval_test_canonical,experiments/exp1_baseline_seed44/eval_test_canonical \
                       --b experiments/exp4_cardiocontrast_seed42/eval_test_canonical,experiments/exp4_cardiocontrast_seed43/eval_test_canonical,experiments/exp4_cardiocontrast_seed44/eval_test_canonical
```

## Things to check before reporting

- `evaluate.py` prints a **sanity line**: our Simpson EF on GT masks vs the EF stored in the CAMUS cfg files. If r is well below ~0.9, the long-axis estimate in `lib/metrics.py` needs work before any EF number goes in the paper.
- `log.csv` columns `gate_stage*` (how much each decoder stage uses language) and `val_prompt_feature_cos` (head-free cosine between prompt-conditioned features at the same pixels; lower = more separated) are figure material for the paper and are comparable across all runs, including the baseline.
- The test split is used only by `evaluate.py`. Do not re-run test evaluation to choose anything.

## Files

```
train_cardiocontrast.py   training (all presets)
evaluate.py               native-resolution test evaluation + EF
evaluate_echonet.py       EchoNet-Dynamic OOD (LV only)
compare_runs.py           paired statistics between runs
config.py                 paths, defaults, presets
data/                     dataset, split, prompts, sampler
lib/                      Swin+PWAM backbone, decoder, text encoders, contrastive loss, metrics, report
tools/                    nnU-Net export/eval, results table
baselines/                SAM / MedSAM prompt baselines
tests/test_core.py        CPU unit tests
```
