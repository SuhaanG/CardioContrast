"""
config.py — Paths and defaults. Every value here can be overridden from the
command line of train_cardiocontrast.py / evaluate.py, so nobody needs to edit
this file between experiments (the old workflow of hand-editing flags before
each run is how ablation runs get mislabeled).
"""

import os

# ---------------------------------------------------------------- paths
CAMUS_DATA_DIR = os.environ.get("CAMUS_DATA_DIR",
                                "/content/CAMUS_public/CAMUS_public/database_nifti")
ECHONET_DATA_DIR = os.environ.get("ECHONET_DATA_DIR", "/content/EchoNet-Dynamic")
PRETRAINED_SWIN = os.environ.get(
    "PRETRAINED_SWIN",
    "/content/CardioContrast/pretrained_weights/swin_base_patch4_window12_384_22k.pth")
BERT_PATH = os.environ.get("BERT_PATH", "bert-base-uncased")   # or a local folder
OUTPUT_ROOT = os.environ.get("CC_OUTPUT_ROOT", "/content/CardioContrast/experiments")

# ---------------------------------------------------------------- defaults
IMG_SIZE = 352
SWIN_TYPE = "base"
WINDOW_SIZE = 12

IMAGES_PER_BATCH = 2          # batch = IMAGES_PER_BATCH x 3 prompts = 6 samples
GRAD_ACCUM_STEPS = 2          # effective batch = 12 samples (same as before)
LR = 5e-5
WEIGHT_DECAY = 1e-2
EPOCHS = 40
WARMUP_STEPS = 500
SEED = 42

CONTRASTIVE_TAU = 0.07
CONTRASTIVE_WEIGHT = 0.1
BERT_TRAINABLE_LAYERS = 10

# Ablation presets. Every preset uses the SAME sampler, loss, augmentation and
# schedule; only the listed fields differ.
# Contrastive loss (v2, default for every preset below unless stated): applied to
# the decoder features pooled over the union of the three structures, with NO
# projection head (pool_region="union", proj_head="none").
PRESETS = {
    # core 2x2 ablation
    "exp1_baseline":        dict(decode_with_lang=0, contrastive_weight=0.0),
    "exp2_decoder_ca":      dict(decode_with_lang=1, contrastive_weight=0.0),
    "exp3_contrastive":     dict(decode_with_lang=0, contrastive_weight=CONTRASTIVE_WEIGHT),
    "exp4_cardiocontrast":  dict(decode_with_lang=1, contrastive_weight=CONTRASTIVE_WEIGHT),
    # does language matter? (same as exp4 but the text pathway is a class lookup table)
    "exp5_class_embedding": dict(decode_with_lang=1, contrastive_weight=CONTRASTIVE_WEIGHT,
                                 text_encoder="embedding"),
    # language generalisation: train on paraphrases, test on held-out paraphrases
    "exp6_paraphrase":      dict(decode_with_lang=1, contrastive_weight=CONTRASTIVE_WEIGHT,
                                 prompt_mode="paraphrase"),
    # v1 contrastive formulation (projection head + predicted-mask pooling).
    # Ablation showing why v2 is needed: the v1 loss collapses to ~0 early.
    "exp7_v1_projhead":     dict(decode_with_lang=1, contrastive_weight=CONTRASTIVE_WEIGHT,
                                 pool_region="pred", proj_head="mlp"),
}
