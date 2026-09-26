import os
import subprocess

import torch

CAMUS_DATA_DIR = os.environ.get("CAMUS_DATA_DIR", os.path.join("data", "CAMUS"))
ECHONET_DATA_DIR = os.environ.get("ECHONET_DATA_DIR", os.path.join("data", "EchoNet"))
PRETRAINED_SWIN = os.environ.get("PRETRAINED_SWIN", "")
BERT_PATH = os.environ.get("BERT_PATH", "bert-base-uncased")
CC_OUTPUT_ROOT = os.environ.get("CC_OUTPUT_ROOT", os.path.join("outputs", "cardiocontrast"))

DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

DEFAULTS = {
    "seed": 42,
    "epochs": 40,
    "img_size": 352,
    "batch_size": 3,
    "grad_accum_steps": 4,
    "learning_rate": 5e-5,
    "weight_decay": 1e-2,
    "swin_type": "base",
    "window_size": 7,
    "decode_with_lang": True,
    "contrastive_weight": 0.1,
    "contrastive_tau": 0.07,
    "bert_trainable_layers": 10,
    "embed_tokens": 8,
    "images_per_batch": 2,
}

PRESETS = {
    "exp1_baseline": {"decode_with_lang": False, "contrastive_weight": 0.0, "exp_name": "exp1_baseline"},
    "exp2_decoder_ca": {"decode_with_lang": True, "contrastive_weight": 0.0, "exp_name": "exp2_decoder_ca"},
    "exp3_contrastive": {"decode_with_lang": False, "contrastive_weight": 0.1, "exp_name": "exp3_contrastive"},
    "exp4_cardiocontrast": {"decode_with_lang": True, "contrastive_weight": 0.1, "exp_name": "exp4_cardiocontrast"},
    "exp5_class_embedding": {"decode_with_lang": False, "contrastive_weight": 0.1, "text_encoder": "embedding", "exp_name": "exp5_class_embedding"},
    "exp6_paraphrase": {"decode_with_lang": True, "contrastive_weight": 0.1, "prompt_mode": "paraphrase", "exp_name": "exp6_paraphrase"},
    "exp7_union_pool": {"decode_with_lang": True, "contrastive_weight": 0.1, "pool_region": "union", "exp_name": "exp7_union_pool"},
}

SEED = int(os.environ.get("CC_SEED", DEFAULTS["seed"]))
EPOCHS = int(os.environ.get("CC_EPOCHS", DEFAULTS["epochs"]))
IMG_SIZE = int(os.environ.get("CC_IMG_SIZE", DEFAULTS["img_size"]))
BATCH_SIZE = int(os.environ.get("CC_BATCH_SIZE", DEFAULTS["batch_size"]))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("CC_GRAD_ACCUM_STEPS", DEFAULTS["grad_accum_steps"]))
LR = float(os.environ.get("CC_LR", DEFAULTS["learning_rate"]))
WEIGHT_DECAY = float(os.environ.get("CC_WEIGHT_DECAY", DEFAULTS["weight_decay"]))
SWIN_TYPE = os.environ.get("CC_SWIN_TYPE", DEFAULTS["swin_type"])
WINDOW_SIZE = int(os.environ.get("CC_WINDOW_SIZE", DEFAULTS["window_size"]))
if WINDOW_SIZE not in {7, 12}:
    raise ValueError("CC_WINDOW_SIZE must be either 7 or 12")
SPACING_UNIT = os.environ.get("CC_SPACING_UNIT", "").strip().lower()
DECODE_WITH_LANG = bool(int(os.environ.get("CC_DECODE_WITH_LANG", int(DEFAULTS["decode_with_lang"]))))
CONTRASTIVE_WEIGHT = float(os.environ.get("CC_CONTRASTIVE_WEIGHT", DEFAULTS["contrastive_weight"]))
CONTRASTIVE_TAU = float(os.environ.get("CC_CONTRASTIVE_TAU", DEFAULTS["contrastive_tau"]))
BERT_TRAINABLE_LAYERS = int(os.environ.get("CC_BERT_TRAINABLE_LAYERS", DEFAULTS["bert_trainable_layers"]))
EMBED_TOKENS = int(os.environ.get("CC_EMBED_TOKENS", DEFAULTS["embed_tokens"]))
IMAGES_PER_BATCH = int(os.environ.get("CC_IMAGES_PER_BATCH", DEFAULTS["images_per_batch"]))
CHECKPOINT_DIR = os.environ.get("CC_CHECKPOINT_DIR", os.path.join(CC_OUTPUT_ROOT, "checkpoints"))
GPU_IDS = []
if os.environ.get("CUDA_VISIBLE_DEVICES"):
    GPU_IDS = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]
if not GPU_IDS:
    GPU_IDS = [0] if torch.cuda.is_available() else []


def apply_preset(name: str):
    global DECODE_WITH_LANG, CONTRASTIVE_WEIGHT
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {sorted(PRESETS)}")
    preset = PRESETS[name]
    for key, value in preset.items():
        if key == "decode_with_lang":
            DECODE_WITH_LANG = bool(value)
        elif key == "contrastive_weight":
            CONTRASTIVE_WEIGHT = float(value)
        elif key == "exp_name":
            continue
        else:
            globals()[key.upper()] = value
    return preset


def describe_runtime():
    return {
        "seed": SEED,
        "epochs": EPOCHS,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRADIENT_ACCUMULATION_STEPS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "swin_type": SWIN_TYPE,
        "window_size": WINDOW_SIZE,
        "window12": WINDOW_SIZE == 12 or "window12" in PRETRAINED_SWIN.lower(),
        "pretrained_swin": PRETRAINED_SWIN,
        "spacing_unit": SPACING_UNIT or None,
        "bert_path": BERT_PATH,
        "bert_trainable_layers": BERT_TRAINABLE_LAYERS,
        "decode_with_lang": DECODE_WITH_LANG,
        "contrastive_weight": CONTRASTIVE_WEIGHT,
        "contrastive_tau": CONTRASTIVE_TAU,
        "checkpoint_dir": CHECKPOINT_DIR,
        "camus_data_dir": CAMUS_DATA_DIR,
        "device": DEVICE,
        "git_hash": git_hash(),
    }


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def initialize_environment():
    os.makedirs(CC_OUTPUT_ROOT, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return {
        "CAMUS_DATA_DIR": CAMUS_DATA_DIR,
        "ECHONET_DATA_DIR": ECHONET_DATA_DIR,
        "PRETRAINED_SWIN": PRETRAINED_SWIN,
        "BERT_PATH": BERT_PATH,
        "CC_OUTPUT_ROOT": CC_OUTPUT_ROOT,
        "CHECKPOINT_DIR": CHECKPOINT_DIR,
        "device": DEVICE,
        "git_hash": git_hash(),
    }


if __name__ == "__main__":
    print(initialize_environment())
