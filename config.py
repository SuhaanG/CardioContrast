import os
import subprocess

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


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def initialize_environment():
    os.makedirs(CC_OUTPUT_ROOT, exist_ok=True)
    return {
        "CAMUS_DATA_DIR": CAMUS_DATA_DIR,
        "ECHONET_DATA_DIR": ECHONET_DATA_DIR,
        "PRETRAINED_SWIN": PRETRAINED_SWIN,
        "BERT_PATH": BERT_PATH,
        "CC_OUTPUT_ROOT": CC_OUTPUT_ROOT,
        "device": DEVICE,
        "git_hash": git_hash(),
    }


if __name__ == "__main__":
    print(initialize_environment())
