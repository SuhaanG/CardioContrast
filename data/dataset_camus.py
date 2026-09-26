"""
data/dataset_camus.py — CAMUS dataset for prompt-conditioned binary segmentation.

One sample = (image, one structure prompt) -> binary mask of that structure.
Every image yields one sample per structure (3 samples), and the samples of an
image are stored contiguously so the batch sampler can group them.

Changes vs. the previous version:
  * Split is by patient via data/splits.py (no test-set model selection).
  * Returns case metadata (patient, view, phase, native size, pixel spacing,
    image quality, reference EF) so evaluation can be done at NATIVE resolution
    in millimetres and stratified the way CAMUS papers report results.
  * Joint image/mask augmentation for training. The augmentation for an image is
    seeded by (seed, epoch, image_idx), so the 3 prompts of the same image see
    the SAME geometry -- required for the contrastive term to compare like with
    like.
  * Prompts are pre-tokenized once instead of per sample.
  * Optional paraphrase prompts (train) and held-out paraphrases (eval).
  * Returns `union` (all three structures) for the "union" contrastive pooling
    variant and `class_id` for the learned-embedding language ablation.
"""

import glob
import os
import re

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from data.prompts import CANONICAL, HELDOUT, STRUCTURE_IDS, TRAIN_BANK
from data.splits import get_split_patients

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_NAME_RE = re.compile(r"(patient\d+)_(2CH|4CH)_(ED|ES)_gt\.nii(\.gz)?$")


def parse_cfg(path):
    """Parse a CAMUS Info_*.cfg file ('key: value' per line)."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def load_nifti_2d(path):
    img = nib.load(path)
    arr = np.squeeze(np.asarray(img.dataobj))
    spacing = tuple(float(z) for z in img.header.get_zooms()[:2])
    return arr, spacing



def index_images(data_dir, split, verbose=True):
    """List the ED/ES images of a split with metadata (no tokenizer needed)."""
    patients = get_split_patients(data_dir, split, verbose=verbose)
    images = []
    for gt_path in sorted(glob.glob(os.path.join(data_dir, "patient*", "*_gt.nii*"))):
        m = _NAME_RE.search(os.path.basename(gt_path))
        if m is None:           # skips half_sequence and anything unexpected
            continue
        patient, view, phase = m.group(1), m.group(2), m.group(3)
        if patient not in patients:
            continue
        img_path = gt_path.replace("_gt.nii", ".nii")
        if not os.path.exists(img_path):
            continue
        cfg = parse_cfg(os.path.join(os.path.dirname(gt_path), "Info_{}.cfg".format(view)))
        try:
            ref_ef = float(cfg["EF"]) if "EF" in cfg else float("nan")
        except ValueError:
            ref_ef = float("nan")
        images.append({
            "image_path": img_path, "mask_path": gt_path,
            "patient": patient, "view": view, "phase": phase,
            "case": "{}_{}_{}".format(patient, view, phase),
            "quality": cfg.get("ImageQuality", "NA"),
            "ref_ef": ref_ef,
        })
    return images


class CAMUSDataset(data.Dataset):
    def __init__(self, data_dir, split, img_size=352, tokenizer_path="bert-base-uncased",
                 max_tokens=20, augment=False, prompt_mode="fixed",
                 eval_prompt_set="canonical", seed=42, max_images=None):
        """
        Args:
            split:           'train' | 'val' | 'test'
            prompt_mode:     'fixed' (canonical prompt) or 'paraphrase'
                             (random TRAIN_BANK prompt per sample; train only)
            eval_prompt_set: which prompts to enumerate when not paraphrasing:
                             'canonical' -> 1 prompt per structure
                             'heldout'   -> every HELDOUT paraphrase
        """
        assert prompt_mode in ("fixed", "paraphrase")
        assert eval_prompt_set in ("canonical", "heldout")
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.max_tokens = max_tokens
        self.augment = augment
        self.prompt_mode = prompt_mode
        self.seed = seed
        self.epoch = 0

        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        self.images = index_images(data_dir, split)
        if len(self.images) == 0:
            raise ValueError("No CAMUS images found for split '{}' in {}".format(split, data_dir))
        if max_images is not None:          # smoke tests only
            self.images = self.images[:max_images]

        # Prompt table per structure.
        if eval_prompt_set == "heldout":
            self.prompt_table = {s: list(HELDOUT[s]) for s in STRUCTURE_IDS}
        else:
            self.prompt_table = {s: [CANONICAL[s]] for s in STRUCTURE_IDS}

        # samples: (image_idx, structure_id, prompt_k). Contiguous per image.
        self.samples = []
        for i in range(len(self.images)):
            for s in STRUCTURE_IDS:
                for k in range(len(self.prompt_table[s])):
                    self.samples.append((i, s, k))

        all_prompts = set(CANONICAL.values())
        for s in STRUCTURE_IDS:
            all_prompts |= set(TRAIN_BANK[s]) | set(HELDOUT[s])
        self._tok = {p: self._tokenize(p) for p in all_prompts}

    def _tokenize(self, sentence):
        ids = self.tokenizer.encode(text=sentence, add_special_tokens=True)[: self.max_tokens]
        input_ids = torch.zeros(self.max_tokens, dtype=torch.long)
        attn = torch.zeros(self.max_tokens, dtype=torch.long)
        input_ids[: len(ids)] = torch.tensor(ids)
        attn[: len(ids)] = 1
        return input_ids, attn

    # ------------------------------------------------------------- utilities
    def set_epoch(self, epoch):
        """Call once per epoch (before building the iterator) so augmentation varies."""
        self.epoch = epoch

    def image_groups(self):
        """Map image_idx -> list of sample indices (used by the batch sampler)."""
        groups = {}
        for n, (i, _, _) in enumerate(self.samples):
            groups.setdefault(i, []).append(n)
        return groups

    def load_full_res(self, image_idx):
        """Native-resolution multi-label GT mask and pixel spacing (mm)."""
        mask, spacing = load_nifti_2d(self.images[image_idx]["mask_path"])
        return mask.astype(np.uint8), spacing

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------ transforms
    def _augment(self, img, mask, rng):
        """img: (1,S,S) float in [0,1]; mask: (1,S,S) uint8 multi-label."""
        angle = float(rng.uniform(-10, 10))
        scale = float(rng.uniform(0.9, 1.1))
        tx = int(round(rng.uniform(-0.05, 0.05) * self.img_size))
        ty = int(round(rng.uniform(-0.05, 0.05) * self.img_size))
        img = TF.affine(img, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0],
                        interpolation=InterpolationMode.BILINEAR)
        mask = TF.affine(mask, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0],
                         interpolation=InterpolationMode.NEAREST)
        gamma = float(rng.uniform(0.8, 1.25))
        contrast = float(rng.uniform(0.85, 1.15))
        brightness = float(rng.uniform(-0.08, 0.08))
        img = img.clamp(0, 1) ** gamma
        img = ((img - img.mean()) * contrast + img.mean() + brightness).clamp(0, 1)
        return img, mask

    def __getitem__(self, index):
        image_idx, structure, prompt_k = self.samples[index]
        rec = self.images[image_idx]

        image, spacing = load_nifti_2d(rec["image_path"])
        mask, _ = load_nifti_2d(rec["mask_path"])
        orig_h, orig_w = image.shape

        image = image.astype(np.float32)
        image = image / image.max() if image.max() > 0 else image
        img_t = torch.from_numpy(image)[None, None]                      # (1,1,H,W)
        img_t = F.interpolate(img_t, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False, antialias=True)[0]
        mask_t = torch.from_numpy(mask.astype(np.uint8))[None, None].float()
        mask_t = F.interpolate(mask_t, size=(self.img_size, self.img_size),
                               mode="nearest")[0].to(torch.uint8)

        # Same augmentation for all prompts of an image within an epoch.
        rng = np.random.default_rng([self.seed, self.epoch, image_idx])
        if self.augment:
            img_t, mask_t = self._augment(img_t, mask_t, rng)

        img_t = (img_t.repeat(3, 1, 1) - IMAGENET_MEAN) / IMAGENET_STD
        mask_t = mask_t[0].long()
        target = (mask_t == structure).long()
        union = (mask_t > 0).float()

        if self.prompt_mode == "paraphrase" and self.split == "train":
            # Independent draw per (image, structure) so prompts vary within an image.
            prng = np.random.default_rng([self.seed, self.epoch, image_idx, structure])
            prompt = TRAIN_BANK[structure][int(prng.integers(len(TRAIN_BANK[structure])))]
        else:
            prompt = self.prompt_table[structure][prompt_k]
        input_ids, attn = self._tok[prompt]

        return {
            "image": img_t,
            "target": target,
            "union": union,
            "input_ids": input_ids,
            "attn_mask": attn,
            "class_id": torch.tensor(structure - 1, dtype=torch.long),
            "structure": torch.tensor(structure, dtype=torch.long),
            "image_idx": torch.tensor(image_idx, dtype=torch.long),
            "prompt": prompt,
            "patient": rec["patient"],
            "view": rec["view"],
            "phase": rec["phase"],
            "quality": rec["quality"],
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "spacing": torch.tensor(spacing, dtype=torch.float32),
        }
