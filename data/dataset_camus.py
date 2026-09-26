import glob
import os
import re

import numpy as np
import torch
from PIL import Image
from torch.utils import data

from .prompts import CANONICAL, HELDOUT, STRUCTURES, TRAIN_BANK

FILENAME_RE = re.compile(r"(patient\d+)_(2CH|4CH)_(ED|ES)_gt\.nii(\.gz)?$")


def spacing_to_mm(zooms, unit, assumed_unit=None):
    if len(zooms) < 2 or any(not np.isfinite(value) or value <= 0 for value in zooms[:2]):
        raise ValueError("NIfTI file has invalid in-plane pixel spacing")
    if unit in {None, "", "unknown"}:
        unit = assumed_unit
    unit_to_mm = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
    if unit not in unit_to_mm:
        raise ValueError(f"Unsupported or missing NIfTI spatial unit: {unit!r}")
    factor = unit_to_mm[unit]
    return float(zooms[0] * factor), float(zooms[1] * factor)


def index_images(data_dir, split):
    from .splits import get_split_patients

    allowed = set(get_split_patients(data_dir, split))
    records = []
    for mask_path in sorted(glob.glob(os.path.join(data_dir, "patient*", "*_gt.nii*")) + glob.glob(os.path.join(data_dir, "*_gt.nii*"))):
        if "half_sequence" in mask_path:
            continue
        base = os.path.basename(mask_path)
        m = FILENAME_RE.match(base)
        if m is None:
            continue
        patient, view, phase, _ = m.groups()
        if patient not in allowed:
            continue
        image_candidates = [
            mask_path.replace("_gt.nii.gz", ".nii.gz"),
            mask_path.replace("_gt.nii", ".nii"),
            mask_path.replace("_gt.nii.gz", ".nii"),
        ]
        image_path = next((p for p in image_candidates if os.path.exists(p)), None)
        if image_path is None:
            continue
        info = _read_info_cfg(os.path.join(os.path.dirname(mask_path), f"Info_{view}.cfg"))
        records.append({
            "image_path": image_path,
            "mask_path": mask_path,
            "patient": patient,
            "view": view,
            "phase": phase,
            "case_id": f"{patient}_{view}",
            "quality": info.get("imagequality", "unknown"),
            "ref_ef": _find_ef(info),
        })
    return records


def _read_info_cfg(path):
    values = {}
    if not os.path.isfile(path):
        return values
    with open(path, "r", encoding="utf-8", errors="replace") as source:
        for line in source:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip().lower()] = value.strip()
    return values


def _find_ef(info):
    for key in ("ef", "lvef", "lvef", "lvejectionfraction"):
        value = info.get(key)
        if value:
            match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
            if match:
                return float(match.group())
    return None


def preprocess_grayscale_image(image, img_size):
    image = np.asarray(image, dtype=np.float32)
    max_value = float(np.max(image)) if image.size else 0.0
    if max_value > 0:
        image = image / max_value * 255.0
    image_rgb = np.repeat(image.astype(np.uint8)[:, :, None], 3, axis=2)
    image_pil = Image.fromarray(image_rgb).resize((img_size, img_size), Image.BILINEAR)
    image_arr = np.asarray(image_pil, dtype=np.float32) / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 1, 3)
    normalized = ((image_arr - mean) / std).transpose(2, 0, 1).copy()
    return torch.from_numpy(normalized)


class CAMUSDataset(data.Dataset):
    def __init__(self, data_dir, split="train", img_size=352, prompt_mode="fixed", eval_prompt_set="canonical", seed=42, epoch=0, use_language=True, bert_tokenizer="bert-base-uncased", spacing_unit="", augment=True, max_images=None):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.prompt_mode = prompt_mode
        self.eval_prompt_set = eval_prompt_set
        self.seed = seed
        self.epoch = epoch
        self.use_language = use_language
        self.bert_tokenizer = bert_tokenizer
        self.spacing_unit = spacing_unit
        self.augment = bool(augment and split == "train")
        if prompt_mode not in {"fixed", "paraphrase"}:
            raise ValueError(f"Unknown prompt_mode={prompt_mode!r}")
        if prompt_mode == "paraphrase" and split != "train":
            raise ValueError("Paraphrase prompt mode is only valid for training")
        if eval_prompt_set not in {"canonical", "heldout"}:
            raise ValueError(f"Unknown eval_prompt_set={eval_prompt_set!r}")
        if split == "train" and eval_prompt_set != "canonical":
            raise ValueError("eval_prompt_set is only valid for evaluation splits")
        self.records = index_images(data_dir, split)
        if max_images is not None:
            if int(max_images) < 1:
                raise ValueError("max_images must be positive")
            self.records = self.records[:int(max_images)]
        self._spacing_cache = {}
        self.samples = []
        for i, record in enumerate(self.records):
            for structure in sorted(STRUCTURES):
                prompts = HELDOUT[structure] if split != "train" and eval_prompt_set == "heldout" else (CANONICAL[structure],)
                for prompt_k in range(len(prompts)):
                    self.samples.append({
                        "record": record,
                        "structure": structure,
                        "image_idx": i,
                        "prompt_k": prompt_k,
                    })
        self.tokenizer = None
        self._prompt_tokens = self._pretokenize_prompts()
        self._init_tokenizer()

    def _init_tokenizer(self):
        if not self.use_language:
            return
        from transformers import BertTokenizer

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_tokenizer)
        self._prompt_tokens = self._pretokenize_prompts()

    def _all_prompts(self):
        prompts = set(CANONICAL.values())
        prompts.update(prompt for bank in TRAIN_BANK.values() for prompt in bank)
        prompts.update(prompt for bank in HELDOUT.values() for prompt in bank)
        return prompts

    def _pretokenize_prompts(self):
        if not self.use_language or self.tokenizer is None:
            return {}
        return {
            prompt: self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=32,
            )
            for prompt in self._all_prompts()
        }

    def _encode_prompt(self, prompt):
        if self.tokenizer is None:
            input_ids = torch.zeros(32, dtype=torch.long)
            attn_mask = torch.zeros(32, dtype=torch.long)
            return input_ids, attn_mask
        encoded = self._prompt_tokens[prompt]
        return encoded["input_ids"][0].clone(), encoded["attention_mask"][0].clone()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _load_nifti_2d(self, path):
        import nibabel as nib

        arr = np.asarray(nib.load(path).get_fdata())
        return np.squeeze(arr).astype(np.float32)

    def get_pixel_spacing(self, path, assumed_unit=None):
        import nibabel as nib

        if path in self._spacing_cache:
            return self._spacing_cache[path]
        header = nib.load(path).header
        spacing = spacing_to_mm(
            header.get_zooms(),
            header.get_xyzt_units()[0],
            assumed_unit=assumed_unit or self.spacing_unit,
        )
        self._spacing_cache[path] = spacing
        return spacing

    def _resolve_prompt(self, structure, image_idx, prompt_k=0):
        if self.prompt_mode == "fixed":
            if self.split != "train" and self.eval_prompt_set == "heldout":
                return HELDOUT[structure][prompt_k]
            return CANONICAL[structure]
        if self.prompt_mode == "paraphrase":
            rng = np.random.default_rng([self.seed, self.epoch, image_idx, structure])
            bank = TRAIN_BANK[structure]
            return bank[int(rng.integers(0, len(bank)))]
        raise ValueError(f"Unknown prompt_mode={self.prompt_mode!r}")

    def _augment_image_and_mask(self, image, mask, image_idx):
        rng = np.random.default_rng([self.seed, self.epoch, image_idx])
        height, width = image.shape
        angle = float(rng.uniform(-10.0, 10.0))
        scale = float(rng.uniform(0.9, 1.1))
        translate = (
            int(round(rng.uniform(-0.05, 0.05) * width)),
            int(round(rng.uniform(-0.05, 0.05) * height)),
        )
        image = np.asarray(image, dtype=np.float32)
        maximum = float(image.max()) if image.size else 0.0
        image = image / maximum if maximum > 0 else image
        image_pil = Image.fromarray(image, mode="F")
        mask_pil = Image.fromarray(np.asarray(mask, dtype=np.uint8))
        radians = np.deg2rad(angle)
        cosine = float(np.cos(radians) / scale)
        sine = float(np.sin(radians) / scale)
        center_x, center_y = width / 2.0, height / 2.0
        translate_x, translate_y = translate
        affine = (
            cosine,
            sine,
            center_x - cosine * (center_x + translate_x) - sine * (center_y + translate_y),
            -sine,
            cosine,
            center_y + sine * (center_x + translate_x) - cosine * (center_y + translate_y),
        )
        image_pil = image_pil.transform(
            (width, height), Image.Transform.AFFINE, affine,
            resample=Image.Resampling.BILINEAR, fillcolor=0.0,
        )
        mask_pil = mask_pil.transform(
            (width, height), Image.Transform.AFFINE, affine,
            resample=Image.Resampling.NEAREST, fillcolor=0,
        )

        intensity_rng = np.random.default_rng([self.seed, self.epoch, image_idx, 7919])
        gamma = float(intensity_rng.uniform(0.8, 1.25))
        contrast = float(intensity_rng.uniform(0.85, 1.15))
        brightness = float(intensity_rng.uniform(-0.08, 0.08))
        image_arr = np.asarray(image_pil, dtype=np.float32)
        image_arr = np.power(np.clip(image_arr, 0.0, None), gamma)
        image_mean = float(image_arr.mean())
        image_arr = (image_arr - image_mean) * contrast + image_mean
        image_arr = np.clip(image_arr + brightness, 0.0, None)
        return image_arr, np.asarray(mask_pil, dtype=np.uint8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        record = sample["record"]
        structure = sample["structure"]
        image_idx = sample["image_idx"]
        prompt = self._resolve_prompt(structure, image_idx, sample["prompt_k"])

        image = self._load_nifti_2d(record["image_path"])
        mask = self._load_nifti_2d(record["mask_path"])
        if self.augment:
            image, mask = self._augment_image_and_mask(image, mask, image_idx)
        image_tensor = preprocess_grayscale_image(image, self.img_size)
        foreground = np.zeros_like(mask, dtype=np.uint8)
        foreground[mask == structure] = 1

        target = np.asarray(Image.fromarray(foreground.astype(np.uint8)).resize((self.img_size, self.img_size), Image.NEAREST), dtype=np.float32)
        union = np.asarray(Image.fromarray((mask > 0).astype(np.uint8)).resize((self.img_size, self.img_size), Image.NEAREST), dtype=np.float32)

        input_ids, attn_mask = self._encode_prompt(prompt)
        return {
            "image": image_tensor,
            "target": torch.from_numpy(target),
            "union": torch.from_numpy(union),
            "input_ids": input_ids,
            "attn_mask": attn_mask,
            "class_id": torch.tensor(structure - 1, dtype=torch.long),
            "structure": structure,
            "image_idx": torch.tensor(image_idx, dtype=torch.long),
            "prompt": prompt,
            "patient": record["patient"],
            "case_id": record["case_id"],
            "view": record["view"],
            "phase": record["phase"],
            "quality": record["quality"],
            "ref_ef": float("nan") if record["ref_ef"] is None else record["ref_ef"],
            "orig_size": image.shape,
            "spacing": np.asarray(self.get_pixel_spacing(record["mask_path"]), dtype=np.float32),
        }

    def load_full_res(self, image_idx):
        record = self.records[image_idx]
        mask = self._load_nifti_2d(record["mask_path"])
        spacing = np.asarray(self.get_pixel_spacing(record["mask_path"]), dtype=np.float32)
        return mask, spacing

    def image_groups(self):
        groups = {}
        for idx, sample in enumerate(self.samples):
            groups.setdefault(sample["image_idx"], []).append(idx)
        return groups

    @property
    def max_images(self):
        return len(self.records)


__all__ = ["CAMUSDataset", "index_images"]
