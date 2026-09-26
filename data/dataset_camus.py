import glob
import os
import re

import numpy as np
import torch
from PIL import Image
from torch.utils import data

from .prompts import CANONICAL, HELDOUT, STRUCTURES, TRAIN_BANK

FILENAME_RE = re.compile(r"(patient\d+)_(2CH|4CH)_(ED|ES)_gt\.nii(\.gz)?$")


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
        records.append({"image_path": image_path, "mask_path": mask_path, "patient": patient, "view": view, "phase": phase})
    return records


class CAMUSDataset(data.Dataset):
    def __init__(self, data_dir, split="train", img_size=352, prompt_mode="fixed", eval_prompt_set="canonical", seed=42, epoch=0, use_language=True):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.prompt_mode = prompt_mode
        self.eval_prompt_set = eval_prompt_set
        self.seed = seed
        self.epoch = epoch
        self.use_language = use_language
        self.records = index_images(data_dir, split)
        self.samples = []
        for i, record in enumerate(self.records):
            for structure in sorted(STRUCTURES):
                self.samples.append({"record": record, "structure": structure, "image_idx": i, "prompt": CANONICAL[structure]})
        self.tokenizer = None
        self._prompt_tokens = {}
        self._init_tokenizer()

    def _init_tokenizer(self):
        if not self.use_language:
            return
        try:
            from transformers import BertTokenizer

            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        except Exception:
            self.tokenizer = None

    def _encode_prompt(self, prompt):
        if self.tokenizer is None:
            input_ids = torch.zeros(16, dtype=torch.long)
            attn_mask = torch.zeros(16, dtype=torch.long)
            return input_ids, attn_mask
        encoded = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
        return encoded["input_ids"][0].clone(), encoded["attention_mask"][0].clone()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _load_nifti_2d(self, path):
        import nibabel as nib

        arr = np.asarray(nib.load(path).get_fdata())
        return np.squeeze(arr).astype(np.float32)

    def _resolve_prompt(self, structure, image_idx):
        if self.prompt_mode == "fixed":
            return CANONICAL[structure]
        if self.prompt_mode == "paraphrase":
            rng = np.random.default_rng([self.seed, self.epoch, image_idx, structure])
            bank = TRAIN_BANK[structure]
            return bank[int(rng.integers(0, len(bank)))]
        if self.eval_prompt_set == "heldout":
            bank = HELDOUT[structure]
            return bank[image_idx % len(bank)]
        return CANONICAL[structure]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        record = sample["record"]
        structure = sample["structure"]
        image_idx = sample["image_idx"]
        prompt = self._resolve_prompt(structure, image_idx)

        image = self._load_nifti_2d(record["image_path"])
        mask = self._load_nifti_2d(record["mask_path"])
        foreground = np.zeros_like(mask, dtype=np.uint8)
        foreground[mask == structure] = 1

        max_value = float(np.max(image)) if np.max(image) > 0 else 1.0
        image_norm = image / max_value
        image_pil = Image.fromarray(np.repeat(np.asarray(Image.fromarray(image_norm).resize((self.img_size, self.img_size), Image.BILINEAR))[:, :, None], 3, axis=2))
        image_arr = np.asarray(image_pil).transpose(2, 0, 1).astype(np.float32)
        image_arr = (image_arr - 0.485) / 0.229

        target = np.asarray(Image.fromarray(foreground.astype(np.uint8)).resize((self.img_size, self.img_size), Image.NEAREST), dtype=np.float32)
        union = (mask > 0).astype(np.float32)

        input_ids, attn_mask = self._encode_prompt(prompt)
        return {
            "image": torch.from_numpy(image_arr),
            "target": torch.from_numpy(target),
            "union": torch.from_numpy(union),
            "input_ids": input_ids,
            "attn_mask": attn_mask,
            "class_id": torch.tensor(structure - 1, dtype=torch.long),
            "structure": structure,
            "image_idx": torch.tensor(image_idx, dtype=torch.long),
            "prompt": prompt,
            "patient": record["patient"],
            "view": record["view"],
            "phase": record["phase"],
            "quality": 1.0,
            "orig_size": image.shape,
            "spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        }

    def load_full_res(self, image_idx):
        record = self.records[image_idx]
        mask = self._load_nifti_2d(record["mask_path"])
        spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        return mask, spacing

    def image_groups(self):
        groups = {}
        for idx, sample in enumerate(self.samples):
            groups.setdefault(sample["image_idx"], []).append(idx)
        return groups

    @property
    def max_images(self):
        return max(1, len(self.records))


__all__ = ["CAMUSDataset", "index_images"]
