import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.utils.data as data
from PIL import Image
from transformers import BertTokenizer

STRUCTURE_PROMPTS = {
    1: "the left ventricular endocardium",
    2: "the myocardium",
    3: "the left atrium",
}

class CAMUSDataset(data.Dataset):
    """
    Loads CAMUS image/mask pairs and produces one training example per
    structure per image (Option A). Returns the same 4-tuple format as
    LAVT's ReferDataset:
        (img, target, tensor_embeddings, attention_mask)
    """
    def __init__(self, data_dir, bert_tokenizer="bert-base-uncased",
                 image_transforms=None, max_tokens=20):
        self.data_dir = data_dir
        self.image_transforms = image_transforms
        self.max_tokens = max_tokens
        self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        self.samples = self._build_index()

    def _build_index(self):
        samples = []
        mask_paths = glob.glob(
            os.path.join(self.data_dir, "patient*", "*_gt.nii.gz")
        )
        # Sort for reproducibility: glob order varies by filesystem/OS.
        # Without sorting, the seeded random_split produces different
        # train/val splits on different machines.
        mask_paths = sorted(mask_paths)

        for mask_path in mask_paths:
            if "half_sequence" in mask_path:
                continue
            image_path = mask_path.replace("_gt.nii.gz", ".nii.gz")
            if not os.path.exists(image_path):
                continue
            for label in STRUCTURE_PROMPTS.keys():
                samples.append({
                    "image_path": image_path,
                    "mask_path":  mask_path,
                    "label":      label,
                    "prompt":     STRUCTURE_PROMPTS[label],
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_nifti_2d(self, path):
        arr = nib.load(path).get_fdata()
        return np.squeeze(arr)

    def _tokenize(self, sentence):
        attention_mask   = [0] * self.max_tokens
        padded_input_ids = [0] * self.max_tokens
        input_ids = self.tokenizer.encode(text=sentence, add_special_tokens=True)
        input_ids = input_ids[:self.max_tokens]
        padded_input_ids[:len(input_ids)] = input_ids
        attention_mask[:len(input_ids)]   = [1] * len(input_ids)
        tensor_embeddings = torch.tensor(padded_input_ids).unsqueeze(0)
        attention_mask    = torch.tensor(attention_mask).unsqueeze(0)
        return tensor_embeddings, attention_mask

    def __getitem__(self, index):
        s = self.samples[index]

        image = self._load_nifti_2d(s["image_path"]).astype(np.float32)
        if image.max() > 0:
            image = image / image.max() * 255.0
        image = image.astype(np.uint8)
        image = np.stack([image, image, image], axis=-1)
        img   = Image.fromarray(image).convert("RGB")

        full_mask = self._load_nifti_2d(s["mask_path"])
        annot     = np.zeros(full_mask.shape)
        annot[full_mask == s["label"]] = 1
        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            img, target = self.image_transforms(img, annot)
        else:
            target = annot

        tensor_embeddings, attention_mask = self._tokenize(s["prompt"])

        return img, target, tensor_embeddings, attention_mask
