import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.utils.data as data
from PIL import Image
from bert.tokenization_bert import BertTokenizer

STRUCTURE_PROMPTS = {
    1: "the left ventricular endocardium",
    2: "the myocardium",
    3: "the left atrium",
}


class CAMUSDatasetContrastive(data.Dataset):
    """
    Extended CAMUS dataset for contrastive training.
    Returns 6-tuple: (img, target, tokens, attn, image_idx, label)
    image_idx and label are needed by ContrastiveAnatomicalLoss to construct
    same-image-different-structure negative pairs.
    """
    def __init__(self, data_dir, bert_tokenizer="bert-base-uncased",
                 image_transforms=None, max_tokens=20):
        self.data_dir         = data_dir
        self.image_transforms = image_transforms
        self.max_tokens       = max_tokens
        self.tokenizer        = BertTokenizer.from_pretrained(bert_tokenizer)
        self.samples          = self._build_index()

    def _build_index(self):
        mask_paths = sorted(glob.glob(
            os.path.join(self.data_dir, "patient*", "*_gt.nii.gz")))
        seen = {}
        samples = []
        for mask_path in mask_paths:
            if "half_sequence" in mask_path:
                continue
            image_path = mask_path.replace("_gt.nii.gz", ".nii.gz")
            if not os.path.exists(image_path):
                continue
            if image_path not in seen:
                seen[image_path] = len(seen)
            img_idx = seen[image_path]
            for label in STRUCTURE_PROMPTS.keys():
                samples.append({
                    "image_path": image_path,
                    "mask_path":  mask_path,
                    "label":      label,
                    "prompt":     STRUCTURE_PROMPTS[label],
                    "image_idx":  img_idx,
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_nifti_2d(self, path):
        return np.squeeze(nib.load(path).get_fdata())

    def _tokenize(self, sentence):
        attention_mask   = [0] * self.max_tokens
        padded_input_ids = [0] * self.max_tokens
        input_ids = self.tokenizer.encode(text=sentence, add_special_tokens=True)
        input_ids = input_ids[:self.max_tokens]
        padded_input_ids[:len(input_ids)] = input_ids
        attention_mask[:len(input_ids)]   = [1] * len(input_ids)
        return (torch.tensor(padded_input_ids).unsqueeze(0),
                torch.tensor(attention_mask).unsqueeze(0))

    def __getitem__(self, index):
        s     = self.samples[index]
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

        tokens, attn = self._tokenize(s["prompt"])
        image_idx    = torch.tensor(s["image_idx"], dtype=torch.int64)
        label        = torch.tensor(s["label"],     dtype=torch.int64)

        return img, target, tokens, attn, image_idx, label
