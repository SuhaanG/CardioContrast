import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from .dataset_camus import index_images, preprocess_grayscale_image

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
                 img_size=352, max_tokens=20, split="train", use_language=True):
        self.data_dir         = data_dir
        self.img_size         = img_size
        self.max_tokens       = max_tokens
        self.split            = split
        self.tokenizer = None
        if use_language:
            from transformers import BertTokenizer

            self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        self.samples          = self._build_index()

    def _build_index(self):
        seen = {}
        samples = []
        for record in index_images(self.data_dir, self.split):
            image_path = record["image_path"]
            mask_path = record["mask_path"]
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
        import nibabel as nib

        return np.squeeze(nib.load(path).get_fdata())

    def _tokenize(self, sentence):
        if self.tokenizer is None:
            return (torch.zeros(1, self.max_tokens, dtype=torch.long),
                    torch.zeros(1, self.max_tokens, dtype=torch.long))
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
        image = self._load_nifti_2d(s["image_path"])
        img = preprocess_grayscale_image(image, self.img_size)

        full_mask = self._load_nifti_2d(s["mask_path"])
        annot     = np.zeros(full_mask.shape)
        annot[full_mask == s["label"]] = 1
        annot = Image.fromarray(annot.astype(np.uint8)).resize(
            (self.img_size, self.img_size), Image.NEAREST
        )
        target = torch.as_tensor(np.asarray(annot).copy(), dtype=torch.long)

        tokens, attn = self._tokenize(s["prompt"])
        image_idx    = torch.tensor(s["image_idx"], dtype=torch.int64)
        label        = torch.tensor(s["label"],     dtype=torch.int64)

        return img, target, tokens, attn, image_idx, label
