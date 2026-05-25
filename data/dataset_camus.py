import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from PIL import Image

# Maps each CAMUS mask label to the text prompt describing it.
STRUCTURE_PROMPTS = {
    1: "the left ventricle",
    2: "the myocardium",
    3: "the left atrium",
}

class CAMUSDataset(Dataset):
    """
    Loads CAMUS image/mask pairs and produces (image, prompt, binary_mask)
    training examples — one per structure per image (Option A).
    """
    def __init__(self, data_dir, img_size=480, transform=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform
        self.samples = self._build_index()

    def _build_index(self):
        samples = []
        mask_paths = glob.glob(
            os.path.join(self.data_dir, "patient*", "*_gt.nii.gz")
        )
        for mask_path in mask_paths:
            if "half_sequence" in mask_path:
                continue
            image_path = mask_path.replace("_gt.nii.gz", ".nii.gz")
            if not os.path.exists(image_path):
                continue
            for label in STRUCTURE_PROMPTS.keys():
                samples.append({
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "label": label,
                    "prompt": STRUCTURE_PROMPTS[label],
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_nifti_2d(self, path):
        arr = nib.load(path).get_fdata()
        return np.squeeze(arr)

    def __getitem__(self, idx):
        s = self.samples[idx]

        image = self._load_nifti_2d(s["image_path"]).astype(np.float32)
        if image.max() > 0:
            image = image / image.max() * 255.0
        image = image.astype(np.uint8)
        image = np.stack([image, image, image], axis=-1)
        image = Image.fromarray(image).resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )

        full_mask = self._load_nifti_2d(s["mask_path"])
        binary_mask = (full_mask == s["label"]).astype(np.uint8)
        binary_mask = Image.fromarray(binary_mask * 255).resize(
            (self.img_size, self.img_size), Image.NEAREST
        )
        binary_mask = (np.array(binary_mask) > 127).astype(np.float32)

        image = np.array(image)

        sample = {
            "image": image,
            "mask": binary_mask,
            "prompt": s["prompt"],
            "label": s["label"],
            "image_path": s["image_path"],
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
