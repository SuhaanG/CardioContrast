"""
data/samplers.py — Custom batch sampler for contrastive training.

Guarantees every batch contains all 3 structure prompts from the same image,
so the contrastive loss always has same-image-different-structure negative pairs.

Fix: the original sampler passed full-dataset indices to a DataLoader that wraps
a Subset. The Subset re-indexes from 0, so full-dataset indices were fetching
wrong samples. This version works directly with the Subset by using
positional indices into train_ds, not into the full dataset.
"""

import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict


class GroupedStructureSampler(Sampler):
    """
    Each batch is guaranteed to contain all 3 structures from the same image.
    Works on the Subset directly using positional indices into train_ds.

    Args:
        train_ds:   torch.utils.data.Subset (the training split)
        batch_size: must be >= 3 to fit all 3 structures from one image
        shuffle:    shuffle image order each epoch
    """
    def __init__(self, train_ds, batch_size, shuffle=True):
        assert batch_size >= 3, "batch_size must be >= 3 to fit all 3 structures"
        self.batch_size = batch_size
        self.shuffle    = shuffle

        # Group positional indices (0..len(train_ds)-1) by image_idx
        self.groups = defaultdict(list)
        for pos, full_idx in enumerate(train_ds.indices):
            sample = train_ds.dataset.samples[full_idx]
            self.groups[sample["image_idx"]].append(pos)

        self.image_keys  = list(self.groups.keys())
        self.all_pos_idx = list(range(len(train_ds)))

    def __iter__(self):
        image_keys  = self.image_keys.copy()
        all_pos_idx = self.all_pos_idx.copy()
        if self.shuffle:
            np.random.shuffle(image_keys)
            np.random.shuffle(all_pos_idx)

        fill_pool = iter(all_pos_idx)

        for key in image_keys:
            group = self.groups[key]           # positional indices for this image
            batch = list(group[:3])            # take all 3 structures (labels 1,2,3)
            while len(batch) < self.batch_size:
                try:
                    candidate = next(fill_pool)
                except StopIteration:
                    all_pos_idx = self.all_pos_idx.copy()
                    np.random.shuffle(all_pos_idx)
                    fill_pool = iter(all_pos_idx)
                    candidate = next(fill_pool)
                if candidate not in group:
                    batch.append(candidate)
            if self.shuffle:
                np.random.shuffle(batch)
            yield from batch[:self.batch_size]

    def __len__(self):
        return len(self.all_pos_idx)