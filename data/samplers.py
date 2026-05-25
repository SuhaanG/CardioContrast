"""
data/samplers.py — Custom batch sampler for contrastive training.

Guarantees every batch contains all 3 structure prompts from at least one
image, so the contrastive loss always has same-image-different-structure
negative pairs. Without this, random sampling puts the same image's
structures in the same batch with ~0.4% probability per step, meaning the
contrastive loss fires almost never.
"""

import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict


class GroupedStructureSampler(Sampler):
    """
    Each batch is guaranteed to contain all 3 structures from at least one
    anchor image. Remaining batch slots are filled with random samples.

    Args:
        dataset:    CAMUSDatasetContrastive instance
        batch_size: physical batch size
        shuffle:    shuffle image order each epoch
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.all_indices = list(range(len(dataset)))

        self.groups = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            self.groups[sample["image_idx"]].append(idx)

        self.image_keys = list(self.groups.keys())

    def __iter__(self):
        image_keys  = self.image_keys.copy()
        all_indices = self.all_indices.copy()
        if self.shuffle:
            np.random.shuffle(image_keys)
            np.random.shuffle(all_indices)

        ptr = 0
        for key in image_keys:
            group     = self.groups[key]
            remaining = self.batch_size - len(group)
            fill      = []
            attempts  = 0
            while len(fill) < max(0, remaining):
                if ptr >= len(all_indices):
                    np.random.shuffle(all_indices)
                    ptr = 0
                if all_indices[ptr] not in group:
                    fill.append(all_indices[ptr])
                ptr += 1
                attempts += 1
                if attempts > len(all_indices) * 2:
                    break
            batch = (group + fill)[:self.batch_size]
            if self.shuffle:
                np.random.shuffle(batch)
            yield from batch

    def __len__(self):
        return len(self.all_indices)
