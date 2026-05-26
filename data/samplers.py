"""
data/samplers.py — Custom batch sampler for contrastive training.

Guarantees every batch contains all 3 structure prompts from at least one
image, so the contrastive loss always has same-image-different-structure
negative pairs. Without this, random sampling puts the same image structures
in the same batch with ~0.4% probability per step, meaning the contrastive
loss fires almost never.

IMPORTANT: always pass train_indices (the split indices, not the full dataset)
to prevent sampling validation examples during training.
"""

import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict


class GroupedStructureSampler(Sampler):
    """
    Each batch is guaranteed to contain all 3 structures from at least one
    anchor image. Remaining batch slots are filled with random samples.

    Args:
        dataset:       CAMUSDatasetContrastive instance (full dataset)
        train_indices: list of integer indices belonging to the train split.
                       Must be train-split indices only, not all indices,
                       to prevent sampling from the validation set.
        batch_size:    physical batch size
        shuffle:       shuffle image order each epoch
    """
    def __init__(self, dataset, train_indices, batch_size, shuffle=True):
        self.batch_size    = batch_size
        self.shuffle       = shuffle
        self.train_indices = list(train_indices)

        self.groups = defaultdict(list)
        for idx in self.train_indices:
            sample = dataset.samples[idx]
            self.groups[sample["image_idx"]].append(idx)

        self.image_keys = list(self.groups.keys())

    def __iter__(self):
        image_keys    = self.image_keys.copy()
        all_train_idx = self.train_indices.copy()
        if self.shuffle:
            np.random.shuffle(image_keys)
            np.random.shuffle(all_train_idx)

        ptr = 0
        for key in image_keys:
            group     = self.groups[key]
            remaining = self.batch_size - len(group)
            fill      = []
            attempts  = 0
            while len(fill) < max(0, remaining):
                if ptr >= len(all_train_idx):
                    np.random.shuffle(all_train_idx)
                    ptr = 0
                if all_train_idx[ptr] not in group:
                    fill.append(all_train_idx[ptr])
                ptr += 1
                attempts += 1
                if attempts > len(all_train_idx) * 2:
                    break
            batch = (group + fill)[:self.batch_size]
            if self.shuffle:
                np.random.shuffle(batch)
            yield from batch

    def __len__(self):
        return len(self.train_indices)
