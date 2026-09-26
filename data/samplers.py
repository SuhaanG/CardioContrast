"""
data/samplers.py — Batch sampler that groups all structure prompts of an image.

Each batch = `images_per_batch` images x 3 structure prompts, so the contrastive
term always has same-image / different-structure negatives.

IMPORTANT: this sampler is used for EVERY experiment, including the baseline.
The previous code trained the baseline with random batches and the other runs
with grouped batches, which changes BatchNorm statistics and batch composition
and confounds the ablation.

Implemented as a proper batch_sampler (yields lists of indices), which removes
the old "fill pool" logic and the reliance on batch_size aligning with groups.
"""

import numpy as np
from torch.utils.data import Sampler


class GroupedStructureBatchSampler(Sampler):
    def __init__(self, dataset, images_per_batch=1, shuffle=True, seed=42,
                 structures_per_image=3):
        self.groups = dataset.image_groups()
        self.image_keys = sorted(self.groups.keys())
        for k in self.image_keys:
            assert len(self.groups[k]) == structures_per_image, (
                "image {} has {} samples, expected {}".format(
                    k, len(self.groups[k]), structures_per_image))
        self.images_per_batch = images_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        keys = list(self.image_keys)
        if self.shuffle:
            np.random.default_rng([self.seed, self.epoch]).shuffle(keys)
        n_full = len(keys) // self.images_per_batch
        for b in range(n_full):
            batch = []
            for k in keys[b * self.images_per_batch:(b + 1) * self.images_per_batch]:
                batch.extend(self.groups[k])
            yield batch

    def __len__(self):
        return len(self.image_keys) // self.images_per_batch
