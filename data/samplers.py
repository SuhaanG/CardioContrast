"""Deterministic minibatches containing complete per-image structure triplets."""

from collections import defaultdict
from typing import Iterator, List

import numpy as np
from torch.utils.data import Sampler


class GroupedStructureBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, images_per_batch=2, shuffle=True, seed=42):
        if images_per_batch < 1:
            raise ValueError("images_per_batch must be positive")
        self.dataset = dataset
        self.images_per_batch = int(images_per_batch)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if hasattr(dataset, "image_groups"):
            self.groups = dataset.image_groups()
        else:
            grouped = defaultdict(list)
            for index, sample in enumerate(dataset.samples):
                grouped[int(sample["image_idx"])].append(index)
            self.groups = dict(grouped)
        if any(len(indices) != 3 for indices in self.groups.values()):
            raise ValueError("Every image must have exactly three structure prompts")
        self.image_ids = sorted(self.groups)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        image_ids = np.asarray(self.image_ids, dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng([self.seed, self.epoch])
            image_ids = rng.permutation(image_ids)
        usable = len(image_ids) // self.images_per_batch * self.images_per_batch
        for offset in range(0, usable, self.images_per_batch):
            batch_images = image_ids[offset:offset + self.images_per_batch]
            batch = [index for image_id in batch_images for index in self.groups[int(image_id)]]
            yield batch

    def __len__(self):
        return len(self.image_ids) // self.images_per_batch
