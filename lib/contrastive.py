"""Contrastive anatomical repulsion loss for CardioContrast."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=128):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def weighted_pool(features, weights=None, detach=True):
    if weights is None:
        weights = torch.ones_like(features[:, :1, :, :])
    if weights.dim() == 3:
        weights = weights.unsqueeze(1)
    if weights.shape[-2:] != features.shape[-2:]:
        weights = F.interpolate(weights, size=features.shape[-2:], mode="bilinear", align_corners=True)
    if detach and weights.requires_grad:
        weights = weights.detach()
    if weights.shape[1] == 1 and features.shape[1] > 1:
        weights = weights.expand(-1, features.shape[1], -1, -1)
    weight_sum = weights.sum(dim=(-1, -2)).clamp_min(1e-6)
    pooled = (features * weights).sum(dim=(-1, -2)) / weight_sum
    return pooled


def masked_average_pool(features, mask_logits):
    return weighted_pool(features, torch.softmax(mask_logits, dim=1)[:, 1:2], detach=True)


def contrastive_repulsion_loss(embeddings, image_ids, structure_ids, tau=0.07):
    if tau <= 0:
        raise ValueError("tau must be positive")
    embeddings = F.normalize(embeddings, dim=1)
    device = embeddings.device
    sim = embeddings @ embeddings.T

    image_ids = image_ids.to(device)
    structure_ids = structure_ids.to(device)
    if embeddings.size(0) < 2:
        zero = embeddings.sum() * 0.0
        return zero, zero, 0

    neg_losses = []
    neg_cosines = []
    valid = 0
    for i in range(embeddings.size(0)):
        same_image = image_ids == image_ids[i]
        same_structure = structure_ids == structure_ids[i]
        neg_idx = torch.nonzero(same_image & (~same_structure), as_tuple=False).flatten()
        if neg_idx.numel() == 0:
            continue
        valid += 1
        neg_sims = sim[i, neg_idx] / tau
        neg_losses.append((tau * F.softplus(neg_sims)).mean())
        neg_cosines.append(F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[neg_idx]).mean())

    if not neg_losses:
        zero = embeddings.sum() * 0.0
        return zero, zero.detach(), 0

    loss = torch.stack(neg_losses).mean()
    neg_cos = torch.stack(neg_cosines).mean() if neg_cosines else torch.zeros((), device=device)
    return loss, neg_cos, valid


class ContrastiveAnatomicalLoss(nn.Module):
    def __init__(self, in_dim, proj_hidden_dim=None, proj_out_dim=128, tau=0.07):
        super().__init__()
        self.projection_head = ProjectionHead(in_dim, proj_hidden_dim, proj_out_dim)
        self.tau = tau

    def forward(self, features, mask_logits, image_ids, structure_ids,
                pool_region="pred", gt_masks=None, union_masks=None,
                detach_pooling=True):
        if pool_region == "pred":
            weights = torch.softmax(mask_logits, dim=1)[:, 1:2]
        elif pool_region == "gt":
            if gt_masks is None:
                raise ValueError("gt_masks are required for pool_region='gt'")
            weights = gt_masks
        elif pool_region == "union":
            if union_masks is None:
                raise ValueError("union_masks are required for pool_region='union'")
            weights = union_masks
        else:
            raise ValueError(f"Unknown pool_region={pool_region!r}")
        pooled = weighted_pool(features, weights, detach=detach_pooling)
        projected = self.projection_head(pooled)
        loss, neg_cos, n_anchors = contrastive_repulsion_loss(projected, image_ids, structure_ids, self.tau)
        return loss, neg_cos, n_anchors


__all__ = [
    "ProjectionHead",
    "weighted_pool",
    "masked_average_pool",
    "contrastive_repulsion_loss",
    "ContrastiveAnatomicalLoss",
]
