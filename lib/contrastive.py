"""
lib/contrastive.py — Contrastive Anatomical Differentiation loss for CardioContrast.

Implements the InfoNCE-based contrastive objective that enforces representationally
distinct decoder activations for different anatomical structures prompted on the
same image. This is the core contribution of CardioContrast.

Design:
1. Hook point: decoder pre-logit features (B, C, H, W) before conv1_1
2. Masked average pooling over predicted structure region (structure-specific)
3. 2-layer MLP projection head (SimCLR best practice)
4. InfoNCE loss with same-image-different-structure negatives, tau=0.07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=128):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def masked_average_pool(features, mask_logits):
    soft_mask = torch.softmax(mask_logits, dim=1)[:, 1:, :, :]
    if soft_mask.shape[-2:] != features.shape[-2:]:
        soft_mask = F.interpolate(soft_mask, size=features.shape[-2:],
                                  mode='bilinear', align_corners=True)
    weighted   = features * soft_mask
    pooled     = weighted.sum(dim=(-2, -1))
    weight_sum = soft_mask.sum(dim=(-2, -1)).clamp(min=1e-6)
    return pooled / weight_sum


def info_nce_same_image(embeddings, image_ids, structure_ids, tau=0.07):
    B      = embeddings.size(0)
    device = embeddings.device
    sim_matrix    = torch.matmul(embeddings, embeddings.T) / tau
    image_ids     = image_ids.unsqueeze(1)
    structure_ids = structure_ids.unsqueeze(1)
    same_image     = (image_ids == image_ids.T)
    same_structure = (structure_ids == structure_ids.T)
    same_sample    = torch.eye(B, dtype=torch.bool, device=device)
    negative_mask  = same_image & ~same_structure & ~same_sample
    has_negatives  = negative_mask.any(dim=1)
    if not has_negatives.any():
        return torch.tensor(0.0, device=device, requires_grad=True)
    losses = []
    for i in range(B):
        if not has_negatives[i]:
            continue
        neg_sims = sim_matrix[i][negative_mask[i]]
        loss_i   = torch.log(torch.exp(neg_sims).sum() + 1e-8)
        losses.append(loss_i)
    return torch.stack(losses).mean()


class ContrastiveAnatomicalLoss(nn.Module):
    def __init__(self, in_dim, proj_hidden_dim=None, proj_out_dim=128, tau=0.07):
        super().__init__()
        self.projection_head = ProjectionHead(in_dim, proj_hidden_dim, proj_out_dim)
        self.tau = tau

    def forward(self, features, mask_logits, image_ids, structure_ids):
        pooled    = masked_average_pool(features, mask_logits)
        projected = self.projection_head(pooled)
        projected = F.normalize(projected, dim=1)
        return info_nce_same_image(projected, image_ids, structure_ids, self.tau)
