"""
lib/contrastive.py — Contrastive Anatomical Differentiation loss for CardioContrast.

Implements a contrastive anatomical repulsion loss that enforces representationally
distinct decoder activations for different anatomical structures prompted on the
same image. This is the core of Contribution 2 of CardioContrast.

NOTE ON LOSS FORMULATION:
This is an anatomical repulsion loss, not standard InfoNCE. Standard InfoNCE
requires explicit positives (two augmented views of the same instance). Our
formulation has no explicit positives: it pushes different anatomical structures
apart for the same image. This is the correct formulation for our task — we want
the decoder to produce separable representations for the left ventricle vs
myocardium vs left atrium on the same image. Refer to this as the
"contrastive anatomical repulsion loss" in the paper, not InfoNCE.

Design decisions:
1. Hook point: decoder pre-logit features (B, C, H, W) before conv1_1.
2. Masked average pooling over predicted structure region (not diluted by background).
3. 2-layer MLP projection head following SimCLR best practice.
4. Temperature tau=0.07, fixed following SimCLR, configurable via config.py.
5. Softplus repulsion loss: natural zero minimum, interpretable loss curves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head following SimCLR.
    Projects pooled decoder features into a dedicated contrastive embedding space.
    Separate from the segmentation head so contrastive optimization does not
    distort the features used for mask prediction.
    """
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
    """
    Pool decoder feature map weighted by predicted structure probability.
    Uses the foreground channel of the softmax-normalized two-channel logits
    as the pooling weight, so background pixels do not dilute the
    structure-specific representation.

    Args:
        features:    (B, C, H, W) pre-logit decoder spatial features
        mask_logits: (B, 2, H, W) decoder output logits (background, foreground)
    Returns:
        pooled: (B, C) structure-specific feature vector
    """
    soft_mask = torch.softmax(mask_logits, dim=1)[:, 1:, :, :]
    if soft_mask.shape[-2:] != features.shape[-2:]:
        soft_mask = F.interpolate(soft_mask, size=features.shape[-2:],
                                  mode="bilinear", align_corners=True)
    weighted   = features * soft_mask
    pooled     = weighted.sum(dim=(-2, -1))
    weight_sum = soft_mask.sum(dim=(-2, -1)).clamp(min=1e-6)
    return pooled / weight_sum


def anatomical_repulsion_loss(embeddings, image_ids, structure_ids, tau=0.07):
    """
    Contrastive anatomical repulsion loss using softplus formulation.

    For each anchor (image_i, structure_a), negatives are all samples
    (image_i, structure_b) where b != a. The softplus formulation ensures
    the loss has a natural minimum of zero when all same-image cross-structure
    similarities are maximally negative, producing interpretable loss curves.

    L = (1/|A|) * sum_{i in A} sum_{j in N(i)} softplus(sim(g_i, g_j) / tau)

    where softplus(x) = log(1 + exp(x)), applied per pair independently.

    Args:
        embeddings:    (B, D) L2-normalized projected embeddings
        image_ids:     (B,)  integer source image ID per sample
        structure_ids: (B,)  structure label (1, 2, 3) per sample
        tau:           float temperature
    Returns:
        scalar loss (0.0 if no same-image pairs exist in batch)
    """
    B      = embeddings.size(0)
    device = embeddings.device

    sim_matrix    = torch.matmul(embeddings, embeddings.T) / tau
    image_ids     = image_ids.unsqueeze(1)
    structure_ids = structure_ids.unsqueeze(1)

    same_image     = (image_ids == image_ids.T)
    same_structure = (structure_ids == structure_ids.T)
    same_sample    = torch.eye(B, dtype=torch.bool, device=device)
    negative_mask  = same_image & ~same_structure & ~same_sample

    has_negatives = negative_mask.any(dim=1)
    if not has_negatives.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    losses = []
    for i in range(B):
        if not has_negatives[i]:
            continue
        neg_sims = sim_matrix[i][negative_mask[i]]
        # Softplus applied per pair, then summed.
        # Penalizes each collapsed pair independently; cancellation
        # between pairs is not possible. Natural minimum is zero.
        loss_i = F.softplus(neg_sims).sum()
        losses.append(loss_i)
    return torch.stack(losses).mean()


class ContrastiveAnatomicalLoss(nn.Module):
    """
    Full contrastive anatomical differentiation loss module.
    Combines masked average pooling, projection head, and repulsion loss.

    Usage in training loop:
        module = ContrastiveAnatomicalLoss(in_dim=512)
        loss = module(features, logits, image_ids, structure_ids)
        total_loss = loss_seg + config.CONTRASTIVE_WEIGHT * loss
    """
    def __init__(self, in_dim, proj_hidden_dim=None, proj_out_dim=128, tau=0.07):
        super().__init__()
        self.projection_head = ProjectionHead(in_dim, proj_hidden_dim, proj_out_dim)
        self.tau = tau

    def forward(self, features, mask_logits, image_ids, structure_ids):
        pooled    = masked_average_pool(features, mask_logits)
        projected = self.projection_head(pooled)
        projected = F.normalize(projected, dim=1)
        return anatomical_repulsion_loss(
            projected, image_ids, structure_ids, self.tau)
