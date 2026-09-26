"""
lib/contrastive.py — Contrastive anatomical repulsion loss.

For an anchor (image i, structure a), negatives are (image i, structure b != a).

    L = (1/|A|) sum_{i in A} (1/|N(i)|) sum_{j in N(i)} tau * softplus( <z_i, z_j> / tau )

where A = anchors that have at least one negative (write |A| in Eq. 9, not |B|).

Scale fix: the old loss summed softplus(sim/tau) with tau = 0.07, so a collapsed
pair (cos ~ 0.95) cost ~13.6 per pair (~27 per anchor) -- with lambda = 0.1 the
repulsion term was ~2x the segmentation loss at the start of training and
lambda was not interpretable. Multiplying by tau makes the per-pair loss a
smooth ReLU of the cosine (~cos when positive, ~0 when negative), bounded by 1,
and independent of tau's scale; averaging over N(i) makes it independent of
the number of structures. lambda must then be tuned on VAL (e.g. 0.1/0.3/1.0).

Changes vs. the previous version:
  * Pooling region is configurable (--pool_region), because the choice changes
    what the loss means:
      'pred'  : weight by the predicted foreground probability (paper default).
                Different prompts pool DIFFERENT pixels, so features can differ
                simply because the regions differ -- a reviewer can call this
                trivially satisfiable.
      'gt'    : weight by the ground-truth mask of the prompted structure.
                Stable early in training when predictions are still noise.
      'union' : weight by the union of all three structures, IDENTICAL for all
                prompts of the same image. The loss then forces prompt-
                conditioned features at the SAME pixels to diverge -- the
                cleanest operationalisation of "disambiguation". Recommended
                ablation; if it wins, make it the main method.
  * detach_weights (default True): the pooling weights are detached, so the
    repulsion term can only change features, not reshape the predicted masks
    to make the loss smaller. The old code let gradients flow into the mask.
  * Projection head uses LayerNorm instead of BatchNorm1d. With 3 samples per
    batch (one image x 3 prompts), BatchNorm statistics over 3 vectors are
    extremely noisy. Update the paper text ("batch normalization") to match.
  * Returns diagnostics (mean cosine similarity of negative pairs) so the paper
    can show the loss actually separates representations.
"""

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


def weighted_pool(features, weights, eps=1e-6):
    """features: (B,C,h,w); weights: (B,1,H,W) in [0,1] -> (B,C)."""
    if weights.shape[-2:] != features.shape[-2:]:
        weights = F.interpolate(weights, size=features.shape[-2:], mode="bilinear",
                                align_corners=False)
    num = (features * weights).sum(dim=(-2, -1))
    den = weights.sum(dim=(-2, -1)).clamp(min=eps)
    return num / den


def anatomical_repulsion_loss(z, image_ids, structure_ids, tau=0.07):
    """z: (B,D) L2-normalised. Returns (loss, mean_negative_cosine, n_anchors)."""
    sim = z @ z.t()
    same_img = image_ids[:, None] == image_ids[None, :]
    diff_struct = structure_ids[:, None] != structure_ids[None, :]
    neg = same_img & diff_struct
    has_neg = neg.any(dim=1)
    if not has_neg.any():
        zero = z.sum() * 0.0
        return zero, float("nan"), 0
    per_pair = tau * F.softplus(sim / tau) * neg
    per_anchor = per_pair.sum(dim=1)[has_neg] / neg.sum(dim=1)[has_neg]
    loss = per_anchor.mean()
    mean_cos = sim[neg].mean().item()
    return loss, mean_cos, int(has_neg.sum().item())


class ContrastiveAnatomicalLoss(nn.Module):
    def __init__(self, in_dim, proj_hidden_dim=None, proj_out_dim=128, tau=0.07,
                 pool_region="pred", detach_weights=True):
        super().__init__()
        assert pool_region in ("pred", "gt", "union")
        self.projection_head = ProjectionHead(in_dim, proj_hidden_dim, proj_out_dim)
        self.tau = tau
        self.pool_region = pool_region
        self.detach_weights = detach_weights

    def forward(self, features, logits, image_ids, structure_ids, target=None, union=None):
        if self.pool_region == "pred":
            w = torch.softmax(logits.float(), dim=1)[:, 1:2]
        elif self.pool_region == "gt":
            w = target.unsqueeze(1).float()
        else:
            w = union.unsqueeze(1).float()
        if self.detach_weights:
            w = w.detach()
        pooled = weighted_pool(features.float(), w)
        z = F.normalize(self.projection_head(pooled), dim=1)
        return anatomical_repulsion_loss(z, image_ids, structure_ids, self.tau)
