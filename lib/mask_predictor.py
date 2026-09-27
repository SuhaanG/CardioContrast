"""
lib/mask_predictor.py — LAVT decoder with optional multi-stage language cross-attention.

Changes vs. the previous version:
  * No double projection. The old layer applied q/k/v Linear layers and then
    nn.MultiheadAttention applied its own in-projections on top. Now a single
    MultiheadAttention with kdim/vdim = lang_dim does the projection (Eq. 5).
  * Pre-norm + zero-initialised tanh gate on the residual:
        S_hat = S + tanh(alpha) * W_out( MHA( LN(S), L, L ) )
    With alpha = 0 at init, the decoder starts EXACTLY as the baseline decoder
    and learns how much language to inject. The old version replaced the
    feature map with out_proj(LN(S + Z)) (no identity path), which perturbs a
    pretrained-style decoder at step 0. Update Eq. 7 in the paper to match.
  * Gate values are exposed (`gate_values()`) so the paper can report how much
    each stage actually uses language after training.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DecoderCrossAttention(nn.Module):
    def __init__(self, visual_dim, lang_dim=768, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(visual_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=visual_dim, num_heads=num_heads, kdim=lang_dim, vdim=lang_dim,
            dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(visual_dim, visual_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, visual_feat, lang_feat, lang_mask=None):
        """
        visual_feat: (B, C, H, W)
        lang_feat:   (B, C_l, N_l)   (LAVT layout)
        lang_mask:   (B, N_l, 1)     1 = valid token, 0 = padding
        """
        B, C, H, W = visual_feat.shape
        vis = visual_feat.flatten(2).transpose(1, 2)            # (B, HW, C)
        lang = lang_feat.transpose(1, 2)                         # (B, N_l, C_l)
        kpm = None
        if lang_mask is not None:
            kpm = lang_mask.squeeze(-1) == 0                     # True = ignore
        z, _ = self.attn(self.norm_q(vis), lang, lang, key_padding_mask=kpm,
                         need_weights=False)
        out = vis + torch.tanh(self.gate) * self.out_proj(z)
        return out.transpose(1, 2).reshape(B, C, H, W)


class SimpleDecoding(nn.Module):
    """
    LAVT decoder. When lang_feat is None the cross-attention layers are skipped
    and the decoder is identical to the original LAVT SimpleDecoding.
    """

    def __init__(self, c4_dims, factor=2, lang_dim=768, num_heads=8):
        super().__init__()
        hidden_size = c4_dims // factor
        c4_size = c4_dims
        c3_size = c4_dims // (factor ** 1)
        c2_size = c4_dims // (factor ** 2)
        c1_size = c4_dims // (factor ** 3)
        self.hidden_size = hidden_size

        # GroupNorm, not BatchNorm: every training batch contains all three
        # prompts of each image, so BatchNorm statistics would couple those samples
        # (the known BN leakage problem in contrastive training) and make train-mode
        # and eval-mode behaviour diverge. GroupNorm is per-sample.
        def block(cin):
            return nn.Sequential(
                nn.Conv2d(cin, hidden_size, 3, padding=1, bias=False),
                nn.GroupNorm(32, hidden_size), nn.ReLU(inplace=True),
                nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False),
                nn.GroupNorm(32, hidden_size), nn.ReLU(inplace=True))

        self.stage1 = block(c4_size + c3_size)
        self.stage2 = block(hidden_size + c2_size)
        self.stage3 = block(hidden_size + c1_size)
        self.ca_stage1 = DecoderCrossAttention(hidden_size, lang_dim, num_heads)
        self.ca_stage2 = DecoderCrossAttention(hidden_size, lang_dim, num_heads)
        self.ca_stage3 = DecoderCrossAttention(hidden_size, lang_dim, num_heads)
        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)

    @staticmethod
    def _up_to(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=True)
        return x

    def gate_values(self):
        return {n: float(torch.tanh(m.gate).item())
                for n, m in [("stage1", self.ca_stage1), ("stage2", self.ca_stage2),
                             ("stage3", self.ca_stage3)]}

    def forward(self, x_c4, x_c3, x_c2, x_c1, lang_feat=None, lang_mask=None,
                return_features=False):
        use_lang = lang_feat is not None

        x = self.stage1(torch.cat([self._up_to(x_c4, x_c3), x_c3], dim=1))
        if use_lang:
            x = self.ca_stage1(x, lang_feat, lang_mask)

        x = self.stage2(torch.cat([self._up_to(x, x_c2), x_c2], dim=1))
        if use_lang:
            x = self.ca_stage2(x, lang_feat, lang_mask)

        x = self.stage3(torch.cat([self._up_to(x, x_c1), x_c1], dim=1))
        if use_lang:
            x = self.ca_stage3(x, lang_feat, lang_mask)

        logits = self.conv1_1(x)
        if return_features:
            return logits, x
        return logits
