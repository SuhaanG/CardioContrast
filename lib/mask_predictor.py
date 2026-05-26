import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict


class DecoderCrossAttention(nn.Module):
    """
    Single cross-attention layer for decoder-stage language conditioning.

    At each decoder refinement stage, spatial features serve as queries
    and language token embeddings serve as keys and values. This allows
    each stage to selectively attend to the language tokens most relevant
    to the current spatial scale, conditioning boundary placement on
    semantic intent at every level of refinement.

    This is the architectural response to the observation that encoder-side
    fusion (LAVT) fixes semantic conditioning before the decoder begins;
    decoder-side cross-attention re-applies it at each refinement step.

    Args:
        visual_dim:  channel dimension of the spatial feature map
        lang_dim:    dimension of language token embeddings (BERT: 768)
        num_heads:   number of attention heads
    """
    def __init__(self, visual_dim, lang_dim=768, num_heads=8):
        super().__init__()
        self.q_proj   = nn.Linear(visual_dim, visual_dim)
        self.k_proj   = nn.Linear(lang_dim, visual_dim)
        self.v_proj   = nn.Linear(lang_dim, visual_dim)
        self.attn     = nn.MultiheadAttention(
            embed_dim=visual_dim, num_heads=num_heads,
            batch_first=True, dropout=0.0)
        self.norm     = nn.LayerNorm(visual_dim)
        self.out_proj = nn.Linear(visual_dim, visual_dim)

    def forward(self, visual_feat, lang_feat, lang_mask=None):
        """
        Args:
            visual_feat: (B, C, H, W)
            lang_feat:   (B, C_l, N_l) language tokens (LAVT format)
            lang_mask:   (B, N_l, 1)   1=valid token, 0=padding
        Returns:
            (B, C, H, W) language-conditioned spatial features
        """
        B, C, H, W = visual_feat.shape
        vis_seq  = visual_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        lang_seq = lang_feat.permute(0, 2, 1)           # (B, N_l, C_l)

        Q = self.q_proj(vis_seq)
        K = self.k_proj(lang_seq)
        V = self.v_proj(lang_seq)

        key_padding_mask = None
        if lang_mask is not None:
            key_padding_mask = (lang_mask.squeeze(-1) == 0)  # (B, N_l)

        attn_out, _ = self.attn(Q, K, V, key_padding_mask=key_padding_mask)
        out = self.norm(vis_seq + attn_out)
        out = self.out_proj(out)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


class SimpleDecoding(nn.Module):
    """
    LAVT decoder with optional multi-stage cross-attention language conditioning.

    When lang_feat is provided (CardioContrast mode), cross-attention is applied
    after each of the three refinement stages, injecting language at every level
    of spatial boundary refinement. When lang_feat is None (baseline mode), the
    cross-attention layers are bypassed and the decoder behaves identically to
    the original LAVT SimpleDecoding, ensuring a clean controlled ablation.
    """
    def __init__(self, c4_dims, factor=2, lang_dim=768, num_heads=8):
        super(SimpleDecoding, self).__init__()
        hidden_size = c4_dims // factor
        c4_size = c4_dims
        c3_size = c4_dims // (factor ** 1)
        c2_size = c4_dims // (factor ** 2)
        c1_size = c4_dims // (factor ** 3)

        self.conv1_4 = nn.Conv2d(c4_size + c3_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_4   = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_4   = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()
        self.ca_stage1 = DecoderCrossAttention(hidden_size, lang_dim, num_heads)

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_3   = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()
        self.conv2_3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_3   = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()
        self.ca_stage2 = DecoderCrossAttention(hidden_size, lang_dim, num_heads)

        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_2   = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_2   = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU()
        self.ca_stage3 = DecoderCrossAttention(hidden_size, lang_dim, num_heads)

        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)

    def forward(self, x_c4, x_c3, x_c2, x_c1,
                lang_feat=None, lang_mask=None,
                return_features=False):
        use_lang = lang_feat is not None

        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(x_c4, size=(x_c3.size(-2), x_c3.size(-1)),
                                  mode="bilinear", align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.relu1_4(self.bn1_4(self.conv1_4(x)))
        x = self.relu2_4(self.bn2_4(self.conv2_4(x)))
        if use_lang:
            x = self.ca_stage1(x, lang_feat, lang_mask)

        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(x, size=(x_c2.size(-2), x_c2.size(-1)),
                              mode="bilinear", align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.relu1_3(self.bn1_3(self.conv1_3(x)))
        x = self.relu2_3(self.bn2_3(self.conv2_3(x)))
        if use_lang:
            x = self.ca_stage2(x, lang_feat, lang_mask)

        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(x, size=(x_c1.size(-2), x_c1.size(-1)),
                              mode="bilinear", align_corners=True)
        x = torch.cat([x, x_c1], dim=1)
        x = self.relu1_2(self.bn1_2(self.conv1_2(x)))
        x = self.relu2_2(self.bn2_2(self.conv2_2(x)))
        if use_lang:
            x = self.ca_stage3(x, lang_feat, lang_mask)

        pre_logit_features = x
        logits = self.conv1_1(x)

        if return_features:
            return logits, pre_logit_features
        return logits
