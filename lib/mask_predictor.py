import torch
from torch import nn
from torch.nn import functional as F


class DecoderCrossAttention(nn.Module):
    def __init__(self, visual_dim, lang_dim=768, num_heads=8, kdim=768, vdim=768):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=visual_dim,
            kdim=kdim,
            vdim=vdim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm = nn.LayerNorm(visual_dim)
        self.out_proj = nn.Linear(visual_dim, visual_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, visual_feat, lang_feat, lang_mask=None):
        B, C, H, W = visual_feat.shape
        S = visual_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        L = lang_feat.permute(0, 2, 1)

        key_padding_mask = None
        if lang_mask is not None:
            key_padding_mask = (lang_mask.squeeze(-1) == 0)

        M, _ = self.mha(self.norm(S), L, L, key_padding_mask=key_padding_mask)
        out = S + torch.tanh(self.gate) * self.out_proj(M)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)

    def gate_value(self):
        return float(self.gate.detach().cpu().item())


class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, factor=2, lang_dim=768, num_heads=8):
        super().__init__()
        hidden_size = c4_dims // factor
        c4_size = c4_dims
        c3_size = c4_dims // factor
        c2_size = c4_dims // (factor ** 2)
        c1_size = c4_dims // (factor ** 3)
        if c3_size == c4_size:
            c3_size = c4_size // 2
        if c2_size == c4_size:
            c2_size = c4_size // 4
        if c1_size == c4_size:
            c1_size = c4_size // 8

        self.stage1 = nn.Sequential(
            nn.Conv2d(c4_size + c3_size, hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.ca_stage1 = DecoderCrossAttention(hidden_size, lang_dim, num_heads)

        self.stage2 = nn.Sequential(
            nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.ca_stage2 = DecoderCrossAttention(hidden_size, lang_dim, num_heads)

        self.stage3 = nn.Sequential(
            nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.ca_stage3 = DecoderCrossAttention(hidden_size, lang_dim, num_heads)

        self.head = nn.Conv2d(hidden_size, 2, 1)

    def gate_values(self):
        values = []
        for mod in (self.ca_stage1, self.ca_stage2, self.ca_stage3):
            values.append(mod.gate_value())
        return values

    def forward(self, x_c4, x_c3, x_c2, x_c1, lang_feat=None, lang_mask=None, return_features=False):
        if x_c4.size(-2) > x_c3.size(-2) or x_c4.size(-1) > x_c3.size(-1):
            x_c4 = F.interpolate(x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode="bilinear", align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.stage1(x)
        if lang_feat is not None:
            x = self.ca_stage1(x, lang_feat, lang_mask)

        if x.size(-2) > x_c2.size(-2) or x.size(-1) > x_c2.size(-1):
            x = F.interpolate(x, size=(x_c2.size(-2), x_c2.size(-1)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.stage2(x)
        if lang_feat is not None:
            x = self.ca_stage2(x, lang_feat, lang_mask)

        if x.size(-2) > x_c1.size(-2) or x.size(-1) > x_c1.size(-1):
            x = F.interpolate(x, size=(x_c1.size(-2), x_c1.size(-1)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x_c1], dim=1)
        x = self.stage3(x)
        if lang_feat is not None:
            x = self.ca_stage3(x, lang_feat, lang_mask)

        pre_logit_features = x
        logits = self.head(x)
        if return_features:
            return logits, pre_logit_features
        return logits
