"""
lib/_utils.py — CardioContrast model wrapper (LAVT-One style: text encoder inside).

Text encoder options (the language ablation reviewers will ask for):
  'bert'      : BERT-base. Embeddings + the last (12 - bert_trainable_layers)
                blocks are frozen with requires_grad=False (the old code left
                them requiring grad but simply omitted them from the optimizer,
                which wasted memory/compute and made the paper text and figure
                disagree about what is frozen).
  'embedding' : a learned (num_classes x num_tokens x 768) table indexed by the
                structure id. Same tensor layout as BERT output, so the rest of
                the network is unchanged. If CardioContrast with BERT does not
                beat this, the "language-guided" claim does not hold.

decode_with_lang is a constructor attribute (not a forward kwarg), so it is
saved with the checkpoint and cannot silently differ between train and eval.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ClassEmbeddingEncoder(nn.Module):
    def __init__(self, num_classes=3, num_tokens=8, dim=768):
        super().__init__()
        self.table = nn.Parameter(torch.randn(num_classes, num_tokens, dim) * 0.02)
        self.num_tokens = num_tokens

    def forward(self, class_ids):
        feats = self.table[class_ids]                              # (B, T, D)
        mask = torch.ones(feats.shape[:2], dtype=torch.long, device=feats.device)
        return feats, mask


class CardioContrastNet(nn.Module):
    def __init__(self, backbone, classifier, text_encoder="bert", bert_path="bert-base-uncased",
                 bert_trainable_layers=10, decode_with_lang=True, num_classes=3,
                 embed_tokens=8):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder_type = text_encoder
        self.decode_with_lang = decode_with_lang

        if text_encoder == "bert":
            from transformers import BertModel
            self.text_encoder = BertModel.from_pretrained(bert_path, add_pooling_layer=False)
            n_layers = len(self.text_encoder.encoder.layer)
            for p in self.text_encoder.embeddings.parameters():
                p.requires_grad = False
            for i, layer in enumerate(self.text_encoder.encoder.layer):
                trainable = i < bert_trainable_layers
                for p in layer.parameters():
                    p.requires_grad = trainable
            print("[model] BERT: embeddings frozen, blocks 1-{} trainable, {}-{} frozen".format(
                bert_trainable_layers, bert_trainable_layers + 1, n_layers), flush=True)
        elif text_encoder == "embedding":
            self.text_encoder = ClassEmbeddingEncoder(num_classes, embed_tokens, 768)
            print("[model] Text encoder: learned class embedding ({} tokens)".format(
                embed_tokens), flush=True)
        else:
            raise ValueError(text_encoder)

    def encode_text(self, input_ids, attn_mask, class_ids):
        if self.text_encoder_type == "bert":
            feats = self.text_encoder(input_ids, attention_mask=attn_mask)[0]    # (B, T, 768)
            mask = attn_mask
        else:
            feats, mask = self.text_encoder(class_ids)
        return feats.permute(0, 2, 1), mask.unsqueeze(-1)                        # (B,768,T),(B,T,1)

    def forward(self, x, input_ids=None, attn_mask=None, class_ids=None, return_features=False):
        input_shape = x.shape[-2:]
        l_feats, l_mask = self.encode_text(input_ids, attn_mask, class_ids)
        x_c1, x_c2, x_c3, x_c4 = self.backbone(x, l_feats, l_mask)

        lang = l_feats if self.decode_with_lang else None
        lmask = l_mask if self.decode_with_lang else None
        logits, feats = self.classifier(x_c4, x_c3, x_c2, x_c1, lang_feat=lang,
                                        lang_mask=lmask, return_features=True)
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=True)
        if return_features:
            return logits, feats
        return logits
