import torch
from torch import nn
from torch.nn import functional as F


class ClassEmbeddingEncoder(nn.Module):
    def __init__(self, num_classes=3, embed_tokens=8, embed_dim=768):
        super().__init__()
        self.num_classes = num_classes
        self.embed_tokens = embed_tokens
        self.table = nn.Parameter(torch.zeros(num_classes, embed_tokens, embed_dim))
        nn.init.normal_(self.table, mean=0.0, std=0.02)
        self.register_buffer("mask", torch.ones(num_classes, embed_tokens, 1, dtype=torch.float32))

    def forward(self, class_ids):
        class_ids = class_ids.clamp(0, self.num_classes - 1)
        return self.table[class_ids]


class CardioContrastNet(nn.Module):
    def __init__(self, backbone, classifier, text_encoder="bert", bert_path=None,
                 bert_trainable_layers=10, decode_with_lang=True, embed_tokens=8):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder_type = text_encoder
        self.decode_with_lang = bool(decode_with_lang)
        self.bert_path = bert_path
        self.bert_trainable_layers = int(bert_trainable_layers)
        self.embed_tokens = int(embed_tokens)

        if text_encoder == "bert":
            from transformers import BertModel

            self.text_encoder = BertModel.from_pretrained(
                bert_path or "bert-base-uncased", add_pooling_layer=False
            )
            for parameter in self.text_encoder.embeddings.parameters():
                parameter.requires_grad = False
            for layer in self.text_encoder.encoder.layer[self.bert_trainable_layers:]:
                layer.requires_grad_(False)
        elif text_encoder == "embedding":
            self.text_encoder = ClassEmbeddingEncoder(3, self.embed_tokens, 768)
        else:
            raise ValueError(f"Unknown text_encoder={text_encoder!r}")

    def encode_text(self, input_ids, attn_mask, class_ids):
        if self.text_encoder_type == "embedding":
            token_embeddings = self.text_encoder(class_ids)
            lang_feat = token_embeddings.permute(0, 2, 1)
            lang_mask = torch.ones(
                (token_embeddings.shape[0], token_embeddings.shape[1], 1),
                device=token_embeddings.device,
                dtype=torch.long,
            )
            return lang_feat, lang_mask
        output = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        return output.last_hidden_state.permute(0, 2, 1), attn_mask.unsqueeze(-1)

    def forward(self, x, input_ids, attn_mask, class_ids, return_features=False):
        lang_feat, lang_mask = self.encode_text(input_ids, attn_mask, class_ids)
        features = self.backbone(x, lang_feat, lang_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        decoder_lang = lang_feat if self.decode_with_lang else None
        decoder_mask = lang_mask if self.decode_with_lang else None
        logits, pre_logit_features = self.classifier(
            x_c4, x_c3, x_c2, x_c1,
            lang_feat=decoder_lang,
            lang_mask=decoder_mask,
            return_features=True,
        )
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=True)
        if return_features:
            return logits, pre_logit_features
        return logits
