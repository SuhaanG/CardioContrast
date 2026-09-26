import torch
from torch import nn
from torch.nn import functional as F


class ClassEmbeddingEncoder(nn.Module):
    def __init__(self, num_classes=3, embed_tokens=8, embed_dim=768):
        super().__init__()
        self.num_classes = num_classes
        self.embed_tokens = embed_tokens
        self.table = nn.Parameter(torch.zeros(num_classes, embed_tokens, embed_dim))
        self.register_buffer("mask", torch.ones(num_classes, embed_tokens, 1, dtype=torch.float32))

    def forward(self, class_ids):
        class_ids = class_ids.clamp(0, self.num_classes - 1)
        emb = self.table[class_ids]
        return emb


class CardioContrastNet(nn.Module):
    def __init__(self, backbone, classifier, text_encoder="bert", bert_path=None, bert_trainable_layers=10, decode_with_lang=True, embed_tokens=8):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = text_encoder
        self.decode_with_lang = decode_with_lang
        self.bert_path = bert_path
        self.bert_trainable_layers = bert_trainable_layers
        self.embed_tokens = embed_tokens

        if text_encoder == "bert":
            from transformers import BertModel
            model = BertModel.from_pretrained(bert_path or "bert-base-uncased", add_pooling_layer=False)
            model.pooler = None
            self.bert = model
            self.bert.embeddings.requires_grad_(False)
            for block in self.bert.encoder.layer[bert_trainable_layers:]:
                block.requires_grad_(False)
        elif text_encoder == "embedding":
            self.bert = ClassEmbeddingEncoder(3, embed_tokens, 768)
        else:
            raise ValueError(f"Unknown text_encoder={text_encoder!r}")

    def encode_text(self, input_ids, attn_mask, class_ids=None):
        if self.text_encoder == "embedding":
            embeds = self.bert(class_ids)
            B, T, C = embeds.shape
            lang_feat = embeds.permute(0, 2, 1)
            lang_mask = torch.ones((B, T, 1), device=embeds.device, dtype=torch.long)
            return lang_feat, lang_mask
        outputs = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        lang_feat = outputs.last_hidden_state.permute(0, 2, 1)
        lang_mask = attn_mask.unsqueeze(-1)
        return lang_feat, lang_mask

    def forward(self, x, input_ids, attn_mask, class_ids, return_features=False):
        lang_feat, lang_mask = self.encode_text(input_ids, attn_mask, class_ids)
        if self.backbone is not None:
            feats = self.backbone(x)
            if isinstance(feats, (tuple, list)):
                x_c1, x_c2, x_c3, x_c4 = feats
            else:
                x_c1, x_c2, x_c3, x_c4 = feats
        else:
            x_c1 = x
            x_c2 = x
            x_c3 = x
            x_c4 = x

        lang_for_decoder = lang_feat if self.decode_with_lang else None
        mask_for_decoder = lang_mask if self.decode_with_lang else None
        logits, pre_logit_features = self.classifier(
            x_c4, x_c3, x_c2, x_c1,
            lang_feat=lang_for_decoder,
            lang_mask=mask_for_decoder,
            return_features=True,
        )
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=True)
        if return_features:
            return logits, pre_logit_features
        return logits


class LAVT(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask, return_features=False, decode_with_lang=False):
        feats = self.backbone(x)
        x_c1, x_c2, x_c3, x_c4 = feats
        lang_for_decoder = l_feats if decode_with_lang else None
        lang_mask = l_mask if decode_with_lang else None
        logits, features = self.classifier(x_c4, x_c3, x_c2, x_c1, lang_feat=lang_for_decoder, lang_mask=lang_mask, return_features=True)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=True)
        if return_features:
            return logits, features
        return logits


class LAVTOne(nn.Module):
    def __init__(self, backbone, classifier, args):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.args = args

    def forward(self, x, text, l_mask, return_features=False, decode_with_lang=False):
        from transformers import BertModel
        bert = BertModel.from_pretrained(self.args.ck_bert if hasattr(self.args, 'ck_bert') else 'bert-base-uncased')
        bert.pooler = None
        l_feats = bert(text, attention_mask=l_mask)[0].permute(0, 2, 1)
        l_mask = l_mask.unsqueeze(-1)
        feats = self.backbone(x)
        x_c1, x_c2, x_c3, x_c4 = feats
        logits, features = self.classifier(x_c4, x_c3, x_c2, x_c1, lang_feat=l_feats if decode_with_lang else None, lang_mask=l_mask if decode_with_lang else None, return_features=True)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=True)
        if return_features:
            return logits, features
        return logits
