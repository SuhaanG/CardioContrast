from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone   = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask,
                return_features=False, decode_with_lang=False):
        input_shape = x.shape[-2:]
        features    = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        lang_for_decoder = l_feats if decode_with_lang else None
        mask_for_decoder = l_mask  if decode_with_lang else None

        if return_features:
            logits, pre_logit_features = self.classifier(
                x_c4, x_c3, x_c2, x_c1,
                lang_feat=lang_for_decoder,
                lang_mask=mask_for_decoder,
                return_features=True)
            logits = F.interpolate(logits, size=input_shape,
                                   mode="bilinear", align_corners=True)
            return logits, pre_logit_features
        else:
            logits = self.classifier(
                x_c4, x_c3, x_c2, x_c1,
                lang_feat=lang_for_decoder,
                lang_mask=mask_for_decoder)
            return F.interpolate(logits, size=input_shape,
                                 mode="bilinear", align_corners=True)


class LAVT(_LAVTSimpleDecode):
    pass


class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        self.backbone     = backbone
        self.classifier   = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

    def forward(self, x, text, l_mask,
                return_features=False, decode_with_lang=False):
        input_shape = x.shape[-2:]

        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]
        l_feats = l_feats.permute(0, 2, 1)
        l_mask  = l_mask.unsqueeze(dim=-1)

        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        lang_for_decoder = l_feats if decode_with_lang else None
        mask_for_decoder = l_mask  if decode_with_lang else None

        if return_features:
            logits, pre_logit_features = self.classifier(
                x_c4, x_c3, x_c2, x_c1,
                lang_feat=lang_for_decoder,
                lang_mask=mask_for_decoder,
                return_features=True)
            logits = F.interpolate(logits, size=input_shape,
                                   mode="bilinear", align_corners=True)
            return logits, pre_logit_features
        else:
            logits = self.classifier(
                x_c4, x_c3, x_c2, x_c1,
                lang_feat=lang_for_decoder,
                lang_mask=mask_for_decoder)
            return F.interpolate(logits, size=input_shape,
                                 mode="bilinear", align_corners=True)


class LAVTOne(_LAVTOneSimpleDecode):
    pass
