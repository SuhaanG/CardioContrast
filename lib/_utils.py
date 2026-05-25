from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone   = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask, return_features=False):
        input_shape = x.shape[-2:]
        features    = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        if return_features:
            logits, pre_logit_features = self.classifier(
                x_c4, x_c3, x_c2, x_c1, return_features=True)
            logits = F.interpolate(logits, size=input_shape,
                                   mode='bilinear', align_corners=True)
            return logits, pre_logit_features
        else:
            x = self.classifier(x_c4, x_c3, x_c2, x_c1)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            return x


class LAVT(_LAVTSimpleDecode):
    pass


class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        self.backbone     = backbone
        self.classifier   = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

    def forward(self, x, text, l_mask, return_features=False):
        input_shape = x.shape[-2:]
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]
        l_feats = l_feats.permute(0, 2, 1)
        l_mask  = l_mask.unsqueeze(dim=-1)
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        if return_features:
            logits, pre_logit_features = self.classifier(
                x_c4, x_c3, x_c2, x_c1, return_features=True)
            logits = F.interpolate(logits, size=input_shape,
                                   mode='bilinear', align_corners=True)
            return logits, pre_logit_features
        else:
            x = self.classifier(x_c4, x_c3, x_c2, x_c1)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            return x


class LAVTOne(_LAVTOneSimpleDecode):
    pass
