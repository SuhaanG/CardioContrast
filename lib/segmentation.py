"""
lib/segmentation.py — Model builder.
"""

from .backbone import MultiModalSwinTransformer
from .mask_predictor import SimpleDecoding
from ._utils import CardioContrastNet

SWIN_CFG = {
    "tiny":  dict(embed_dim=96,  depths=[2, 2, 6, 2],  num_heads=[3, 6, 12, 24]),
    "small": dict(embed_dim=96,  depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
    "base":  dict(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    "large": dict(embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48]),
}


def build_model(swin_type="base", pretrained_swin="", window_size=12, text_encoder="bert",
                bert_path="bert-base-uncased", bert_trainable_layers=10,
                decode_with_lang=True, embed_tokens=8, drop_path_rate=0.3):
    cfg = SWIN_CFG[swin_type]
    backbone = MultiModalSwinTransformer(
        embed_dim=cfg["embed_dim"], depths=cfg["depths"], num_heads=cfg["num_heads"],
        window_size=window_size, ape=False, drop_path_rate=drop_path_rate, patch_norm=True,
        out_indices=(0, 1, 2, 3), use_checkpoint=False, num_heads_fusion=[1, 1, 1, 1],
        fusion_drop=0.0)
    if pretrained_swin:
        print("[model] Initializing Swin-{} from {}".format(swin_type, pretrained_swin), flush=True)
        backbone.init_weights(pretrained=pretrained_swin)
    else:
        print("[model] WARNING: Swin initialised randomly (no pretrained weights).", flush=True)
        backbone.init_weights()

    classifier = SimpleDecoding(8 * cfg["embed_dim"])
    return CardioContrastNet(backbone, classifier, text_encoder=text_encoder,
                             bert_path=bert_path, bert_trainable_layers=bert_trainable_layers,
                             decode_with_lang=decode_with_lang, embed_tokens=embed_tokens)


def decoder_hidden_size(swin_type):
    return (8 * SWIN_CFG[swin_type]["embed_dim"]) // 2
