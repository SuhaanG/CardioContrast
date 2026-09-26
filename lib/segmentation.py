import warnings

from .backbone import MultiModalSwinTransformer
from .mask_predictor import SimpleDecoding
from ._utils import CardioContrastNet


def decoder_hidden_size(swin_type):
    embed_dims = {"tiny": 96, "small": 96, "base": 128, "large": 192}
    if swin_type not in embed_dims:
        raise ValueError(f"Unsupported Swin type: {swin_type}")
    return embed_dims[swin_type] * 4


def build_model(swin_type="base", pretrained_swin="", window_size=7,
                text_encoder="bert", bert_path="bert-base-uncased",
                bert_trainable_layers=10, decode_with_lang=True,
                embed_tokens=8, warn_random_init=True):
    configs = {
        "tiny": (96, [2, 2, 6, 2], [3, 6, 12, 24]),
        "small": (96, [2, 2, 18, 2], [3, 6, 12, 24]),
        "base": (128, [2, 2, 18, 2], [4, 8, 16, 32]),
        "large": (192, [2, 2, 18, 2], [6, 12, 24, 48]),
    }
    if swin_type not in configs:
        raise ValueError(f"Unsupported Swin type: {swin_type}")
    if int(window_size) < 1:
        raise ValueError("window_size must be positive")

    embed_dim, depths, heads = configs[swin_type]
    backbone = MultiModalSwinTransformer(
        embed_dim=embed_dim,
        depths=depths,
        num_heads=heads,
        window_size=int(window_size),
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,
        num_heads_fusion=[1, 1, 1, 1],
        fusion_drop=0.0,
    )
    if pretrained_swin:
        backbone.init_weights(pretrained=str(pretrained_swin))
    else:
        if warn_random_init:
            warnings.warn(
                "PRETRAINED_SWIN is empty; the Swin backbone is randomly initialized.",
                RuntimeWarning,
                stacklevel=2,
            )
        backbone.init_weights()

    classifier = SimpleDecoding(2 * decoder_hidden_size(swin_type))
    return CardioContrastNet(
        backbone=backbone,
        classifier=classifier,
        text_encoder=text_encoder,
        bert_path=bert_path,
        bert_trainable_layers=bert_trainable_layers,
        decode_with_lang=decode_with_lang,
        embed_tokens=embed_tokens,
    )


__all__ = ["build_model", "decoder_hidden_size"]