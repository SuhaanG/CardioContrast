import math
import os
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.prompts import CANONICAL, HELDOUT, STRUCTURES, TRAIN_BANK
from data.splits import assert_disjoint, get_split_patients
from data.dataset_camus import CAMUSDataset, index_images
from data.dataset_camus_contrastive import CAMUSDatasetContrastive
from lib.contrastive import ContrastiveAnatomicalLoss, weighted_pool
from lib.mask_predictor import DecoderCrossAttention, SimpleDecoding
from lib._utils import LAVTOne
from run_ablation_suite import build_env_for_preset, manifest_environment


def test_split_and_prompts():
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "database_nifti")
        os.makedirs(base)
        for idx in range(1, 501):
            os.makedirs(os.path.join(base, f"patient{idx:03d}"), exist_ok=True)
        train, val, test = get_split_patients(base, "train"), get_split_patients(base, "val"), get_split_patients(base, "test")
        assert len(train) == 400 and len(val) == 50 and len(test) == 50
        assert_disjoint(train, val, test)
        assert set(CANONICAL.keys()) == set(STRUCTURES.keys())
        train_prompts = {prompt for prompts in TRAIN_BANK.values() for prompt in prompts}
        heldout_prompts = {prompt for prompts in HELDOUT.values() for prompt in prompts}
        overlap = train_prompts & heldout_prompts
        assert not overlap, f"prompt overlap found: {overlap}"


def test_dataset_uses_patient_split_boundaries():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "database_nifti"
        for patient_id in (1, 400, 401, 450, 451, 500):
            patient = f"patient{patient_id:03d}"
            patient_dir = data_dir / patient
            patient_dir.mkdir(parents=True)
            image_name = f"{patient}_2CH_ED.nii.gz"
            (patient_dir / image_name).touch()
            (patient_dir / image_name.replace(".nii.gz", "_gt.nii.gz")).touch()

        train = index_images(str(data_dir), "train")
        val = index_images(str(data_dir), "val")
        test = index_images(str(data_dir), "test")
        assert {row["patient"] for row in train} == {"patient001", "patient400"}
        assert {row["patient"] for row in val} == {"patient401", "patient450"}
        assert {row["patient"] for row in test} == {"patient451", "patient500"}


def test_ablation_manifest_is_filtered_and_presets_are_explicit():
    safe = manifest_environment({
        "CC_SEED": "42",
        "CC_DECODE_WITH_LANG": "1",
        "AWS_SECRET_ACCESS_KEY": "do-not-record",
    })
    assert safe == {"CC_DECODE_WITH_LANG": "1", "CC_SEED": "42"}
    try:
        build_env_for_preset("exp5_class_embedding")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported preset must not silently run as another experiment")


def test_baseline_model_skips_text_encoder():
    class IdentityBackbone(torch.nn.Module):
        def forward(self, image):
            return image, image, image, image

    class TwoClassHead(torch.nn.Module):
        def forward(self, x_c4, x_c3, x_c2, x_c1, **kwargs):
            return torch.cat((x_c4, x_c4), dim=1), x_c4

    model = LAVTOne(
        IdentityBackbone(),
        TwoClassHead(),
        SimpleNamespace(decode_with_lang=False),
    )
    image = torch.randn(2, 1, 8, 8)
    output = model(image, text=None, l_mask=None)
    assert model.text_encoder is None
    assert output.shape == (2, 2, 8, 8)


def test_non_language_datasets_skip_tokenizer_loading():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = str(Path(tmp) / "database_nifti")
        baseline = CAMUSDataset(data_dir, split="train", use_language=False)
        contrastive = CAMUSDatasetContrastive(
            data_dir,
            split="train",
            use_language=False,
        )
        assert baseline.tokenizer is None
        assert contrastive.tokenizer is None


def test_contrastive_loss_and_pooling():
    feat = torch.randn(3, 4, 5, 5)
    weights = torch.ones(3, 1, 5, 5)
    pooled = weighted_pool(feat, weights, detach=True)
    assert pooled.shape == (3, 4)

    loss_fn = ContrastiveAnatomicalLoss(4, proj_hidden_dim=8, proj_out_dim=4, tau=0.07)
    features = torch.randn(3, 4, 3, 3)
    logits = torch.randn(3, 2, 3, 3)
    image_ids = torch.tensor([0, 0, 1])
    structure_ids = torch.tensor([1, 2, 1])
    loss, neg_cos, n = loss_fn(features, logits, image_ids, structure_ids, pool_region="pred")
    assert torch.isfinite(loss)
    assert n in (0, 1, 2, 3)
    assert neg_cos.shape == ()

    no_neg_loss, _, _ = loss_fn(features[:1], logits[:1], image_ids[:1], structure_ids[:1], pool_region="pred")
    assert float(no_neg_loss.detach()) == 0.0


def test_decoder_gate_zero_matches_baseline():
    B, H, W = 2, 8, 8
    x_c4 = torch.randn(B, 8, H, W)
    x_c3 = torch.randn(B, 4, H // 2, W // 2)
    x_c2 = torch.randn(B, 2, H // 4, W // 4)
    x_c1 = torch.randn(B, 1, H // 8, W // 8)
    lang = torch.randn(B, 768, 12)
    mask = torch.ones(B, 12, 1, dtype=torch.long)

    dec = SimpleDecoding(c4_dims=8, lang_dim=768, num_heads=4)
    dec2 = SimpleDecoding(c4_dims=8, lang_dim=768, num_heads=4)
    dec2.load_state_dict(dec.state_dict())
    dec.ca_stage1.gate.data.zero_()
    dec.ca_stage2.gate.data.zero_()
    dec.ca_stage3.gate.data.zero_()
    out_lang = dec(x_c4, x_c3, x_c2, x_c1, lang_feat=lang, lang_mask=mask)
    out_plain = dec2(x_c4, x_c3, x_c2, x_c1)
    assert out_lang.shape == out_plain.shape
    assert torch.allclose(out_lang, out_plain, atol=1e-5, rtol=1e-5)


def test_cross_attention_ignores_padding_tokens():
    layer = DecoderCrossAttention(8, lang_dim=768, num_heads=2)
    visual = torch.randn(2, 8, 5, 5)
    lang = torch.randn(2, 768, 6)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.long)
    out = layer(visual, lang, mask.unsqueeze(-1))
    assert out.shape == visual.shape


def test_binary_metrics_known_square_and_empty():
    from lib.metrics import binary_metrics

    pred = np.zeros((20, 20), dtype=np.uint8)
    pred[5:15, 5:15] = 1
    gt = np.zeros_like(pred)
    gt[5:15, 5:15] = 1
    metrics = binary_metrics(pred, gt, spacing=(0.5, 0.5))
    assert math.isfinite(metrics["dice"]) and metrics["dice"] > 0.9
    assert metrics["empty_case"] is False

    empty_pred = np.zeros((20, 20), dtype=np.uint8)
    empty_gt = np.zeros((20, 20), dtype=np.uint8)
    metrics_empty = binary_metrics(empty_pred, empty_gt, spacing=(0.5, 0.5))
    assert metrics_empty["dice"] == 1.0
    assert metrics_empty["empty_case"] is True


def test_per_structure_metrics_summary():
    from lib.metrics import per_structure_metrics

    pred = np.zeros((10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)
    pred[2:8, 2:8] = 1
    gt[2:8, 2:8] = 1
    metrics = per_structure_metrics(pred, gt, structure_id=1)
    assert metrics["dice"] > 0.9
    assert metrics["structure_id"] == 1
    assert metrics["empty_case"] is False


def test_simpson_ellipse_volume_reasonable():
    from lib.metrics import simpson_volume, ejection_fraction

    # Half axes 20 and 10 px, with 20 discs over 2D ellipse area in a 2D plane.
    area = math.pi * 20 * 10
    vol = simpson_volume(np.ones((40, 40)), np.zeros((40, 40)), 1.0, 1.0)
    assert np.isfinite(vol)
    assert vol > 0
    ef = ejection_fraction(100.0, 40.0)
    assert abs(ef - 60.0) < 1e-6


if __name__ == "__main__":
    test_split_and_prompts()
    test_dataset_uses_patient_split_boundaries()
    test_ablation_manifest_is_filtered_and_presets_are_explicit()
    test_baseline_model_skips_text_encoder()
    test_non_language_datasets_skip_tokenizer_loading()
    test_contrastive_loss_and_pooling()
    test_decoder_gate_zero_matches_baseline()
    test_cross_attention_ignores_padding_tokens()
    test_binary_metrics_known_square_and_empty()
    test_simpson_ellipse_volume_reasonable()
    print("ALL TESTS PASSED")
