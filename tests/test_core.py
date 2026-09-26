import math
import sys
import tempfile
from argparse import Namespace
import csv
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset_camus import (
    CAMUSDataset,
    index_images,
    preprocess_grayscale_image,
    spacing_to_mm,
)
from data.prompts import CANONICAL, HELDOUT, STRUCTURES, TRAIN_BANK
from data.samplers import GroupedStructureBatchSampler
from data.splits import assert_disjoint, get_split_patients
from compare_runs import compare_runs
from evaluate_echonet import build_tracing_masks, orient
from lib._utils import CardioContrastNet
from lib.contrastive import ContrastiveAnatomicalLoss, weighted_pool
from lib.mask_predictor import DecoderCrossAttention, SimpleDecoding
from lib.metrics import binary_metrics, leakage, per_structure_metrics, simpson_volume, ejection_fraction
from lib.paper_results import aggregate_patient_metrics
from lib.report import Reporter
from run_ablation_suite import build_env_for_preset, manifest_environment
from train_cardiocontrast import build_optimizer, resolve_args, segmentation_loss
from tools.make_table import make_tables
import config


def test_split_and_prompt_protocol():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "database_nifti"
        data_dir.mkdir()
        for patient_id in range(1, 501):
            (data_dir / f"patient{patient_id:03d}").mkdir()
        train = get_split_patients(str(data_dir), "train")
        val = get_split_patients(str(data_dir), "val")
        test = get_split_patients(str(data_dir), "test")
        assert (len(train), len(val), len(test)) == (400, 50, 50)
        assert_disjoint(train, val, test)
        try:
            get_split_patients(str(data_dir), "testing_typo")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid split names must not fall back to test")

    assert set(CANONICAL) == set(STRUCTURES)
    train_prompts = {prompt for bank in TRAIN_BANK.values() for prompt in bank}
    heldout_prompts = {prompt for bank in HELDOUT.values() for prompt in bank}
    assert not train_prompts & heldout_prompts
    assert all(len(bank) >= 6 for bank in TRAIN_BANK.values())
    assert all(len(bank) == 3 for bank in HELDOUT.values())


def test_dataset_split_metadata_and_heldout_prompts():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "database_nifti"
        for patient_id in (1, 400, 401, 450, 451, 500):
            patient = f"patient{patient_id:03d}"
            patient_dir = data_dir / patient
            patient_dir.mkdir(parents=True)
            (patient_dir / "Info_2CH.cfg").write_text(
                "ImageQuality: Good\nLVef: 57.5\n", encoding="utf-8"
            )
            image = f"{patient}_2CH_ED.nii.gz"
            (patient_dir / image).touch()
            (patient_dir / image.replace(".nii.gz", "_gt.nii.gz")).touch()

        train = index_images(str(data_dir), "train")
        val = index_images(str(data_dir), "val")
        test = index_images(str(data_dir), "test")
        assert {record["patient"] for record in train} == {"patient001", "patient400"}
        assert {record["patient"] for record in val} == {"patient401", "patient450"}
        assert {record["patient"] for record in test} == {"patient451", "patient500"}
        assert test[0]["quality"] == "Good"
        assert test[0]["ref_ef"] == 57.5
        assert test[0]["case_id"] == "patient451_2CH"

        heldout = CAMUSDataset(
            str(data_dir), split="test", eval_prompt_set="heldout", use_language=False
        )
        assert len(heldout.samples) == 2 * 3 * 3
        samples = [s for s in heldout.samples if s["image_idx"] == 0 and s["structure"] == 1]
        resolved = [heldout._resolve_prompt(1, 0, s["prompt_k"]) for s in samples]
        assert resolved == HELDOUT[1]


def test_dataset_augmentation_shared_and_seeded_by_epoch():
    with tempfile.TemporaryDirectory() as tmp:
        dataset = CAMUSDataset(tmp, split="train", use_language=False, seed=27)
        image = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (24, 1))
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[5:18, 8:23] = 2
        first_image, first_mask = dataset._augment_image_and_mask(image, mask, 4)
        repeated_image, repeated_mask = dataset._augment_image_and_mask(image, mask, 4)
        assert np.array_equal(first_image, repeated_image)
        assert np.array_equal(first_mask, repeated_mask)
        assert set(np.unique(first_mask)).issubset({0, 2})
        dataset.set_epoch(1)
        next_image, next_mask = dataset._augment_image_and_mask(image, mask, 4)
        assert not np.array_equal(first_image, next_image)
        assert not np.array_equal(first_mask, next_mask)


def test_grouped_batch_sampler_contract():
    class SampleSet:
        samples = [
            {"image_idx": image_id, "structure": structure_id}
            for image_id in range(5)
            for structure_id in (1, 2, 3)
        ]

        def __len__(self):
            return len(self.samples)

    dataset = SampleSet()
    sampler = GroupedStructureBatchSampler(dataset, images_per_batch=2, seed=15)
    epoch_zero = list(sampler)
    assert epoch_zero == list(sampler)
    assert len(sampler) == 2
    for batch in epoch_zero:
        assert len(batch) == 6
        samples = [dataset.samples[index] for index in batch]
        for image_id in {sample["image_idx"] for sample in samples}:
            assert {s["structure"] for s in samples if s["image_idx"] == image_id} == {1, 2, 3}
    sampler.set_epoch(1)
    assert list(sampler) != epoch_zero


def test_contrastive_pooling_and_loss():
    features = torch.randn(3, 4, 5, 5)
    pooled = weighted_pool(features, torch.ones(3, 1, 5, 5))
    assert pooled.shape == (3, 4)

    loss_fn = ContrastiveAnatomicalLoss(4, proj_hidden_dim=8, proj_out_dim=4, tau=0.07)
    features = torch.randn(3, 4, 3, 3, requires_grad=True)
    logits = torch.randn(3, 2, 3, 3, requires_grad=True)
    image_ids = torch.tensor([0, 0, 1])
    structures = torch.tensor([1, 2, 1])
    loss, neg_cos, anchors = loss_fn(features, logits, image_ids, structures, pool_region="pred")
    assert torch.isfinite(loss) and neg_cos.shape == () and anchors == 2
    loss.backward()
    assert logits.grad is None

    masks = torch.zeros(3, 3, 3)
    masks[:, 1, 1] = 1
    for region in ("gt", "union"):
        region_loss, _, count = loss_fn(
            features.detach(), logits.detach(), image_ids, structures,
            pool_region=region, gt_masks=masks, union_masks=torch.ones_like(masks),
        )
        assert torch.isfinite(region_loss) and count == 2

    no_negative_features = torch.randn(1, 4, 3, 3, requires_grad=True)
    no_negative, _, count = loss_fn(
        no_negative_features, logits[:1].detach(), image_ids[:1], structures[:1]
    )
    assert no_negative.item() == 0.0 and count == 0
    no_negative.backward()
    assert no_negative_features.grad is not None


def test_decoder_zero_gates_and_padding_mask():
    batch, height, width = 2, 8, 8
    pyramid = (
        torch.randn(batch, 8, height, width),
        torch.randn(batch, 4, height // 2, width // 2),
        torch.randn(batch, 2, height // 4, width // 4),
        torch.randn(batch, 1, height // 8, width // 8),
    )
    lang = torch.randn(batch, 768, 8)
    mask = torch.ones(batch, 8, 1, dtype=torch.long)
    conditioned = SimpleDecoding(c4_dims=8, lang_dim=768, num_heads=4)
    baseline = SimpleDecoding(c4_dims=8, lang_dim=768, num_heads=4)
    baseline.load_state_dict(conditioned.state_dict())
    for stage in (conditioned.ca_stage1, conditioned.ca_stage2, conditioned.ca_stage3):
        stage.gate.data.zero_()
    assert torch.allclose(
        conditioned(*pyramid, lang_feat=lang, lang_mask=mask), baseline(*pyramid),
        atol=1e-5, rtol=1e-5,
    )

    layer = DecoderCrossAttention(8, lang_dim=768, num_heads=2).eval()
    visual = torch.randn(2, 8, 5, 5)
    tokens = torch.randn(2, 768, 6)
    token_mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]])
    changed_padding = tokens.clone()
    changed_padding[:, :, 2:] = torch.randn_like(changed_padding[:, :, 2:]) * 1e5
    out1 = layer(visual, tokens, token_mask.unsqueeze(-1))
    out2 = layer(visual, changed_padding, token_mask.unsqueeze(-1))
    assert torch.allclose(out1, out2, atol=1e-5, rtol=1e-5)


def test_cardio_model_class_embedding_forward():
    class Backbone(torch.nn.Module):
        def forward(self, image, lang_feat, lang_mask):
            assert lang_feat.shape == (2, 768, 8)
            assert lang_mask.shape == (2, 8, 1)
            return image, image, image, image

    class Head(torch.nn.Module):
        def forward(self, x_c4, x_c3, x_c2, x_c1, lang_feat=None,
                    lang_mask=None, return_features=False):
            assert lang_feat is None and lang_mask is None
            output = torch.cat((x_c4, x_c4), dim=1)
            return (output, x_c4) if return_features else output

    model = CardioContrastNet(
        Backbone(), Head(), text_encoder="embedding", decode_with_lang=False,
        embed_tokens=8,
    )
    output = model(
        torch.randn(2, 1, 8, 8), torch.zeros(2, 4, dtype=torch.long),
        torch.ones(2, 4, dtype=torch.long), torch.tensor([0, 2]),
    )
    assert output.shape == (2, 2, 8, 8)


def test_metric_shapes_empty_leakage_and_spacing():
    pred = np.zeros((40, 40), dtype=np.uint8)
    gt = np.zeros_like(pred)
    pred[10:20, 10:20] = 1
    gt[10:20, 15:25] = 1
    metrics = binary_metrics(pred, gt, spacing=(0.5, 0.5))
    assert metrics["dice"] == 0.5
    assert abs(metrics["hd"] - 2.5) < 1e-6

    wide_left = np.zeros((40, 70), dtype=np.uint8)
    wide_right = np.zeros_like(wide_left)
    wide_left[10:30, 5:45] = 1
    wide_right[10:30, 15:55] = 1
    shifted = binary_metrics(wide_left, wide_right, spacing=(0.5, 0.5))
    assert abs(shifted["dice"] - 0.75) < 1e-6
    assert abs(shifted["hd"] - 5.0) < 1e-6

    unequal_contour = np.zeros_like(gt)
    unequal_contour[8:22, 15:25] = 1
    unequal_metrics = binary_metrics(pred, unequal_contour, spacing=(0.5, 0.5))
    assert np.isfinite(unequal_metrics["hd95"])

    empty = binary_metrics(np.zeros_like(gt), np.zeros_like(gt), (0.5, 0.5))
    assert empty["dice"] == 1 and empty["empty_case"]
    one_empty = binary_metrics(pred, np.zeros_like(gt), (0.5, 0.5))
    assert one_empty["dice"] == 0 and one_empty["empty_case"]
    full = np.zeros_like(gt)
    full[10:20, 10:20] = 1
    full[20:23, 10:20] = 2
    predicted = np.zeros_like(full)
    predicted[10:20, 10:20] = 1
    predicted[20:23, 10:20] = 1
    assert abs(leakage(predicted, full, 1) - 0.23076923076923078) < 1e-6
    structure_metrics = per_structure_metrics(pred, gt, structure_id=1, spacing=(0.5, 0.5))
    assert structure_metrics["structure_id"] == 1

    assert spacing_to_mm((0.002, 0.003), "meter") == (2.0, 3.0)
    assert spacing_to_mm((0.5, 0.6), "unknown", assumed_unit="mm") == (0.5, 0.6)


def test_preprocessing_and_simpson_metrics():
    image = preprocess_grayscale_image(np.full((3, 4), 255, dtype=np.uint8), 2)
    expected = torch.tensor([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225])
    assert image.shape == (3, 2, 2)
    assert torch.allclose(image[:, 0, 0], expected, atol=1e-6)

    rows, columns = np.ogrid[:101, :101]
    ellipse = ((columns - 50) ** 2 / 40 ** 2 + (rows - 50) ** 2 / 20 ** 2) <= 1
    la = np.zeros_like(ellipse)
    la[45:56, 2:10] = 1
    analytic_ml = 4.0 / 3.0 * math.pi * 40 * 20 * 20 / 1000.0
    volume_contact = simpson_volume(ellipse, ellipse, la, la)
    volume_fallback = simpson_volume(ellipse, ellipse)
    assert abs(volume_contact - analytic_ml) / analytic_ml < 0.05
    assert abs(volume_fallback - analytic_ml) / analytic_ml < 0.10
    assert simpson_volume(np.zeros_like(ellipse), ellipse) == 0.0
    assert abs(ejection_fraction(100.0, 40.0) - 60.0) < 1e-6


def test_patient_aggregates_and_report_outputs():
    rows = [
        {"patient": "p1", "structure_id": 1, "dice": 0.2, "iou": 0.1},
        {"patient": "p1", "structure_id": 1, "dice": 0.4, "iou": 0.3},
        {"patient": "p2", "structure_id": 1, "dice": 0.8, "iou": 0.7},
        {"patient": "p3", "structure_id": 1, "dice": 0.9, "iou": 0.8},
    ]
    aggregated = aggregate_patient_metrics(rows, n_bootstrap=100, seed=7)
    assert len(aggregated["patient_metrics"]) == 3
    assert abs(aggregated["patient_metrics"][0]["dice"] - 0.3) < 1e-8

    reporter = Reporter()
    height, width = 80, 100
    y, x = np.ogrid[:height, :width]
    for phase, radius_x, radius_y in (("ED", 30, 16), ("ES", 22, 12)):
        lv = ((x - 50) ** 2 / radius_x ** 2 + (y - 40) ** 2 / radius_y ** 2) <= 1
        la = np.zeros((height, width), dtype=bool)
        la[35:46, 12:20] = True
        gt_full = np.zeros((height, width), dtype=np.uint8)
        gt_full[lv] = 1
        gt_full[la] = 3
        for view in ("2CH", "4CH"):
            for structure, prediction in ((1, lv), (3, la)):
                reporter.add("p1", view, phase, "Good", 58.0,
                             structure, "prompt", prediction, gt_full, (1.0, 1.0))
    with tempfile.TemporaryDirectory() as tmp:
        summary = reporter.write(tmp, {"experiment": "synthetic"})
        for filename in ("per_sample.csv", "per_patient.csv", "clinical.csv", "summary.json"):
            assert (Path(tmp) / filename).is_file()
        assert summary["clinical"]["gt_ef_vs_cfg"]["n"] == 1


def test_training_presets_and_optimizer_coverage():
    parsed = Namespace(**{
        key: None for key in (
            "seed", "epochs", "img_size", "window_size", "swin_type", "pretrained_swin",
            "data_dir", "output_root", "bert_path", "bert_trainable_layers", "embed_tokens",
            "decode_with_lang", "contrastive_weight", "contrastive_tau", "pool_region",
            "text_encoder", "prompt_mode", "images_per_batch", "grad_accum_steps",
            "learning_rate", "weight_decay", "max_images", "device", "resume",
        )
    })
    parsed.workers = 0
    for preset in config.PRESETS:
        parsed.preset = preset
        resolved = resolve_args(parsed)
        assert resolved["exp_name"] == preset

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(3, 2)
            self.norm = torch.nn.LayerNorm(2)
            self.gate = torch.nn.Parameter(torch.zeros(1))

    model = ToyModel()
    contrastive = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(model, contrastive, {"learning_rate": 5e-5, "weight_decay": 0.01})
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    expected = list(model.parameters()) + list(contrastive.parameters())
    assert len(grouped) == len({id(p) for p in grouped})
    assert {id(p) for p in grouped} == {id(p) for p in expected}
    gates = next(group for group in optimizer.param_groups if group["group_name"] == "gates")
    assert gates["lr"] == 5e-3
    assert torch.isfinite(segmentation_loss(torch.randn(2, 2, 8, 8), torch.zeros(2, 8, 8, dtype=torch.long)))


def test_ablation_manifest_filters_environment_and_supports_seven_presets():
    safe = manifest_environment({"CC_SEED": "42", "CC_DECODE_WITH_LANG": "1", "SECRET": "omit"})
    assert safe == {"CC_DECODE_WITH_LANG": "1", "CC_SEED": "42"}
    assert build_env_for_preset("exp5_class_embedding")["CC_DECODE_WITH_LANG"] == "1"
    try:
        build_env_for_preset("invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown experiment must fail")


def test_patient_statistics_and_paper_tables():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        side_a = base / "condition_a_seed42" / "eval_test_canonical"
        side_b = base / "condition_b_seed42" / "eval_test_canonical"
        for directory, increment in ((side_a, 0.0), (side_b, 0.1)):
            directory.mkdir(parents=True)
            with (directory / "per_patient.csv").open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=(
                    "patient", "structure_id", "structure", "dice", "hd95", "mad", "leakage"
                ))
                writer.writeheader()
                for patient_index in range(4):
                    writer.writerow({
                        "patient": f"p{patient_index}",
                        "structure_id": 1,
                        "structure": "lv_endo",
                        "dice": 0.6 + patient_index * 0.02 + increment,
                        "hd95": 3.0 - increment,
                        "mad": 1.0,
                        "leakage": 0.05,
                    })
            (directory / "clinical.csv").write_text(
                "patient,pred_ef_percent,gt_ef_percent\np0,55,60\np1,65,70\n",
                encoding="utf-8",
            )

        result = compare_runs([str(side_a.parent)], [str(side_b.parent)], min_patients=3, n_bootstrap=200)
        assert result["status"] == "ok"
        assert all(row["n_patients"] == 4 for row in result["results"])
        assert all("holm_p" in row for row in result["results"])
        insufficient = compare_runs([str(side_a.parent)], [str(side_b.parent)], min_patients=5, n_bootstrap=100)
        assert insufficient["results"] == []
        assert insufficient["status"] == "no comparisons met min_patients"

        markdown, latex = make_tables([f"CardioContrast={base / 'condition_*'}"])
        assert "CardioContrast" in markdown
        assert "\\begin{tabular}" in latex


def test_echonet_polygon_and_orientation_helpers():
    rows = [{"Frame": 1, "X1": 1, "Y1": 1, "X2": 1, "Y2": 1}]
    for x_left, x_right, y in ((20, 40, 20), (20, 40, 40), (20, 40, 60), (20, 40, 80)):
        rows.append({"Frame": 1, "X1": x_left, "Y1": y, "X2": x_right, "Y2": y})
    for x_left, x_right, y in ((25, 35, 30), (25, 35, 40), (25, 35, 50), (25, 35, 60)):
        rows.append({"Frame": 2, "X1": x_left, "Y1": y, "X2": x_right, "Y2": y})
    masks = build_tracing_masks(rows)
    assert set(masks) == {1, 2}
    assert masks[1].shape == (112, 112)
    assert masks[1].sum() > masks[2].sum()

    array = np.arange(9).reshape(3, 3)
    assert np.array_equal(orient(array, transpose=True), array.T)
    assert np.array_equal(orient(array, flip_ud=True), np.flipud(array))


if __name__ == "__main__":
    test_split_and_prompt_protocol()
    test_dataset_split_metadata_and_heldout_prompts()
    test_dataset_augmentation_shared_and_seeded_by_epoch()
    test_grouped_batch_sampler_contract()
    test_contrastive_pooling_and_loss()
    test_decoder_zero_gates_and_padding_mask()
    test_cardio_model_class_embedding_forward()
    test_metric_shapes_empty_leakage_and_spacing()
    test_preprocessing_and_simpson_metrics()
    test_patient_aggregates_and_report_outputs()
    test_training_presets_and_optimizer_coverage()
    test_ablation_manifest_filters_environment_and_supports_seven_presets()
    test_patient_statistics_and_paper_tables()
    test_echonet_polygon_and_orientation_helpers()
    print("ALL TESTS PASSED")
