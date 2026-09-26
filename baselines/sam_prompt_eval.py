import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from data.dataset_camus import index_images, spacing_to_mm
from data.prompts import STRUCTURES
from lib.report import Reporter


def _load_nifti(path):
    import nibabel as nib

    image = nib.load(path)
    return np.squeeze(np.asarray(image.get_fdata())), image


def _rgb8(image):
    image = np.asarray(image, dtype=np.float32)
    maximum = float(image.max()) if image.size else 0.0
    if maximum > 0:
        image = image / maximum * 255.0
    gray = np.clip(image, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _box_prompt(mask, rng):
    rows, columns = np.where(mask)
    if not len(rows):
        return None
    height, width = mask.shape
    x0, x1 = float(columns.min()), float(columns.max())
    y0, y1 = float(rows.min()), float(rows.max())
    box_width, box_height = max(1.0, x1 - x0), max(1.0, y1 - y0)
    jitter_x = rng.uniform(-0.05, 0.05, size=2) * box_width
    jitter_y = rng.uniform(-0.05, 0.05, size=2) * box_height
    return np.asarray((
        np.clip(x0 - jitter_x[0], 0, width - 1),
        np.clip(y0 - jitter_y[0], 0, height - 1),
        np.clip(x1 + jitter_x[1], 0, width - 1),
        np.clip(y1 + jitter_y[1], 0, height - 1),
    ), dtype=np.float32)


def _point_prompt(mask, full_mask, structure_id, spacing, prompt_type):
    from scipy import ndimage

    rows, columns = np.where(mask)
    if not len(rows):
        return None
    if prompt_type == "point_center":
        distance = ndimage.distance_transform_edt(mask, sampling=(spacing[1], spacing[0]))
        row, column = np.unravel_index(np.argmax(distance), distance.shape)
    else:
        other = (full_mask > 0) & (full_mask != structure_id)
        if not other.any():
            return None
        distance_to_other = ndimage.distance_transform_edt(~other, sampling=(spacing[1], spacing[0]))
        distances = distance_to_other[rows, columns]
        eligible = np.flatnonzero(distances <= 3.0)
        if not len(eligible):
            return None
        selected = eligible[np.argmin(distances[eligible])]
        row, column = rows[selected], columns[selected]
    return np.asarray([[float(column), float(row)]], dtype=np.float32)


def evaluate_sam(weights, model_type="vit_b", variant="sam", prompt_type="box",
                 data_dir=None, output_dir=None, device=None, seed=42):
    if prompt_type not in {"box", "point_center", "point_boundary"}:
        raise ValueError(f"Unsupported SAM prompt type: {prompt_type}")
    if variant == "medsam" and prompt_type != "box":
        raise ValueError("MedSAM evaluation in this workflow supports box prompts only")
    from segment_anything import SamPredictor, sam_model_registry

    if device is None:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = sam_model_registry[model_type](checkpoint=str(weights))
    model.to(device=device)
    predictor = SamPredictor(model)
    records = index_images(data_dir or config.CAMUS_DATA_DIR, "test")
    if not records:
        raise ValueError("CAMUS test split has no indexed images")
    reporter = Reporter()
    rng = np.random.default_rng(seed)

    for record in records:
        image, _ = _load_nifti(record["image_path"])
        full_mask, nifti = _load_nifti(record["mask_path"])
        spacing = spacing_to_mm(
            nifti.header.get_zooms(), nifti.header.get_xyzt_units()[0],
            assumed_unit=config.SPACING_UNIT,
        )
        predictor.set_image(_rgb8(image))
        ref_ef = float("nan") if record["ref_ef"] is None else record["ref_ef"]

        for structure_id, structure_name in STRUCTURES.items():
            target = full_mask == structure_id
            if prompt_type == "box":
                box = _box_prompt(target, rng)
                if box is None:
                    prediction = np.zeros_like(target)
                else:
                    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
                    prediction = masks[int(np.argmax(scores))]
            else:
                point = _point_prompt(target, full_mask, structure_id, spacing, prompt_type)
                if point is None:
                    prediction = np.zeros_like(target)
                else:
                    masks, scores, _ = predictor.predict(
                        point_coords=point,
                        point_labels=np.ones(len(point), dtype=np.int32),
                        multimask_output=True,
                    )
                    prediction = masks[int(np.argmax(scores))]
            reporter.add(
                patient=record["patient"], view=record["view"], phase=record["phase"],
                quality=record["quality"], ref_ef=ref_ef,
                structure=structure_id, prompt=f"{variant}:{prompt_type}",
                pred=prediction, gt_full=full_mask, spacing=spacing,
            )
        predictor.reset_image()

    if output_dir is None:
        output_dir = Path(config.CC_OUTPUT_ROOT) / "baselines" / f"{variant}_{prompt_type}"
    return reporter.write(output_dir, meta={
        "baseline": variant,
        "prompt_type": prompt_type,
        "split": "test",
        "checkpoint": str(Path(weights).resolve()),
        "model_type": model_type,
    })


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM or MedSAM prompts through the shared Reporter.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model_type", default="vit_b", choices=("vit_b", "vit_l", "vit_h"))
    parser.add_argument("--variant", choices=("sam", "medsam"), default="sam")
    parser.add_argument("--prompt_type", choices=("box", "point_center", "point_boundary"), default="box")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = evaluate_sam(
        args.weights, args.model_type, args.variant, args.prompt_type,
        args.data_dir, args.output_dir, args.device, args.seed,
    )
    print(json.dumps(summary["clinical"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
