import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from data.dataset_camus import index_images
from lib.report import Reporter


def _case_id(record):
    return f"{record['patient']}_{record['view']}_{record['phase']}"


def _read_prediction(path, shape):
    if path.suffix.lower() == ".png":
        prediction = np.asarray(Image.open(path))
    elif path.name.lower().endswith((".nii", ".nii.gz")):
        import nibabel as nib

        prediction = np.squeeze(np.asarray(nib.load(path).get_fdata()))
    else:
        raise ValueError(f"Unsupported prediction format: {path}")
    prediction = np.squeeze(prediction)
    if prediction.shape != shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match native mask shape {shape}: {path}")
    return np.rint(prediction).astype(np.uint8)


def evaluate_nnunet(pred_dir, data_dir=None, output_dir=None):
    pred_dir = Path(pred_dir)
    data_dir = Path(data_dir or config.CAMUS_DATA_DIR)
    output_dir = Path(output_dir or pred_dir / "report")
    reporter = Reporter()
    for record in index_images(str(data_dir), "test"):
        import nibabel as nib

        gt_full = np.squeeze(np.asarray(nib.load(record["mask_path"]).get_fdata()))
        prediction_path = next((path for path in (
            pred_dir / f"{_case_id(record)}.png",
            pred_dir / f"{_case_id(record)}.nii.gz",
            pred_dir / f"{_case_id(record)}.nii",
        ) if path.is_file()), None)
        if prediction_path is None:
            raise FileNotFoundError(f"No multi-label prediction found for {_case_id(record)} in {pred_dir}")
        predicted_labels = _read_prediction(prediction_path, gt_full.shape)
        spacing = data_spacing(record["mask_path"])
        for structure_id in (1, 2, 3):
            reporter.add(
                patient=record["patient"], view=record["view"], phase=record["phase"],
                quality=record["quality"],
                ref_ef=float("nan") if record["ref_ef"] is None else record["ref_ef"],
                structure=structure_id, prompt="nnU-Net multi-label output",
                pred=predicted_labels == structure_id, gt_full=gt_full, spacing=spacing,
            )
    return reporter.write(output_dir, meta={
        "baseline": "nnU-Net v2",
        "split": "test",
        "prediction_dir": str(pred_dir.resolve()),
    })


def data_spacing(mask_path):
    import nibabel as nib

    image = nib.load(mask_path)
    zooms = image.header.get_zooms()
    unit = image.header.get_xyzt_units()[0]
    from data.dataset_camus import spacing_to_mm

    return spacing_to_mm(zooms, unit, assumed_unit=config.SPACING_UNIT)


def main():
    parser = argparse.ArgumentParser(description="Score nnU-Net multi-label CAMUS predictions through Reporter.")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    summary = evaluate_nnunet(args.pred_dir, args.data_dir, args.output_dir)
    print(json.dumps(summary["clinical"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
