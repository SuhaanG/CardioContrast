"""Export CAMUS PNGs for nnU-Net v2.

Example after export:
  nnUNetv2_plan_and_preprocess -d 501 -c 2d
  nnUNetv2_train 501 2d 0
  nnUNetv2_predict -i <dataset>/imagesTs -o <predictions> -d 501 -c 2d -f 0
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset_camus import index_images


def _case_id(record):
    return f"{record['patient']}_{record['view']}_{record['phase']}"


def _read_nifti(path):
    import nibabel as nib

    return np.squeeze(np.asarray(nib.load(path).get_fdata()))


def _save_image(image, path):
    image = np.asarray(image, dtype=np.float32)
    maximum = float(image.max()) if image.size else 0.0
    if maximum > 0:
        image = image / maximum * 255.0
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="L").save(path)


def export_dataset(data_dir, output_root, dataset_id=501, dataset_name="CardioContrast"):
    root = Path(output_root) / f"Dataset{int(dataset_id):03d}_{dataset_name}"
    subsets = {
        split: index_images(data_dir, split)
        for split in ("train", "val", "test")
    }
    for folder in ("imagesTr", "labelsTr", "imagesTs"):
        (root / folder).mkdir(parents=True, exist_ok=True)

    training_ids = []
    validation_ids = []
    for split in ("train", "val"):
        for record in subsets[split]:
            case_id = _case_id(record)
            _save_image(_read_nifti(record["image_path"]), root / "imagesTr" / f"{case_id}_0000.png")
            mask = _read_nifti(record["mask_path"])
            labels = np.rint(mask).astype(np.uint8)
            if not np.isin(labels, (0, 1, 2, 3)).all():
                raise ValueError(f"Unexpected label ids in {record['mask_path']}")
            Image.fromarray(labels, mode="L").save(root / "labelsTr" / f"{case_id}.png")
            (training_ids if split == "train" else validation_ids).append(case_id)

    test_ids = []
    for record in subsets["test"]:
        case_id = _case_id(record)
        _save_image(_read_nifti(record["image_path"]), root / "imagesTs" / f"{case_id}_0000.png")
        test_ids.append(case_id)

    dataset_json = {
        "channel_names": {"0": "ultrasound"},
        "labels": {"background": 0, "lv_endo": 1, "myocardium": 2, "left_atrium": 3},
        "numTraining": len(training_ids) + len(validation_ids),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
        "name": dataset_name,
        "description": "CAMUS native-resolution 2D echocardiography",
        "reference": "CAMUS",
        "licence": "Research use subject to CAMUS terms",
        "release": "1.0",
    }
    (root / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    (root / "splits_final.json").write_text(json.dumps([
        {"train": training_ids, "val": validation_ids}
    ], indent=2), encoding="utf-8")
    return {"dataset_dir": str(root), "train": len(training_ids), "val": len(validation_ids), "test": len(test_ids)}


def main():
    parser = argparse.ArgumentParser(description="Export patient-disjoint CAMUS PNGs to nnU-Net v2 format.")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_root", required=True, help="nnUNet_raw directory")
    parser.add_argument("--dataset_id", type=int, default=501)
    parser.add_argument("--dataset_name", default="CardioContrast")
    args = parser.parse_args()
    result = export_dataset(args.data_dir or __import__("config").CAMUS_DATA_DIR,
                            args.output_root, args.dataset_id, args.dataset_name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
