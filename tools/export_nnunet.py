"""
tools/export_nnunet.py — Export CAMUS to nnU-Net v2 format with OUR split.

nnU-Net (multi-class, no prompt) is the baseline a TMI reviewer will ask for
first. This exports train+val patients to imagesTr/labelsTr and test patients
to imagesTs, at native resolution, and writes splits_final.json so nnU-Net's
fold 0 uses exactly our train/val split.

  python tools/export_nnunet.py --out $nnUNet_raw/Dataset500_CAMUS
  nnUNetv2_plan_and_preprocess -d 500 --verify_dataset_integrity
  cp $nnUNet_raw/Dataset500_CAMUS/splits_final.json $nnUNet_preprocessed/Dataset500_CAMUS/
  nnUNetv2_train 500 2d 0
  nnUNetv2_predict -i $nnUNet_raw/Dataset500_CAMUS/imagesTs -o preds_nnunet -d 500 -c 2d -f 0
  python tools/eval_nnunet.py --pred_dir preds_nnunet --out_dir experiments/nnunet_2d_fold0

PNG + NaturalImage2DIO is used because it is the documented 2D path in nnU-Net v2.
Spacing is not needed by nnU-Net here; our evaluation reads spacing from the
original NIfTI files.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from data.dataset_camus import index_images, load_nifti_2d  # noqa: E402


def to_uint8(img):
    img = img.astype(np.float32)
    if img.max() > 0:
        img = img / img.max() * 255.0
    return img.round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=config.CAMUS_DATA_DIR)
    ap.add_argument("--out", required=True, help=".../nnUNet_raw/Dataset500_CAMUS")
    args = ap.parse_args()

    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    split_cases = {}
    for split, (img_dir, lab_dir) in {"train": ("imagesTr", "labelsTr"),
                                      "val": ("imagesTr", "labelsTr"),
                                      "test": ("imagesTs", "labelsTs")}.items():
        recs = index_images(args.data_dir, split)
        split_cases[split] = []
        for r in recs:
            img, _ = load_nifti_2d(r["image_path"])
            lab, _ = load_nifti_2d(r["mask_path"])
            Image.fromarray(to_uint8(img)).save(
                os.path.join(args.out, img_dir, r["case"] + "_0000.png"))
            Image.fromarray(lab.astype(np.uint8)).save(
                os.path.join(args.out, lab_dir, r["case"] + ".png"))
            split_cases[split].append(r["case"])

    n_tr = len(split_cases["train"]) + len(split_cases["val"])
    with open(os.path.join(args.out, "dataset.json"), "w") as f:
        json.dump({
            "channel_names": {"0": "US"},
            "labels": {"background": 0, "LV": 1, "MYO": 2, "LA": 3},
            "numTraining": n_tr,
            "file_ending": ".png",
            "overwrite_image_reader_writer": "NaturalImage2DIO",
        }, f, indent=2)
    with open(os.path.join(args.out, "splits_final.json"), "w") as f:
        json.dump([{"train": split_cases["train"], "val": split_cases["val"]}], f, indent=1)
    print("Exported {} train, {} val, {} test images to {}".format(
        len(split_cases["train"]), len(split_cases["val"]), len(split_cases["test"]), args.out))
    print("Copy splits_final.json into nnUNet_preprocessed/<dataset>/ after planning.")


if __name__ == "__main__":
    main()
