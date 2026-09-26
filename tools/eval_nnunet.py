"""
tools/eval_nnunet.py — Score nnU-Net (or any multi-label PNG/NIfTI predictions)
with the SAME code used for CardioContrast (lib/report.py).

  python tools/eval_nnunet.py --pred_dir preds_nnunet --out_dir experiments/nnunet_2d_fold0

Predictions must be named <patient>_<view>_<phase>.png (or .nii.gz) with labels
0=bg, 1=LV, 2=MYO, 3=LA at native resolution.
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from data.dataset_camus import index_images, load_nifti_2d  # noqa: E402
from lib.report import Reporter  # noqa: E402


def load_pred(pred_dir, case):
    png = os.path.join(pred_dir, case + ".png")
    if os.path.isfile(png):
        return np.array(Image.open(png))
    nii = os.path.join(pred_dir, case + ".nii.gz")
    if os.path.isfile(nii):
        return load_nifti_2d(nii)[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_dir", default=config.CAMUS_DATA_DIR)
    ap.add_argument("--split", default="test")
    ap.add_argument("--name", default="nnunet")
    args = ap.parse_args()

    rep = Reporter()
    missing = 0
    for r in index_images(args.data_dir, args.split):
        pred = load_pred(args.pred_dir, r["case"])
        if pred is None:
            missing += 1
            continue
        gt, spacing = load_nifti_2d(r["mask_path"])
        gt = gt.astype(np.uint8)
        assert pred.shape == gt.shape, "{}: pred {} vs gt {}".format(r["case"], pred.shape,
                                                                     gt.shape)
        for s in (1, 2, 3):
            rep.add(r["patient"], r["view"], r["phase"], r["quality"], r["ref_ef"], s,
                    args.name, pred == s, gt, spacing)
    if missing:
        print("WARNING: {} test cases had no prediction file".format(missing))
    rep.write(os.path.join(args.out_dir, "eval_{}_canonical".format(args.split)),
              meta={"model": args.name, "pred_dir": args.pred_dir, "split": args.split})


if __name__ == "__main__":
    main()
