import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def create_dataset(output_dir):
    root = Path(output_dir) / "database_nifti"
    rows, columns = np.mgrid[:64, :64]
    for patient_number in (1, 2, 401, 402):
        patient = f"patient{patient_number:03d}"
        patient_dir = root / patient
        patient_dir.mkdir(parents=True, exist_ok=True)
        image = (30 + 180 * np.exp(-((rows - 32) ** 2 + (columns - 32) ** 2) / 500)).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)
        lv = ((rows - 34) / 13) ** 2 + ((columns - 29) / 16) ** 2 <= 1
        outer = ((rows - 34) / 16) ** 2 + ((columns - 29) / 19) ** 2 <= 1
        la = ((rows - 24) / 7) ** 2 + ((columns - 47) / 9) ** 2 <= 1
        mask[outer] = 2
        mask[lv] = 1
        mask[la] = 3

        affine = np.diag((1.0, 1.0, 1.0, 1.0))
        for suffix, array in (("", image), ("_gt", mask)):
            image_file = patient_dir / f"{patient}_2CH_ED{suffix}.nii.gz"
            nifti = nib.Nifti1Image(array, affine)
            nifti.header.set_xyzt_units("mm")
            nifti.header.set_zooms((1.0, 1.0))
            nib.save(nifti, image_file)
        (patient_dir / "Info_2CH.cfg").write_text(
            "ImageQuality: Good\nLVef: 60\n", encoding="utf-8"
        )
    return root


def main():
    parser = argparse.ArgumentParser(description="Create tiny synthetic CAMUS-style NIfTI files for CPU smoke tests.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(create_dataset(args.output_dir))


if __name__ == "__main__":
    main()
