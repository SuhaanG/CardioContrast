"""
compute_class_weights.py
Run once on the training set to compute inverse-frequency class weights.
Usage: python compute_class_weights.py
"""
import os, glob
import numpy as np
import nibabel as nib
import config

mask_paths = sorted(glob.glob(
    os.path.join(config.CAMUS_DATA_DIR, "patient*", "*_gt.nii.gz")))

# Use only training patients (1-450)
def patient_num(p):
    folder = os.path.basename(os.path.dirname(p))
    digits = ''.join(filter(str.isdigit, folder))
    return int(digits) if digits else 0

mask_paths = [p for p in mask_paths
              if "half_sequence" not in p and patient_num(p) <= 450]

total_bg = total_fg = 0
per_structure = {1: 0, 2: 0, 3: 0}

for mp in mask_paths:
    mask = np.squeeze(nib.load(mp).get_fdata())
    total_pixels = mask.size
    total_bg += np.sum(mask == 0)
    for label in [1, 2, 3]:
        px = np.sum(mask == label)
        per_structure[label] += px
        total_fg += px

total = total_bg + total_fg
print(f"Total pixels: {total}")
print(f"Background:   {total_bg} ({100*total_bg/total:.1f}%)")
print(f"LV endo:      {per_structure[1]} ({100*per_structure[1]/total:.1f}%)")
print(f"Myocardium:   {per_structure[2]} ({100*per_structure[2]/total:.1f}%)")
print(f"Left atrium:  {per_structure[3]} ({100*per_structure[3]/total:.1f}%)")
print(f"All fg:       {total_fg} ({100*total_fg/total:.1f}%)")

# Inverse-frequency weights normalized to sum to 2
w_bg = (total / (2 * total_bg))
w_fg = (total / (2 * total_fg))
print(f"\nInverse-frequency weights (sum to 2):")
print(f"  w_bg = {w_bg:.4f}")
print(f"  w_fg = {w_fg:.4f}")
