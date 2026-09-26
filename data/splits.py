"""
data/splits.py — Patient-level train / val / test split for CAMUS.

Why this file exists
--------------------
The previous code used patients 451-500 as "validation" AND selected the best
checkpoint on them. Patients 451-500 are the CAMUS test set, so every reported
number was chosen by peeking at the test set (optimistic bias).

Rules enforced here:
  * Splits are by PATIENT, never by image, so no patient appears in two splits.
  * Model selection uses VAL only. TEST is touched once, by evaluate.py.

Split source (first match wins):
  1. The official CAMUS_public split files, if present next to database_nifti:
       <root>/database_split/subgroup_training.txt
       <root>/database_split/subgroup_validation.txt
       <root>/database_split/subgroup_testing.txt
     (one "patientXXXX" per line). Verify these exist in your download.
  2. Fallback: train = 1-400, val = 401-450, test = 451-500
     (test = the 50 held-out CAMUS test patients; val carved from the 450).
"""

import os
import re

FALLBACK_RANGES = {
    "train": (1, 400),
    "val":   (401, 450),
    "test":  (451, 500),
}


def patient_number(patient_id):
    m = re.search(r"(\d+)", patient_id)
    if m is None:
        raise ValueError("Cannot parse patient number from '{}'".format(patient_id))
    return int(m.group(1))


def _official_split_dir(data_dir):
    root = os.path.dirname(os.path.normpath(data_dir))
    d = os.path.join(root, "database_split")
    names = ["subgroup_training.txt", "subgroup_validation.txt", "subgroup_testing.txt"]
    if all(os.path.isfile(os.path.join(d, n)) for n in names):
        return d
    return None


def _read_list(path):
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def get_split_patients(data_dir, split, verbose=True):
    """Return the set of patient folder names (e.g. 'patient0001') for a split."""
    assert split in ("train", "val", "test"), split
    all_patients = sorted(
        p for p in os.listdir(data_dir)
        if p.startswith("patient") and os.path.isdir(os.path.join(data_dir, p)))

    official = _official_split_dir(data_dir)
    if official is not None:
        fname = {"train": "subgroup_training.txt",
                 "val": "subgroup_validation.txt",
                 "test": "subgroup_testing.txt"}[split]
        wanted = _read_list(os.path.join(official, fname))
        patients = {p for p in all_patients if p in wanted}
        source = "official CAMUS split files ({})".format(official)
    else:
        lo, hi = FALLBACK_RANGES[split]
        patients = {p for p in all_patients if lo <= patient_number(p) <= hi}
        source = "fallback ranges {}".format(FALLBACK_RANGES)

    if verbose:
        print("[split] {:5s}: {} patients from {}".format(split, len(patients), source),
              flush=True)
    return patients


def assert_disjoint(data_dir):
    tr = get_split_patients(data_dir, "train", verbose=False)
    va = get_split_patients(data_dir, "val", verbose=False)
    te = get_split_patients(data_dir, "test", verbose=False)
    assert not (tr & va), "train/val overlap"
    assert not (tr & te), "train/test overlap"
    assert not (va & te), "val/test overlap"
    return len(tr), len(va), len(te)
