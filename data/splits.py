import os


def get_split_patients(data_dir, split):
    split = split.lower()
    data_dir = os.path.abspath(data_dir)
    subgroup_dir = os.path.join(os.path.dirname(data_dir), "database_split")
    if split in {"train", "val", "test"}:
        if os.path.isdir(subgroup_dir):
            target = {"train": "subgroup_training.txt", "val": "subgroup_validation.txt", "test": "subgroup_testing.txt"}[split]
            path = os.path.join(subgroup_dir, target)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    patients = [line.strip() for line in fh if line.strip()]
                print(f"Using official CAMUS split source for {split}: {path}")
                return patients
    patients = [f"patient{idx:03d}" for idx in range(1, 401)] if split == "train" else [f"patient{idx:03d}" for idx in range(401, 451)] if split == "val" else [f"patient{idx:03d}" for idx in range(451, 501)]
    print(f"Using fallback numeric CAMUS split for {split}: {os.path.dirname(data_dir)}")
    return patients


def assert_disjoint(*splits):
    seen = set()
    for split in splits:
        current = set(split)
        overlap = current & seen
        if overlap:
            raise AssertionError(f"Patient overlap between splits: {sorted(overlap)[:10]}")
        seen |= current


if __name__ == "__main__":
    print(get_split_patients(".", "train"))
