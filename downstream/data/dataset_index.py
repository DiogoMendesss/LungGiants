"""Index real CT/mask pairs and create or reuse a fixed patient split."""

import argparse
import json
import random
from pathlib import Path

import yaml


def nifti_id(path):
    """Remove .nii or .nii.gz while preserving the full case identifier."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Not a NIfTI file: {path}")


def index_folder(folder):
    """Return {case_id: absolute_path}, rejecting duplicate identifiers."""
    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {folder}")

    files = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if not path.name.endswith((".nii", ".nii.gz")):
            continue

        case_id = nifti_id(path)
        if case_id in files:
            raise ValueError(f"Duplicate case ID '{case_id}' in {folder}")

        files[case_id] = str(path.resolve())

    if not files:
        raise ValueError(f"No NIfTI files found in {folder}")

    return files


def collect_cases(root, datasets):
    """Match images and masks within each configured real dataset."""
    cases = []

    for dataset, folders in sorted(datasets.items()):
        images = index_folder(root / folders["images"])
        masks = index_folder(root / folders["masks"])

        missing_masks = sorted(images.keys() - masks.keys())
        missing_images = sorted(masks.keys() - images.keys())

        if missing_masks or missing_images:
            raise ValueError(
                f"Unmatched files in {dataset}.\n"
                f"Images without masks: {missing_masks[:10]}\n"
                f"Masks without images: {missing_images[:10]}"
            )

        for case_id in sorted(images):
            cases.append(
                {
                    "case_id": case_id,
                    # Dataset prefix avoids collisions between datasets.
                    "patient_id": f"{dataset}/{case_id}",
                    "dataset": dataset,
                    "image": images[case_id],
                    "label": masks[case_id],
                }
            )

    ids = [case["patient_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate patient IDs found.")

    return cases


def create_split(cases, settings):
    """Split patients, optionally selecting validation cases per dataset."""
    fraction = settings["validation_fraction"]
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    rng = random.Random(settings["seed"])
    groups = {}

    for case in cases:
        group = (
            case["dataset"]
            if settings["stratify_by_dataset"]
            else "all"
        )
        groups.setdefault(group, []).append(case["patient_id"])

    train_ids, val_ids = [], []

    for group, ids in sorted(groups.items()):
        ids = sorted(ids)
        rng.shuffle(ids)

        # Round to the nearest integer, with .5 rounded upward.
        n_val = int(len(ids) * fraction + 0.5)
        if not 0 < n_val < len(ids):
            raise ValueError(
                f"Group '{group}' has {len(ids)} cases: "
                "the requested fraction gives an empty train or val split."
            )

        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
    }


def validate_saved_split(saved, train_cases, test_cases, settings):
    """Reject overlapping, changed, or incomplete saved splits."""
    if saved["settings"] != settings:
        raise ValueError(
            "Split settings differ from the saved split. "
            "Reuse the original settings or choose a new split file."
        )

    ids = {}
    for split in ("train", "val", "test"):
        values = saved[split]
        ids[split] = set(values)
        if len(values) != len(ids[split]):
            raise ValueError(f"Duplicate IDs in saved {split} split.")

    if (
        ids["train"] & ids["val"]
        or ids["train"] & ids["test"]
        or ids["val"] & ids["test"]
    ):
        raise ValueError("Patient overlap in saved splits.")

    expected_train = {case["patient_id"] for case in train_cases}
    expected_test = {case["patient_id"] for case in test_cases}

    if ids["train"] | ids["val"] != expected_train:
        raise ValueError(
            "The real training case list has changed since the split was saved."
        )

    if ids["test"] != expected_test:
        raise ValueError(
            "The real test case list has changed since the split was saved."
        )


def build_real_index(config_path, split_path):
    """Return train/val/test records, creating the split only if absent."""
    config_path = Path(config_path)
    split_path = Path(split_path)

    with config_path.open() as file:
        config = yaml.safe_load(file)

    root = Path(config["root"])
    settings = config["split"]

    train_cases = collect_cases(root, config["real"]["train"])
    test_cases = collect_cases(root, config["real"]["test"])

    train_ids = {case["patient_id"] for case in train_cases}
    test_ids = {case["patient_id"] for case in test_cases}

    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Train/test patient overlap: {sorted(overlap)}")

    if split_path.exists():
        with split_path.open() as file:
            saved = json.load(file)
    else:
        saved = {
            "settings": settings,
            **create_split(train_cases, settings),
            "test": sorted(test_ids),
        }

        split_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents overwriting an existing split.
        with split_path.open("x") as file:
            json.dump(saved, file, indent=2)
            file.write("\n")

    validate_saved_split(saved, train_cases, test_cases, settings)

    lookup = {
        case["patient_id"]: case
        for case in train_cases + test_cases
    }

    return {
        split: [lookup[patient_id] for patient_id in saved[split]]
        for split in ("train", "val", "test")
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--split", default="configs/splits/real_split.json")
    args = parser.parse_args()

    index = build_real_index(args.config, args.split)

    for split, cases in index.items():
        print(f"\n{split}: {len(cases)} cases")
        for dataset in sorted({case["dataset"] for case in cases}):
            count = sum(case["dataset"] == dataset for case in cases)
            print(f"  {dataset}: {count}")

    print("\nExample training record:")
    print(json.dumps(index["train"][0], indent=2))