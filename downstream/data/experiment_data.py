"""Build the case lists used by a particular training experiment."""

import argparse
import json
import math
import random
from pathlib import Path

import yaml

from data.dataset_index import build_real_index, index_folder
from data.synthetic_index import build_synthetic_cases
from tasks import CLASS_CHOICES


def select_real_cases(cases, percent, seed):
    """Select a reproducible subset, preserving dataset proportions."""
    if not 0 <= percent <= 100:
        raise ValueError("real_percent must be between 0 and 100.")

    # Start from a stable order so selection is reproducible.
    groups = {}
    for case in cases:
        groups.setdefault(case["dataset"], []).append(case)

    # Allocate the requested total proportionally across datasets.
    target = int(len(cases) * percent / 100 + 0.5)
    quotas = {
        dataset: len(group) * percent / 100
        for dataset, group in groups.items()
    }
    counts = {
        dataset: int(quota)
        for dataset, quota in quotas.items()
    }

    remaining = target - sum(counts.values())
    ranked = sorted(
        groups,
        key=lambda dataset: (
            -(quotas[dataset] - counts[dataset]),
            dataset,
        ),
    )
    for dataset in ranked[:remaining]:
        counts[dataset] += 1

    rng = random.Random(seed)
    selected = []

    for dataset in sorted(groups):
        group = sorted(
            groups[dataset],
            key=lambda case: case["patient_id"],
        )
        rng.shuffle(group)
        selected.extend(group[:counts[dataset]])

    return sorted(selected, key=lambda case: case["patient_id"])


def validate_percent(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 100):
        raise ValueError(f"{name} must be a finite number between 0 and 100.")


def real_record(case):
    return {
        **case, "source": "real", "sample_id": f"real/{case['patient_id']}",
        "source_patient_id": case["patient_id"], "source_dataset": case["dataset"],
    }


def select_synthetic_ids(index, real_cases, settings, real_percent, synthetic_percent,
                         images, seed):
    """Select conditioning patients, or independently generated case IDs."""
    if settings["conditioned_on_real"]:
        known_ids = {case["patient_id"] for cases in index.values() for case in cases}
        unknown = sorted(images.keys() - known_ids)
        if unknown:
            raise ValueError(f"Unknown synthetic conditioning patient IDs: {unknown[:10]}")
        if 0 < real_percent < 100 and 0 < synthetic_percent < 100:
            if not math.isclose(real_percent + synthetic_percent, 100, abs_tol=1e-9):
                raise ValueError(
                    "Conditioned mixes must sum to 100%, or one percentage must be "
                    "0 or 100. Other partial-mix overlap policies are undefined."
                )
            # Use the exact complement, including rounding, not a second sample.
            selected_real = {case["patient_id"] for case in real_cases}
            return [case["patient_id"] for case in index["train"]
                    if case["patient_id"] not in selected_real]
        return [case["patient_id"] for case in select_real_cases(
            index["train"], synthetic_percent, seed,
        )]

    count = int(len(index["train"]) * synthetic_percent / 100 + 0.5)
    if count > len(images):
        raise ValueError(f"Requested {count} independent synthetic cases, only {len(images)} available.")
    ids = sorted(images)
    random.Random(seed).shuffle(ids)
    return ids[:count]


def save_case_manifest(path, case_lists):
    """Persist exact selected inputs and provenance for each split."""
    with Path(path).open("w") as file:
        json.dump(dict(zip(("train", "val", "test"), case_lists)), file, indent=2)
        file.write("\n")


def build_experiment_data(
    datasets_config,
    experiment_config,
    split_path="configs/splits/real_split.json",
    classes="nodule",
):
    """Return train, validation, and test case lists."""
    with open(experiment_config) as file:
        experiment = yaml.safe_load(file)

    with open(datasets_config) as file:
        datasets = yaml.safe_load(file)

    settings = experiment["data"]

    real_percent = settings["real_percent"]
    synthetic_percent = settings["synthetic_percent"]
    validate_percent(real_percent, "real_percent")
    validate_percent(synthetic_percent, "synthetic_percent")
    name = settings["synthetic_dataset"]
    if synthetic_percent == 0 and name is not None:
        raise ValueError(
            "Set synthetic_dataset to null for a real-only experiment."
        )
    if synthetic_percent > 0 and name not in datasets.get("synthetic", {}):
        raise ValueError(f"Choose a configured synthetic_dataset, got {name!r}.")

    index = build_real_index(datasets_config, split_path, classes=classes)

    # Reuse the configured seed for reproducible subset selection.
    train_cases = select_real_cases(
        cases=index["train"],
        percent=real_percent,
        seed=datasets["split"]["seed"],
    )

    synthetic_cases = []
    if synthetic_percent > 0:
        synthetic_settings = datasets["synthetic"][name]
        if not isinstance(synthetic_settings.get("conditioned_on_real"), bool):
            raise ValueError(f"{name}: conditioned_on_real must be true or false.")
        images = index_folder(Path(datasets["root"]) / synthetic_settings["images"])
        selected_ids = select_synthetic_ids(
            index, train_cases, synthetic_settings, real_percent, synthetic_percent,
            images, datasets["split"]["seed"],
        )
        if selected_ids:
            synthetic_cases = build_synthetic_cases(
                datasets["root"], name, synthetic_settings, selected_ids, images, classes,
                {case["patient_id"]: case for case in index["train"]},
            )
    train_cases = [real_record(case) for case in train_cases] + synthetic_cases

    if not train_cases:
        raise ValueError("The experiment has no training cases.")

    return (train_cases, [real_record(case) for case in index["val"]],
            [real_record(case) for case in index["test"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="configs/datasets.yaml")
    parser.add_argument(
        "--experiment",
        default="configs/experiments/real_100.yaml",
    )
    parser.add_argument(
        "--split",
        default="configs/splits/real_split.json",
    )
    parser.add_argument("--classes", choices=CLASS_CHOICES, default="nodule")
    parser.add_argument("--manifest", help="Write the exact selected cases to this JSON file.")
    args = parser.parse_args()

    train, val, test = build_experiment_data(
        datasets_config=args.datasets,
        experiment_config=args.experiment,
        split_path=args.split,
        classes=args.classes,
    )

    if args.manifest:
        save_case_manifest(args.manifest, (train, val, test))

    for name, cases in (("train", train), ("val", val), ("test", test)):
        print(f"\n{name}: {len(cases)} cases")
        for dataset in sorted({case["dataset"] for case in cases}):
            count = sum(case["dataset"] == dataset for case in cases)
            print(f"  {dataset}: {count}")
