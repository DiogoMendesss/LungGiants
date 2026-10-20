"""Build the case lists used by a particular training experiment."""

import argparse
import random

import yaml

from data.dataset_index import build_real_index
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

    if settings["synthetic_percent"] != 0:
        raise NotImplementedError(
            "Synthetic training is pending aligned CT–mask pairs."
        )

    if settings["synthetic_dataset"] is not None:
        raise ValueError(
            "Set synthetic_dataset to null for a real-only experiment."
        )

    index = build_real_index(datasets_config, split_path, classes=classes)

    # Reuse the configured seed for reproducible subset selection.
    train_cases = select_real_cases(
        cases=index["train"],
        percent=settings["real_percent"],
        seed=datasets["split"]["seed"],
    )

    if not train_cases:
        raise ValueError("The experiment has no training cases.")

    return train_cases, index["val"], index["test"]


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
    args = parser.parse_args()

    train, val, test = build_experiment_data(
        datasets_config=args.datasets,
        experiment_config=args.experiment,
        split_path=args.split,
        classes=args.classes,
    )

    for name, cases in (("train", train), ("val", val), ("test", test)):
        print(f"\n{name}: {len(cases)} cases")
        for dataset in sorted({case["dataset"] for case in cases}):
            count = sum(case["dataset"] == dataset for case in cases)
            print(f"  {dataset}: {count}")