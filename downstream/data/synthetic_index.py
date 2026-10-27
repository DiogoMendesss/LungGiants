"""Resolve generated CTs to the masks belonging to their case IDs."""

from pathlib import Path

from data.dataset_index import index_folder
from tasks import mask_keys


def index_mask_folders(root, folders):
    """Merge one or more mask directories without ambiguous case IDs."""
    if isinstance(folders, str):
        folders = [folders]
    if not isinstance(folders, list) or not folders:
        raise ValueError("Mask folders must be a path or a nonempty list of paths.")
    merged = {}
    for folder in folders:
        entries = index_folder(root / folder)
        duplicates = merged.keys() & entries.keys()
        if duplicates:
            raise ValueError(f"Duplicate mask IDs across folders: {sorted(duplicates)[:10]}")
        merged.update(entries)
    return merged


def build_synthetic_cases(root, name, settings, selected_ids, images, classes,
                          real_train):
    """Load paths for the selected IDs; extra conditioning masks are permitted."""
    root = Path(root)
    masks = {
        key: index_mask_folders(root, settings[f"{key}s"])
        for key in mask_keys(classes)
    }
    for key, paths in {"image": images, **masks}.items():
        missing = sorted(set(selected_ids) - paths.keys())
        if missing:
            raise ValueError(
                f"{name}: missing {key} for {len(missing)} selected cases: {missing[:10]}"
            )
    conditioned = settings["conditioned_on_real"]
    records = []
    for case_id in sorted(selected_ids):
        if conditioned and case_id not in real_train:
            raise ValueError(f"{name}: conditioning patient is outside training: {case_id}")
        records.append({
            "case_id": case_id,
            "patient_id": case_id,
            "dataset": name,
            "source": "synthetic",
            "sample_id": f"synthetic/{name}/{case_id}",
            "source_patient_id": case_id if conditioned else "",
            "source_dataset": real_train[case_id]["dataset"] if conditioned else "",
            "image": images[case_id],
            **{key: paths[case_id] for key, paths in masks.items()},
        })
    return records
