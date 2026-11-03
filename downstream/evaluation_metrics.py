"""Nodule matching and attribute summaries, independent of model inference."""

from pathlib import Path

import numpy as np
from scipy import ndimage


ATTRIBUTES = {"size": (1, 2, 3), "sphericity": (1, 2, 3, 4, 5),
              "spiculation": (1, 2, 3, 4, 5)}
METRICS = ("dsc", "precision", "recall")
LABEL_OFFSETS = {"LIDC_IDRI": 1, "LNDb": 0}


def validate_instances(mask):
    mask = np.asarray(mask)
    if mask.ndim != 3 or not np.isfinite(mask).all():
        raise ValueError("Instance mask must be a finite 3D array.")
    if np.any(mask < 0) or np.any(mask != np.floor(mask)):
        raise ValueError("Instance mask must contain nonnegative integer labels.")
    if mask.max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError("Instance label exceeds int32 range.")
    return mask.astype(np.int32, copy=False)


def voxel_metrics(prediction, target):
    """Keep empty-GT cases in the CSV, but exclude their metrics from means."""
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ.")
    tp = int(np.count_nonzero(prediction & target))
    p, g = int(prediction.sum()), int(target.sum())
    return {
        "target_voxels": g, "predicted_voxels": p, "true_positive_voxels": tp,
        "status": "scored" if g else "empty_gt",
        "dsc": 2 * tp / (p + g) if g else None,
        "precision": (tp / p if p else 0.0) if g else None,
        "recall": tp / g if g else None,
    }


def match_components(prediction, instances):
    """Assign entire 26-connected components; discard unmatched/ambiguous ones.

    Multiple components may receive the same GT ID. The returned diagnostics
    identify every discarded component; ambiguities make summaries provisional.
    """
    instances = validate_instances(instances)
    prediction = np.asarray(prediction, dtype=bool)
    if prediction.shape != instances.shape:
        raise ValueError("Prediction and instance mask shapes differ.")
    components, count = ndimage.label(prediction, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(components.ravel())
    overlap = prediction & (instances > 0)
    pairs = np.unique(np.column_stack((components[overlap], instances[overlap])), axis=0)
    overlaps = {}
    for component, label in pairs:
        overlaps.setdefault(int(component), []).append(int(label))
    assignment = np.zeros(count + 1, dtype=np.int32)
    assigned_counts, ambiguous_labels, diagnostics = {}, set(), []
    for component in range(1, count + 1):
        labels = overlaps.get(component, [])
        if len(labels) == 1:
            label = labels[0]
            assignment[component] = label
            assigned_counts[label] = assigned_counts.get(label, 0) + 1
        else:
            ambiguous_labels.update(labels)
            diagnostics.append({
                "component_id": component, "voxel_count": int(sizes[component]),
                "reason": "multiple_gt_labels" if labels else "no_gt_overlap",
                "gt_labels": labels,
            })
    return assignment[components], assigned_counts, ambiguous_labels, diagnostics


def read_attributes(annotation_dir, case_id, dataset):
    """Map filenames to GT IDs; preserve missing/invalid attribute diagnostics."""
    folder = Path(annotation_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Annotation directory does not exist: {folder}")
    offset = LABEL_OFFSETS[dataset]
    records = {}
    for path in sorted(folder.glob(f"{case_id}_nodule*.npz")):
        index = int(path.stem.rsplit("_nodule", 1)[1])
        label = index + offset
        if label <= 0 or label in records:
            raise ValueError(f"Invalid/duplicate annotation label: {path}")
        row = {"label_id": label, "annotation": path.name, "attribute_issues": []}
        with np.load(path, allow_pickle=False) as info:
            if str(info["patient_id"].item()) != case_id:
                raise ValueError(f"patient_id does not match filename: {path}")
            for attribute, classes in ATTRIBUTES.items():
                key = "equiv_diameter_mm" if attribute == "size" else f"char_{attribute}"
                values = np.asarray(info[key], dtype=float).ravel() if key in info else np.array([])
                valid = np.isfinite(values) & (
                    np.isin(values, classes) if attribute == "size" else
                    ((values >= 1) & (values <= 5))
                )
                if attribute == "size":
                    value = int(values[0]) if len(values) == 1 and valid.all() else None
                else:
                    value = int(np.ceil(np.median(values[valid]))) if valid.any() else None
                if value is None:
                    row["attribute_issues"].append(f"{attribute}:missing_or_invalid")
                elif not valid.all():
                    row["attribute_issues"].append(f"{attribute}:invalid_ratings_ignored")
                row[f"{attribute}_class"] = value
        records[label] = row
    return records


def score_nodules(prediction, instances, attributes):
    instances = validate_instances(instances)
    assigned, component_counts, ambiguous, diagnostics = match_components(prediction, instances)
    gt_ids, gt_counts = np.unique(instances, return_counts=True)
    gt_counts = {int(k): int(v) for k, v in zip(gt_ids, gt_counts) if k > 0}
    pred_ids, pred_counts = np.unique(assigned, return_counts=True)
    pred_counts = {int(k): int(v) for k, v in zip(pred_ids, pred_counts) if k > 0}
    equal = (assigned > 0) & (assigned == instances)
    tp_ids, tp_counts = np.unique(instances[equal], return_counts=True)
    tp_counts = {int(k): int(v) for k, v in zip(tp_ids, tp_counts)}
    rows = []
    for label in sorted(set(gt_counts) | set(attributes)):
        metadata = attributes.get(label, {
            "label_id": label, "annotation": None,
            **{f"{key}_class": None for key in ATTRIBUTES},
            "attribute_issues": ["missing_annotation"],
        })
        g, p, tp = gt_counts.get(label, 0), pred_counts.get(label, 0), tp_counts.get(label, 0)
        status = ("missing_from_evaluation_grid" if not g else
                  "ambiguous_component_discarded" if label in ambiguous else
                  "missed" if not p else "scored")
        rows.append({**metadata, "status": status, "target_voxels": g,
                     "predicted_voxels": p, "true_positive_voxels": tp,
                     "assigned_components": component_counts.get(label, 0),
                     "dsc": 2 * tp / (g + p) if g else None})
    return rows, diagnostics, assigned


def mean_std(values):
    values = [float(v) for v in values if v is not None]
    return {"n": len(values), "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def summarize_attributes(rows):
    summaries = []
    for attribute, classes in ATTRIBUTES.items():
        key = f"{attribute}_class"
        for value in classes:
            group = [row for row in rows if row[key] == value]
            stats = mean_std(row["dsc"] for row in group)
            summaries.append({
                "attribute": attribute, "class": value,
                "n_nodules": len(group), "n_scored": stats["n"],
                "n_cases": len({row["case_id"] for row in group if row["dsc"] is not None}),
                "n_missing_from_grid": sum(row["dsc"] is None for row in group),
                "n_missed": sum(row["status"] == "missed" for row in group),
                "n_ambiguous": sum(row["status"] == "ambiguous_component_discarded" for row in group),
                "dsc_mean": stats["mean"], "dsc_std": stats["std"],
            })
    return summaries


def summarize_cases(rows):
    return {name: {
        "n_cases": sum(row["class"] == name for row in rows),
        "n_empty_gt": sum(row["class"] == name and row["status"] == "empty_gt" for row in rows),
        "n_unavailable": sum(row["class"] == name and row["status"] == "unavailable" for row in rows),
        **{metric: mean_std(row[metric] for row in rows if row["class"] == name)
           for metric in METRICS},
    } for name in ("lung", "nodule")}
