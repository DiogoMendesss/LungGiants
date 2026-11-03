"""Evaluate a saved checkpoint on its fixed real validation and test splits.

Set MODEL_PATH below, set the MODEL_PATH environment variable, or pass
--model-path. Run configuration is read from the checkpoint's run/configs/.
"""

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import warnings

import numpy as np
import yaml

from evaluation_metrics import (
    ATTRIBUTES, METRICS, read_attributes, score_nodules, summarize_attributes,
    summarize_cases, validate_instances, voxel_metrics,
)
from tasks import class_ids, output_channels, predict_labels, validate_classes


MODEL_PATH = os.environ.get("MODEL_PATH", "")
CASE_FIELDS = ["split", "dataset", "case_id", "class", "status", "target_voxels",
               "predicted_voxels", "true_positive_voxels", *METRICS]
NODULE_FIELDS = ["split", "dataset", "case_id", "label_id", "annotation", "status",
                 "size_class", "sphericity_class", "spiculation_class", "attribute_issues",
                 "target_voxels", "predicted_voxels", "true_positive_voxels",
                 "assigned_components", "dsc"]
COMPONENT_FIELDS = ["split", "dataset", "case_id", "component_id", "reason",
                    "voxel_count", "gt_labels"]
ATTRIBUTE_FIELDS = ["attribute", "class", "n_nodules", "n_scored", "n_cases",
                    "n_missing_from_grid", "n_missed", "n_ambiguous", "dsc_mean", "dsc_std"]


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def config_directory(model_path, explicit=None):
    if explicit:
        directory = Path(explicit)
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        return directory
    for parent in Path(model_path).resolve().parents:
        if (parent / "configs" / "training.yaml").is_file():
            return parent / "configs"
    raise FileNotFoundError("Cannot find this checkpoint's run/configs; pass --config-dir.")


def resolve_data_root(value):
    root = Path(value)
    if root.is_dir():
        return root
    # Saved cluster paths can be accessed through the existing Windows NAS mount.
    if os.name == "nt" and str(value).startswith("/nas-"):
        mounted = Path("Z:" + str(value))
        if mounted.is_dir():
            return mounted
    raise FileNotFoundError(f"Dataset root unavailable: {value}; pass --data-root.")


def nifti_path(folder, case_id):
    candidates = [Path(folder) / f"{case_id}{suffix}" for suffix in (".nii.gz", ".nii")]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one NIfTI for {case_id} in {folder}: {found}")
    return str(found[0])


def evaluation_cases(datasets_config, saved_split, root, classes, annotation_root,
                     splits=("val", "test")):
    """Use saved IDs only; never regenerate splits or index synthetic data."""
    sets = {}
    for split in ("train", "val", "test"):
        ids = saved_split[split]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs in saved {split} split.")
        sets[split] = set(ids)
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("Patient overlap between saved splits.")
    result = {}
    for split, source_split in (("val", "train"), ("test", "test")):
        if split not in splits:
            continue
        lookup = {}
        for dataset, folders in datasets_config["real"][source_split].items():
            image_folder = root / folders["images"]
            if not image_folder.is_dir():
                raise FileNotFoundError(image_folder)
            for image_path in sorted(image_folder.iterdir()):
                if not image_path.is_file() or not image_path.name.endswith((".nii", ".nii.gz")):
                    continue
                case_id = image_path.name[:-7] if image_path.name.endswith(".nii.gz") else image_path.stem
                if case_id not in sets[split]:
                    continue
                if case_id in lookup:
                    raise ValueError(f"Duplicate case identifier: {case_id}")
                nod_folder = folders.get("nod_masks", folders.get("masks"))
                if not nod_folder:
                    raise ValueError(f"Missing nod_masks in configuration for {dataset}.")
                record = {
                    "case_id": case_id, "patient_id": case_id, "dataset": dataset,
                    "image": str(image_path), "nod_mask": nifti_path(root / nod_folder, case_id),
                    "annotation_dir": str(annotation_root / source_split / dataset / "annotations"),
                }
                if classes == "lung+nodule":
                    record["ts_mask"] = nifti_path(root / folders["ts_masks"], case_id)
                lookup[case_id] = record
        missing = sets[split] - lookup.keys()
        if missing:
            raise FileNotFoundError(f"Missing {split} cases: {sorted(missing)}")
        result[split] = [lookup[case_id] for case_id in saved_split[split]]
    return result


def load_model(model_path):
    """Load this repository's Lightning checkpoint without training/logging setup."""
    import torch
    from models.segresnet import SegResNet

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    parameters = checkpoint["hyper_parameters"]
    model_config = dict(parameters["model_config"])
    classes = validate_classes(parameters.get("classes", "nodule"))
    channels = output_channels(classes)
    if model_config.get("out_channels", channels) != channels:
        raise ValueError("Checkpoint task and output channel count disagree.")
    model = SegResNet(**{**model_config, "out_channels": channels})
    state = checkpoint["state_dict"]
    if not all(key.startswith("network.") for key in state):
        raise ValueError("Expected a SegmentationModule checkpoint with network.* weights.")
    model.load_state_dict({key.removeprefix("network."): value for key, value in state.items()}, strict=True)
    model.eval()
    return model, classes


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def render_html(path, summary, attributes):
    def text(value):
        return html.escape(str(value))

    def number(value):
        return "N/A" if value is None else f"{value:.4f}"

    def table(headers, rows):
        return "<table><thead><tr>" + "".join(f"<th>{text(v)}</th>" for v in headers) + \
            "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{text(v)}</td>" for v in row)
                                              + "</tr>" for row in rows) + "</tbody></table>"

    case_rows = []
    for name, values in summary["case_metrics"].items():
        for metric in METRICS:
            stats = values[metric]
            case_rows.append([name, metric, stats["n"],
                              f"{number(stats['mean'])} ± {number(stats['std'])}",
                              values["n_empty_gt"], values["n_unavailable"]])
    attribute_rows = [[row["attribute"], row["class"], row["n_nodules"], row["n_scored"],
                       row["n_missing_from_grid"], row["n_missed"], row["n_ambiguous"],
                       f"{number(row['dsc_mean'])} ± {number(row['dsc_std'])}"] for row in attributes]
    status = "PROVISIONAL: components overlap multiple GT nodules; inspect component_warnings.csv." \
        if summary["provisional"] else "No components overlap multiple GT nodules."
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>{text(summary['split'])} segmentation evaluation</title>
<style>body{{font:15px system-ui;max-width:1200px;margin:35px auto;padding:0 20px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:9px;border:1px solid #ddd;text-align:left}}
th{{background:#eef3f7}}code{{overflow-wrap:anywhere}}</style>
<h1>{text(summary['split'])} evaluation</h1><p>Checkpoint: <code>{text(summary['model_path'])}</code></p>
<p>{text(status)}</p><h2>Per-case metrics</h2>
{table(['Class','Metric','Scored cases','Mean ± sample std','Empty GT','Unavailable'], case_rows)}
<h2>Per-nodule attribute DSC</h2>
{table(['Attribute','Class','Total nodules','Scored','Missing from grid','Missed','Ambiguous','Mean ± sample std'], attribute_rows)}
<p>Unmatched predicted components: {summary['components']['unmatched_count']}
({summary['components']['unmatched_voxels']} voxels). Ambiguous components:
{summary['components']['ambiguous_count']}.</p>
<p>Missing/invalid attribute counts: {text(summary['missing_attributes'])}.</p>
<p>All per-case metrics use the full prediction. Attribute DSC excludes unmatched and ambiguous
components, retains missed nodules as zero, and averages individual nodule scores equally.
Nodules absent from the evaluation grid are counted but not scored. Empty-GT case metrics are
unavailable and excluded. Lung is class 1, excluding nodules, and is unavailable for nodule-only models.
Standard deviation uses ddof=1; it is unavailable for fewer than two observations.</p>
<p><a href="case_metrics.csv">Per-case CSV</a> · <a href="nodule_metrics.csv">Per-nodule CSV</a> ·
<a href="attribute_summary.csv">Attribute summaries</a> ·
<a href="component_warnings.csv">Discarded components</a> · <a href="summary.json">Full metadata</a></p>
</html>"""
    Path(path).write_text(content, encoding="utf-8")


def evaluate_split(model, cases, classes, split, output_dir, data_config, device,
                   precision, provenance, save_masks=False):
    import nibabel as nib
    import torch
    from data.transforms import build_transforms

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    transform = build_transforms(
        expected_shape=data_config.get("expected_shape", (256, 256, 128)),
        in_min=data_config.get("in_min", 0.0), in_max=data_config.get("in_max", 1.0), classes=classes,
    )
    all_cases, all_nodules, all_components = [], [], []
    model.to(device).eval()
    dtype = torch.float16 if precision == "16-mixed" else torch.bfloat16
    for index, case in enumerate(cases, 1):
        identity = {"split": split, "dataset": case["dataset"], "case_id": case["case_id"]}
        attributes = read_attributes(case["annotation_dir"], case["case_id"], case["dataset"])
        instance_image = nib.load(case["nod_mask"])
        instances = validate_instances(np.asarray(instance_image.dataobj))
        loaded = transform(case)
        images = loaded["image"].as_tensor().unsqueeze(0).to(device)
        target = loaded["label"].as_tensor().numpy()[0]
        context = torch.autocast(device_type="cuda", dtype=dtype) \
            if device.type == "cuda" and precision != "32-true" else nullcontext()
        with torch.inference_mode(), context:
            prediction = predict_labels(model(images), classes)[0, 0].cpu().numpy()
        for name in ("lung", "nodule"):
            class_id = class_ids(classes).get(name)
            metrics = voxel_metrics(prediction == class_id, target == class_id) if class_id else {
                "status": "unavailable", "target_voxels": None, "predicted_voxels": None,
                "true_positive_voxels": None, **dict.fromkeys(METRICS),
            }
            all_cases.append({**identity, "class": name, **metrics})
        nodule_rows, components, assigned = score_nodules(
            prediction == class_ids(classes)["nodule"], instances, attributes,
        )
        all_nodules.extend({**identity, **row} for row in nodule_rows)
        all_components.extend({**identity, **row} for row in components)
        ambiguous = [row for row in components if row["reason"] == "multiple_gt_labels"]
        if ambiguous:
            warnings.warn(f"{split}/{case['case_id']}: discarded {len(ambiguous)} components "
                          f"overlapping multiple GT labels: {ambiguous}", stacklevel=2)
        issues = [row for row in nodule_rows if row["attribute_issues"] or row["dsc"] is None]
        if issues:
            warnings.warn(f"{split}/{case['case_id']}: {len(issues)} nodules have annotation/grid "
                          "issues; see nodule_metrics.csv.", stacklevel=2)
        if save_masks:
            mask_dir = output_dir / "masks" / case["dataset"]
            mask_dir.mkdir(parents=True, exist_ok=True)
            for suffix, mask in (("prediction", prediction), ("assigned_nodules", assigned)):
                header = instance_image.header.copy()
                header.set_data_dtype(np.int32)
                nib.save(nib.Nifti1Image(mask.astype(np.int32), instance_image.affine, header),
                         mask_dir / f"{case['case_id']}_{suffix}.nii.gz")
        print(f"{split}: {index}/{len(cases)} {case['case_id']}", flush=True)
    attributes = summarize_attributes(all_nodules)
    unmatched = [row for row in all_components if row["reason"] == "no_gt_overlap"]
    ambiguous = [row for row in all_components if row["reason"] == "multiple_gt_labels"]
    summary = {
        **provenance, "split": split, "classes": classes, "device": str(device), "precision": precision,
        "provisional": bool(ambiguous), "n_cases": len(cases), "n_nodules": len(all_nodules),
        "n_scored_nodules": sum(row["dsc"] is not None for row in all_nodules),
        "n_missing_from_grid": sum(row["dsc"] is None for row in all_nodules),
        "case_metrics": summarize_cases(all_cases), "attribute_metrics": attributes,
        "missing_attributes": {attribute: sum(row[f"{attribute}_class"] is None for row in all_nodules)
                               for attribute in ATTRIBUTES},
        "components": {"unmatched_count": len(unmatched),
                       "unmatched_voxels": sum(row["voxel_count"] for row in unmatched),
                       "ambiguous_count": len(ambiguous),
                       "ambiguous_voxels": sum(row["voxel_count"] for row in ambiguous)},
        "conventions": {
            "connectivity": 26, "std_ddof": 1, "empty_gt_case_metrics": "excluded",
            "missed_nodule_dsc": 0, "missing_grid_nodule_dsc": None,
            "prediction_rule": "argmax" if classes == "lung+nodule" else "sigmoid > 0.5",
            "lung": "class 1 excludes nodule voxels; unavailable in nodule-only models",
            "size": "equiv_diameter_mm stores the size class 1..3",
            "shape_attributes": "ceil(median(finite ratings in [1,5], including fractional LNDb values))",
            "attribute_averaging": "equal weight per scored nodule, including missed nodules",
            "ambiguous_components": "discarded; affected scores and summaries are provisional",
            "unmatched_components": "excluded only from per-nodule DSC",
            "label_mapping": {"LIDC_IDRI": "filename X + 1", "LNDb": "filename X"},
        },
        "cases": cases,
    }
    write_csv(output_dir / "case_metrics.csv", all_cases, CASE_FIELDS)
    write_csv(output_dir / "nodule_metrics.csv", all_nodules, NODULE_FIELDS)
    write_csv(output_dir / "attribute_summary.csv", attributes, ATTRIBUTE_FIELDS)
    write_csv(output_dir / "component_warnings.csv", all_components, COMPONENT_FIELDS)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    render_html(output_dir / "report.html", summary, attributes)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", "--model", dest="model_path", default=MODEL_PATH)
    parser.add_argument("--set", dest="evaluation_set", choices=("val", "test", "both"),
                        default="both", help="Which split(s) to evaluate (default: both).")
    parser.add_argument("--config-dir", help="Defaults to this checkpoint's run/configs directory.")
    parser.add_argument("--data-root", help="Override datasets.yaml root (LungNodule-nifti).")
    parser.add_argument("--annotations-root", help="Original Lung_Nodule root; defaults to sibling of data root.")
    parser.add_argument("--output-dir", "--output-path", dest="output_dir",
                        help="New directory; defaults to run/evaluation/<checkpoint>_<timestamp>.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--precision", choices=("auto", "32-true", "16-mixed", "bf16-mixed"), default="auto")
    parser.add_argument("--save-masks", action="store_true", help="Save semantic and assigned-instance predictions.")
    args = parser.parse_args(argv)
    selected_splits = ("val", "test") if args.evaluation_set == "both" else (args.evaluation_set,)
    if not args.model_path:
        parser.error("Set MODEL_PATH or provide --model-path pointing to a checkpoint.")
    model_path = Path(args.model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    config_dir = config_directory(model_path, args.config_dir)
    datasets_path, training_path, split_path = [config_dir / name for name in
                                               ("datasets.yaml", "training.yaml", "real_split.json")]
    datasets_config, training = read_yaml(datasets_path), read_yaml(training_path)
    saved_split = json.loads(split_path.read_text(encoding="utf-8"))
    manifest_path = config_dir / "selected_cases.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for split in selected_splits:
            ids = [row["case_id"] for row in manifest[split]]
            if len(ids) != len(set(ids)) or set(ids) != set(saved_split[split]):
                raise ValueError(f"Saved manifest and split disagree for {split}.")
    root = resolve_data_root(args.data_root or datasets_config["root"])
    annotations = Path(args.annotations_root) if args.annotations_root else root.parent / "Lung_Nodule"
    model, classes = load_model(model_path)
    cases = evaluation_cases(datasets_config, saved_split, root, classes, annotations, selected_splits)
    import torch
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    precision = training.get("trainer", {}).get("precision", "32-true") if args.precision == "auto" else args.precision
    if str(precision) in ("32", "32.0"):
        precision = "32-true"
    if precision not in ("32-true", "16-mixed", "bf16-mixed"):
        raise ValueError(f"Unsupported precision: {precision}; specify --precision.")
    if device.type != "cuda":
        precision = "32-true"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output_dir) if args.output_dir else config_dir.parent / "evaluation" / f"{model_path.stem}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"model_path": str(model_path), "config_dir": str(config_dir),
                  "split_path": str(split_path), "data_root": str(root),
                  "annotations_root": str(annotations), "created_utc": timestamp,
                  "training_data_config": training["data"], "requested_set": args.evaluation_set}
    for split, folder in (("val", "validation"), ("test", "test")):
        if split not in selected_splits:
            continue
        summary = evaluate_split(model, cases[split], classes, split, output / folder,
                                 training["data"], device, precision, provenance, args.save_masks)
        print(f"{folder}: {summary['n_cases']} cases, {summary['n_scored_nodules']} scored nodules; "
              f"report: {output / folder / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
