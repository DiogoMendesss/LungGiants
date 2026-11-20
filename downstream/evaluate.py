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
ATTRIBUTE_FIELDS = ["split", "attribute", "class", "n_nodules", "n_scored", "n_cases",
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


def attribute_boxplots(summary, nodule_rows):
    """Render dependency-free SVG box plots from individual scored nodules."""
    panels = []
    names = {"val": "Validation", "test": "Test"}
    for split in summary["splits"]:
        for attribute, levels in ATTRIBUTES.items():
            title = f"{names[split]}: {attribute.capitalize()}"
            parts = [f'<svg viewBox="0 0 420 350" role="img" '
                     f'aria-label="{html.escape(title)} DSC by attribute level">',
                     f'<title>{html.escape(title)}</title>']
            y = lambda value: 260 - 210 * value
            for tick in np.linspace(0, 1, 6):
                parts.append(f'<path d="M50 {y(tick):.2f} H400" stroke="#d5dde5"/>'
                             f'<text x="42" y="{y(tick)+4:.2f}" text-anchor="end">{tick:.1f}</text>')
            parts.append('<text x="18" y="155" transform="rotate(-90 18 155)" '
                         'text-anchor="middle">DSC</text>')
            for index, level in enumerate(levels):
                x = 50 + (index + 0.5) * 350 / len(levels)
                group = [row for row in nodule_rows
                         if row["split"] == split and row[f"{attribute}_class"] == level]
                scores = np.array([row["dsc"] for row in group
                                   if row["dsc"] is not None], dtype=float)
                missed = sum(row.get("status") == "missed" for row in group)
                parts.append(f'<text x="{x:.2f}" y="280" text-anchor="middle">{level}</text>'
                             f'<text x="{x:.2f}" y="297" text-anchor="middle">(N={len(scores)})</text>'
                             f'<text x="{x:.2f}" y="314" text-anchor="middle">Missed: {missed}</text>')
                if not len(scores):
                    parts.append(f'<text x="{x:.2f}" y="155" text-anchor="middle">N/A</text>')
                    continue
                q1, median, q3 = np.percentile(scores, [25, 50, 75])
                iqr = q3 - q1
                inside = scores[(scores >= q1 - 1.5 * iqr) & (scores <= q3 + 1.5 * iqr)]
                low, high = inside.min(), inside.max()
                tip = html.escape(f'Level {level}: N={len(scores)}, median={median:.4f}, '
                                  f'Q1={q1:.4f}, Q3={q3:.4f}, whiskers={low:.4f} to {high:.4f}')
                parts.append(f'<g><title>{tip}</title>'
                             f'<path d="M{x:.2f} {y(high):.2f} V{y(low):.2f} '
                             f'M{x-9:.2f} {y(high):.2f} h18 M{x-9:.2f} {y(low):.2f} h18" '
                             'stroke="#465b70" fill="none"/>'
                             f'<rect x="{x-18:.2f}" y="{y(q3):.2f}" width="36" '
                             f'height="{y(q1)-y(q3):.2f}" fill="#7da2cc" stroke="#465b70"/>'
                             f'<path d="M{x-18:.2f} {y(median):.2f} h36" stroke="#243b53" stroke-width="2"/>')
                for value in scores[(scores < low) | (scores > high)]:
                    parts.append(f'<circle cx="{x:.2f}" cy="{y(value):.2f}" r="2" '
                                 'fill="#465b70" fill-opacity="0.35"/>')
                parts.append('</g>')
            parts.append('<text x="225" y="342" text-anchor="middle">Attribute level</text></svg>')
            panels.append(f'<figure><figcaption>{html.escape(title)}</figcaption>{"".join(parts)}</figure>')
    return '<div class="attribute-plots">' + ''.join(panels) + '</div>'


def render_html(path, summary, nodule_rows=()):
    """Compare selected splits without pooling their metrics or counts."""
    def text(value):
        return html.escape(str(value))

    def number(value):
        return "N/A" if value is None else f"{value:.4f}"

    def score(mean, std):
        return f"{number(mean)} &plusmn; {number(std)}"

    splits = summary["splits"]
    names = {"val": "Validation", "test": "Test"}

    def table_header(labels, metrics):
        fixed = "".join(f'<th rowspan="2" scope="col">{text(v)}</th>' for v in labels)
        groups = "".join(
            f'<th colspan="{len(metrics)}" scope="colgroup" class="split-start">'
            f'{names[split]}</th>' for split in splits
        )
        columns = "".join(
            f'<th scope="col" class="{"split-start" if i == 0 else ""}">{text(metric)}</th>'
            for _ in splits for i, metric in enumerate(metrics)
        )
        return f"<thead><tr>{fixed}{groups}</tr><tr>{columns}</tr></thead>"

    case_rows = []
    for name in ("lung", "nodule"):
        cells = [f'<th scope="row">{name.capitalize()}</th>']
        for values in splits.values():
            for i, metric in enumerate(METRICS):
                stats = values["case_metrics"][name][metric]
                cells.append(f'<td class="{"split-start" if i == 0 else ""}">'
                             f'{score(stats["mean"], stats["std"])}</td>')
        case_rows.append("<tr>" + "".join(cells) + "</tr>")
    case_table = '<table id="case-metrics">' + table_header(
        ["Class"], ["DSC", "Precision", "Recall"]
    ) + "<tbody>" + "".join(case_rows) + "</tbody></table>"

    lookup = {split: {(row["attribute"], row["class"]): row
                      for row in values["attribute_metrics"]}
              for split, values in splits.items()}
    attribute_groups = []
    for attribute, classes in ATTRIBUTES.items():
        rows = []
        for index, value in enumerate(classes):
            cells = []
            if index == 0:
                cells.append(f'<th rowspan="{len(classes)}" scope="rowgroup" '
                             f'class="attribute-name">{text(attribute.capitalize())}</th>')
            cells.append(f'<th scope="row">{value}</th>')
            for split in splits:
                row = lookup[split][attribute, value]
                cells.extend([
                    f'<td class="split-start">{row["n_nodules"]}</td>',
                    f'<td>{row["n_missed"]}</td>',
                    f'<td>{score(row["dsc_mean"], row["dsc_std"])}</td>',
                ])
            rows.append("<tr>" + "".join(cells) + "</tr>")
        attribute_groups.append('<tbody class="attribute-group">' + "".join(rows) + "</tbody>")
    attribute_table = '<table id="attribute-metrics">' + table_header(
        ["Attribute", "Level"], ["Total nodules", "Missed", "DSC (mean / sample std)"]
    ) + "".join(attribute_groups) + "</table>"

    details = []
    for split, values in splits.items():
        status = ("PROVISIONAL: components overlap multiple GT nodules; inspect component_warnings.csv."
                  if values["provisional"] else "No components overlap multiple GT nodules.")
        missing = [f"{attribute}: {count}" for attribute, count in values["missing_attributes"].items() if count]
        note = ("<p>Nodules without a usable attribute level (" + text(", ".join(missing)) +
                "). Excluded only from the corresponding attribute groups; see nodule_metrics.csv.</p>"
                if missing else "")
        components = values["components"]
        details.append(f"<section><h2>{names[split]}</h2>"
                       f"<p>{values['n_cases']} cases; {values['n_scored_nodules']} scored nodules.</p>"
                       f"<p>{text(status)}</p>"
                       f"<p>Unmatched predicted components: {components['unmatched_count']} "
                       f"({components['unmatched_voxels']} voxels). "
                       f"Ambiguous components: {components['ambiguous_count']}.</p>{note}</section>")
    title = " / ".join(names[split] for split in splits) + " evaluation"
    content = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{text(title)}</title>
<style>
body{{font:15px system-ui;max-width:1400px;margin:35px auto;padding:0 20px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:20px 0;border:3px solid #52677c}}
td,th{{padding:10px 12px;border:1px solid #d5dde5;text-align:left}}
thead th{{background:#eaf0f6}}tbody th{{background:#f7f9fb}}
td{{white-space:nowrap;font-variant-numeric:tabular-nums}}
.split-start{{border-left:3px solid #8496a8}}
.attribute-group + .attribute-group tr:first-child > *{{border-top:3px solid #52677c}}
.attribute-name{{vertical-align:middle;font-weight:700}}
.table-scroll{{overflow-x:auto}}.split-details{{display:grid;grid-template-columns:repeat({len(splits)},minmax(0,1fr));gap:24px}}
.split-details section{{background:#f7f9fb;padding:4px 18px;border:1px solid #d5dde5;border-radius:6px}}
.attribute-plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
figure{{margin:0}}figcaption{{text-align:center;font-weight:600}}svg{{width:100%;font:12px system-ui}}
code{{overflow-wrap:anywhere}}a{{color:#245a89}}
@media(max-width:700px){{.split-details{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>{text(title)}</h1><p>Checkpoint: <code>{text(summary['model_path'])}</code></p>
<div class="split-details">{''.join(details)}</div>
<h2>Per-case metrics</h2><div class="table-scroll">{case_table}</div>
<p>Values are mean &plusmn; sample standard deviation, calculated separately for each set.</p>
<h2>DSC distributions by attribute level</h2>
{attribute_boxplots(summary, nodule_rows)}
<p>Each box spans the 25th to 75th percentiles; the line is the median.
Whiskers reach the most extreme observations within 1.5 times the interquartile range;
points beyond them are outliers. N is the number of scored nodules, including missed nodules
with DSC zero. N/A means no scored nodules. Splits are shown separately.</p>
<h2>Per-nodule attribute DSC</h2><div class="table-scroll">{attribute_table}</div>
</body></html>"""
    Path(path).write_text(content, encoding="utf-8")


def write_evaluation(output_dir, results):
    """Write one report and one CSV per category, with explicit split identities."""
    output_dir = Path(output_dir)
    summary = {"model_path": next(iter(results.values()))["summary"]["model_path"],
               "splits": {split: result["summary"] for split, result in results.items()}}
    for name, fields in (("case_metrics", CASE_FIELDS), ("nodule_metrics", NODULE_FIELDS),
                         ("component_warnings", COMPONENT_FIELDS)):
        rows = [row for result in results.values() for row in result[name]]
        write_csv(output_dir / f"{name}.csv", rows, fields)
    attributes = [{"split": split, **row} for split, result in results.items()
                  for row in result["summary"]["attribute_metrics"]]
    write_csv(output_dir / "attribute_summary.csv", attributes, ATTRIBUTE_FIELDS)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    render_html(output_dir / "report.html", summary,
                [row for result in results.values() for row in result["nodule_metrics"]])
    return summary


def evaluate_split(model, cases, classes, split, output_dir, data_config, device,
                   precision, provenance, save_masks=False):
    import nibabel as nib
    import torch
    from data.transforms import build_transforms

    output_dir = Path(output_dir)
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
            mask_dir = output_dir / "masks" / split / case["dataset"]
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
    return {"summary": summary, "case_metrics": all_cases, "nodule_metrics": all_nodules,
            "component_warnings": all_components}


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
    results = {}
    for split in selected_splits:
        results[split] = evaluate_split(model, cases[split], classes, split, output,
                                       training["data"], device, precision, provenance, args.save_masks)
        summary = results[split]["summary"]
        # Save completed splits together, retaining their results if a later split fails.
        write_evaluation(output, results)
        print(f"{split}: {summary['n_cases']} cases, {summary['n_scored_nodules']} scored nodules; "
              f"report: {output / 'report.html'}", flush=True)



if __name__ == "__main__":
    main()
