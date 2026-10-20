"""Create a post-training report from the best checkpoint and validation cases."""

import csv
import json
from pathlib import Path

import numpy as np

from tasks import class_ids, predict_labels, validate_classes


METRICS = ("dice", "precision", "recall")


def check_report_dependencies():
    """Fail before training if the PNG report dependencies are missing."""
    import nibabel  # noqa: F401
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401


def case_metrics(true_positive, predicted_voxels, target_voxels):
    """Foreground voxel metrics; empty ground truth is excluded from reports."""
    if target_voxels == 0:
        return None
    return {
        "dice": 2 * true_positive / (predicted_voxels + target_voxels),
        "precision": true_positive / predicted_voxels if predicted_voxels else 0.0,
        "recall": true_positive / target_voxels,
    }


def select_examples(records):
    """Choose distinct best/worst cases, with deterministic ordering for ties."""
    ranked = sorted(records, key=lambda row: (-row["dice"], row["patient_id"]))
    best = ranked[:2]
    best_ids = {row["patient_id"] for row in best}
    worst = [row for row in reversed(ranked) if row["patient_id"] not in best_ids][:2]
    return [(f"Best {i + 1}", row) for i, row in enumerate(best)] + [
        (f"Worst {i + 1}", row) for i, row in enumerate(worst)
    ]


def read_loss_history(metrics_path):
    """Read epoch metrics, ignoring step-only rows and merging split CSV rows."""
    history = {"train": {}, "val": {}}
    if not Path(metrics_path).is_file():
        return history
    with open(metrics_path, newline="") as file:
        for row in csv.DictReader(file):
            if not row.get("epoch"):
                continue
            epoch = int(float(row["epoch"])) + 1
            for split, columns in (
                ("train", ("train_loss_epoch", "train_loss")),
                ("val", ("val_loss", "val_loss_epoch")),
                ("val_nodule", ("val_loss_nodule", "val_loss_nodule_epoch")),
            ):
                value = next((row.get(key) for key in columns if row.get(key)), None)
                if value is not None and np.isfinite(float(value)):
                    history.setdefault(split, {})[epoch] = float(value)
    return history


def axial_example(image, target, prediction, affine, lung_target=None, lung_prediction=None):
    """Choose a target slice in RAS and align optional lung overlays to it."""
    from nibabel.orientations import (
        apply_orientation, axcodes2ornt, io_orientation, ornt_transform,
    )

    orientation = ornt_transform(io_orientation(affine), axcodes2ornt(("R", "A", "S")))
    image, target, prediction = [
        apply_orientation(np.asarray(array), orientation)
        for array in (image, target, prediction)
    ]
    slice_index = int(target.sum(axis=(0, 1)).argmax())
    # Copy 2D slices so they do not hold entire 3D volumes alive.
    voxel_sizes = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    canonical_sizes = voxel_sizes[np.argsort(orientation[:, 0])]
    example = {
        "image": image[:, :, slice_index].T.copy(),
        "target": target[:, :, slice_index].T.copy(),
        "prediction": prediction[:, :, slice_index].T.copy(),
        "slice_index": slice_index,
        "aspect": float(canonical_sizes[1] / canonical_sizes[0]),
    }
    for key, mask in (("lung_target", lung_target), ("lung_prediction", lung_prediction)):
        if mask is not None:
            oriented = apply_orientation(np.asarray(mask), orientation)
            example[key] = oriented[:, :, slice_index].T.copy()
    return example


def evaluate_validation_by_class(
    model, dataset, batch_size, device, forward_context, classes="nodule",
):
    """One inference pass with independent metrics and examples for each class."""
    import torch
    from monai.data import DataLoader

    validate_classes(classes)
    reported_classes = class_ids(classes, include_background=classes == "lung+nodule")
    results = {
        name: {"class_id": class_id, "records": [], "examples": {}, "excluded": 0}
        for name, class_id in reported_classes.items()
    }
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False,
    )
    model.to(device)
    model.eval()
    processed = 0
    with torch.inference_mode():
        for batch in loader:
            labels = batch["label"]
            labels = labels.as_tensor() if hasattr(labels, "as_tensor") else labels
            images = batch["image"]
            image_tensor = images.as_tensor() if hasattr(images, "as_tensor") else images
            # Binary mode keeps the original positive-case evaluation policy.
            # Joint mode must also evaluate nodule-negative cases for lung/background.
            if classes == "nodule":
                indices = (labels.flatten(1).sum(1) > 0).nonzero(as_tuple=True)[0].tolist()
                results["nodule"]["excluded"] += len(labels) - len(indices)
            else:
                indices = list(range(len(labels)))
            if indices:
                with forward_context():
                    predictions = predict_labels(model(image_tensor[indices].to(device)), classes)
                device_labels = labels[indices].to(device)
                for name, class_id in reported_classes.items():
                    result = results[name]
                    targets = device_labels == class_id
                    predicted = predictions == class_id
                    target_counts = targets.flatten(1).sum(1).tolist()
                    predicted_counts = predicted.flatten(1).sum(1).tolist()
                    intersections = (targets & predicted).flatten(1).sum(1).tolist()
                    for output_index, input_index in enumerate(indices):
                        metrics = case_metrics(
                            intersections[output_index], predicted_counts[output_index],
                            target_counts[output_index],
                        )
                        if metrics is None:
                            result["excluded"] += 1
                            continue
                        record = {
                            "patient_id": batch["patient_id"][input_index],
                            "case_id": batch["case_id"][input_index],
                            "dataset": batch["dataset"][input_index],
                            **metrics,
                        }
                        result["records"].append(record)
                        selected_ids = {
                            row["patient_id"] for _, row in select_examples(result["records"])
                        }
                        result["examples"] = {
                            key: value for key, value in result["examples"].items()
                            if key in selected_ids
                        }
                        if record["patient_id"] in selected_ids:
                            image = images[input_index]
                            lung_overlays = {}
                            if classes == "lung+nodule" and name == "nodule":
                                # Both structures share the nodule-selected case and slice.
                                lung_overlays = {
                                    "lung_target": (device_labels[output_index, 0] == 1).cpu().numpy(),
                                    "lung_prediction": (predictions[output_index, 0] == 1).cpu().numpy(),
                                }
                            result["examples"][record["patient_id"]] = axial_example(
                                image[0].detach().cpu().numpy(),
                                targets[output_index, 0].cpu().numpy(),
                                predicted[output_index, 0].cpu().numpy(),
                                image.affine.detach().cpu().numpy(),
                                **lung_overlays,
                            )
            processed += len(labels)
            if processed % 10 < len(labels) or processed == len(dataset):
                print(f"Report: validation cases {processed}/{len(dataset)}", flush=True)
    return results


def evaluate_validation(model, dataset, batch_size, device, forward_context):
    """Compatibility entry point returning the binary nodule report results."""
    result = evaluate_validation_by_class(
        model, dataset, batch_size, device, forward_context, classes="nodule",
    )["nodule"]
    return result["records"], result["examples"], result["excluded"]


def render_report(output_path, history, summary, records, examples):
    """Render a headless PNG with loss curves, mean metrics, and paired overlays."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    figure = Figure(figsize=(15, 16), layout="constrained")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(3, 4, height_ratios=(1.05, 1.4, 1.4))
    report_class = summary.get("reported_class", "nodule")
    figure.suptitle(f"{summary['experiment']} - {report_class} report", fontsize=20)
    loss_axis = figure.add_subplot(grid[0, :2])
    for split, label in (("train", "Training (global)"), ("val", "Validation (global)"),
                         ("val_nodule", "Validation (nodule vs rest)")):
        values = sorted(history.get(split, {}).items())
        if values:
            loss_axis.plot(*zip(*values), marker="o", label=label)
    if any(history.values()):
        loss_axis.legend()
    else:
        loss_axis.text(
            0.5, 0.5, "No epoch loss history available",
            ha="center", transform=loss_axis.transAxes,
        )
    loss_axis.set(xlabel="Epoch", ylabel="Weighted loss", title="Recorded epoch losses")
    loss_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    loss_axis.grid(alpha=0.25)

    table_axis = figure.add_subplot(grid[0, 2:])
    table_axis.axis("off")
    table_axis.set_title("Best checkpoint — validation metrics", pad=20)
    per_class = summary.get("per_class", {report_class: summary})
    table_rows = []
    for name, values in per_class.items():
        means = values["mean_metrics"]
        table_rows.append([
            name, str(values["eligible_cases"]), str(values["excluded_empty_cases"]),
            *[f"{means[key]:.4f}" if means[key] is not None else "N/A" for key in METRICS],
        ])
    table = table_axis.table(
        cellText=table_rows,
        colLabels=["Class", "Cases", "Excluded", "Dice", "Precision", "Recall"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    table_axis.text(
        0.5, 0.08,
        "Each class averaged over cases with that class in ground truth\n"
        f"Predictions: {summary.get('prediction_rule', 'sigmoid(logit) > 0.5')}\n"
        f"Checkpoint: {summary['checkpoint_monitor']} ({summary['checkpoint_mode']})",
        ha="center", va="center", transform=table_axis.transAxes, fontsize=9,
    )

    selected = select_examples(records)
    for slot in range(4):
        grid_row, start_column = 1 + slot // 2, (slot % 2) * 2
        if slot >= len(selected):
            axis = figure.add_subplot(grid[grid_row, start_column:start_column + 2])
            axis.axis("off")
            axis.text(
                0.5, 0.5, "Not enough eligible cases for another example",
                ha="center", va="center", transform=axis.transAxes,
            )
            continue
        rank, record = selected[slot]
        example = examples[record["patient_id"]]
        title = (
            f"{rank}: {record['patient_id']}\n"
            f"{report_class.capitalize()} Dice {record['dice']:.3f} | precision {record['precision']:.3f} | "
            f"recall {record['recall']:.3f}\nAxial slice {example['slice_index']} (RAS)"
        )
        for column_offset, mask_key, color, label in (
            (0, "target", "lime", "Ground truth"),
            (1, "prediction", "magenta", "Prediction"),
        ):
            axis = figure.add_subplot(grid[grid_row, start_column + column_offset])
            axis.imshow(
                example["image"], cmap="gray", origin="lower", vmin=0,
                vmax=1, aspect=example["aspect"],
            )
            overlay = np.zeros((*example[mask_key].shape, 4), dtype=np.float32)
            lung_key = f"lung_{mask_key}"
            if lung_key in example:
                # A faint lung overlay preserves CT detail; nodules remain prominent.
                overlay[example[lung_key]] = (0, 1, 1, 0.25)
                legend = [Patch(color="cyan", label="Lung"), Patch(color=color, label="Nodule")]
            else:
                legend = [Patch(color=color, label=label)]
            overlay[example[mask_key]] = (
                (0, 1, 0, 0.55) if mask_key == "target" else (1, 0, 1, 0.55)
            )
            axis.imshow(overlay, origin="lower", aspect=example["aspect"])
            axis.set_title(f"{title}\n{label}" if column_offset == 0 else label, fontsize=9)
            axis.legend(handles=legend, loc="lower right", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.supxlabel(
        f"Examples ranked by {report_class} Dice; slices chosen by largest {report_class} ground-truth area.\n"
        "Loss curves include the cases used during training; report metrics exclude empty ground truth.",
        fontsize=10,
    )
    figure.savefig(output_path, dpi=160)
    figure.clear()


def summarize_class(result):
    """Summarize one class with its own eligible cases and ranked examples."""
    records, examples = result["records"], result["examples"]
    return {
        "class_id": result["class_id"],
        "eligible_cases": len(records),
        "excluded_empty_cases": result["excluded"],
        "mean_metrics": {
            key: float(np.mean([row[key] for row in records])) if records else None
            for key in METRICS
        },
        "examples": [
            {"rank": rank, **row,
             "axial_slice_index_ras": examples[row["patient_id"]]["slice_index"]}
            for rank, row in select_examples(records)
        ],
    }


def generate_training_report(
    model, datamodule, trainer, logger, checkpoint_callback, run_dir, experiment_name,
):
    """Reload the selected checkpoint and report class metrics in one val pass."""
    checkpoint_path = checkpoint_callback.best_model_path
    if not checkpoint_path:
        raise RuntimeError("Cannot create report: no best checkpoint was saved.")
    check_report_dependencies()
    logger.save()
    # This checkpoint belongs to this run and can contain MONAI metadata.
    best_model = type(model).load_from_checkpoint(
        checkpoint_path, map_location="cpu", weights_only=False,
    )
    classes = getattr(best_model, "classes", "nodule")
    print(f"\nGenerating report from best checkpoint: {checkpoint_path}", flush=True)
    results = evaluate_validation_by_class(
        best_model, datamodule.val_dataset, datamodule.hparams.eval_batch_size,
        trainer.strategy.root_device, trainer.precision_plugin.forward_context,
        classes=classes,
    )
    per_class = {name: summarize_class(result) for name, result in results.items()}
    prediction_rule = "sigmoid(logit) > 0.5" if classes == "nodule" else "argmax over class logits"
    summary = {
        "experiment": experiment_name,
        "classes": classes,
        "reported_class": "nodule",
        "checkpoint": str(checkpoint_path),
        "checkpoint_monitor": checkpoint_callback.monitor,
        "checkpoint_mode": checkpoint_callback.mode,
        # Top-level metrics remain nodule metrics for existing consumers.
        **per_class["nodule"],
        "per_class": per_class,
        "prediction_rule": prediction_rule,
        "prediction_threshold": "sigmoid(logit) > 0.5" if classes == "nodule" else None,
        "averaging": "arithmetic mean of per-case metrics; empty GT excluded per class",
        "precision_when_no_predicted_foreground": 0.0,
        "lung_definition": "class 1 excludes nodule voxels" if classes == "lung+nodule" else None,
    }
    report_dir = Path(run_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "validation_metrics.csv").open("w", newline="") as file:
        fields = ["patient_id", "case_id", "dataset"]
        if classes == "lung+nodule":
            fields += ["class", "class_id"]
        writer = csv.DictWriter(file, fieldnames=[*fields, *METRICS])
        writer.writeheader()
        for name, result in results.items():
            for row in result["records"]:
                if classes == "lung+nodule":
                    row = {**row, "class": name, "class_id": result["class_id"]}
                writer.writerow(row)
    with (report_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
    history = read_loss_history(Path(logger.log_dir) / "metrics.csv")
    output_path = report_dir / "report.png"
    render_report(
        output_path, history, summary,
        results["nodule"]["records"], results["nodule"]["examples"],
    )
    print(f"Report saved: {output_path.resolve()}", flush=True)
    for name, values in per_class.items():
        print(
            f"{name} ({values['eligible_cases']} eligible cases): {values['mean_metrics']}",
            flush=True,
        )
    return output_path
