"""Evaluation protocol tests, including real checkpoint/inference/report plumbing."""

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

import evaluate
from evaluation_metrics import (
    match_components, mean_std, read_attributes, score_nodules,
    summarize_attributes, validate_instances, voxel_metrics,
)


def metadata(label, size=1):
    return {"label_id": label, "annotation": f"case_nodule{label}.npz", "size_class": size,
            "sphericity_class": 3, "spiculation_class": 1, "attribute_issues": []}


def test_26_connectivity_keeps_entire_component_and_discards_island():
    gt = np.zeros((8, 8, 8), dtype=int)
    gt[1, 1, 1] = 7
    pred = gt > 0
    pred[2, 2, 2] = True  # Only diagonally connected: must remain assigned to 7.
    pred[6, 6, 6] = True
    rows, diagnostics, assigned = score_nodules(pred, gt, {7: metadata(7)})
    assert assigned[2, 2, 2] == 7
    assert assigned[6, 6, 6] == 0
    assert rows[0]["dsc"] == pytest.approx(2 / 3)
    assert rows[0]["predicted_voxels"] == 2
    assert diagnostics == [{"component_id": 2, "voxel_count": 1,
                            "reason": "no_gt_overlap", "gt_labels": []}]
    assert voxel_metrics(pred, gt > 0)["dsc"] == 0.5  # Full-case score retains the island.


def test_fragmented_prediction_is_unioned_and_missed_gt_is_zero():
    gt = np.zeros((9, 9, 9), dtype=int)
    gt[1:6, 1, 1] = 1
    gt[7, 7, 7] = 2
    pred = np.zeros_like(gt, bool)
    pred[1, 1, 1] = pred[5, 1, 1] = True
    rows, diagnostics, _ = score_nodules(pred, gt, {1: metadata(1), 2: metadata(2)})
    assert not diagnostics
    assert rows[0]["assigned_components"] == 2
    assert rows[0]["dsc"] == pytest.approx(4 / 7)
    assert rows[1]["status"] == "missed"
    assert rows[1]["dsc"] == 0


def test_merged_component_is_discarded_and_every_affected_label_flagged():
    gt = np.zeros((6, 6, 6), dtype=int)
    gt[1, 1, 1], gt[3, 1, 1] = 1, 2
    pred = np.zeros_like(gt, bool)
    pred[1:4, 1, 1] = True
    rows, diagnostics, assigned = score_nodules(pred, gt, {1: metadata(1), 2: metadata(2)})
    assert not assigned.any()
    assert all(row["status"] == "ambiguous_component_discarded" for row in rows)
    assert all(row["dsc"] == 0 for row in rows)
    assert diagnostics[0]["gt_labels"] == [1, 2]
    assert diagnostics[0]["reason"] == "multiple_gt_labels"


def test_empty_prediction_missing_grid_label_and_empty_gt():
    gt = np.zeros((4, 4, 4), int)
    gt[1, 1, 1] = 1
    rows, _, _ = score_nodules(np.zeros_like(gt), gt, {1: metadata(1), 2: metadata(2)})
    assert rows[0]["dsc"] == 0
    assert rows[1]["dsc"] is None
    assert rows[1]["status"] == "missing_from_evaluation_grid"
    result = voxel_metrics(np.zeros_like(gt), gt > 0)
    assert (result["dsc"], result["precision"], result["recall"]) == (0, 0, 0)
    result = voxel_metrics(gt > 0, np.zeros_like(gt))
    assert result["dsc"] is None and result["status"] == "empty_gt"
    assert result["predicted_voxels"] == 1
    assigned, _, _, diagnostics = match_components(np.zeros_like(gt), np.zeros_like(gt))
    assert not assigned.any() and not diagnostics
    with pytest.raises(ValueError, match="integer"):
        validate_instances(gt.astype(float) + 0.1)


def test_attributes_mapping_and_invalid_ratings(tmp_path):
    np.savez(tmp_path / "LIDC-IDRI-0003_nodule0.npz", patient_id="LIDC-IDRI-0003",
             equiv_diameter_mm=2, char_sphericity=[2, 3], char_spiculation=[0, np.nan, 4])
    record = read_attributes(tmp_path, "LIDC-IDRI-0003", "LIDC_IDRI")[1]
    assert record["size_class"] == 2 and record["sphericity_class"] == 3
    assert record["spiculation_class"] == 4
    assert record["attribute_issues"] == ["spiculation:invalid_ratings_ignored"]
    np.savez(tmp_path / "LNDb-0003_nodule5.npz", patient_id="LNDb-0003",
             equiv_diameter_mm=np.nan, char_sphericity=[], char_spiculation=[-1])
    record = read_attributes(tmp_path, "LNDb-0003", "LNDb")[5]
    assert all(record[f"{key}_class"] is None for key in ("size", "sphericity", "spiculation"))
    assert len(record["attribute_issues"]) == 3
    np.savez(tmp_path / "LNDb-0003_nodule1.npz", patient_id="LNDb-0003",
             equiv_diameter_mm=1, char_sphericity=[2.5], char_spiculation=[1.5])
    record = read_attributes(tmp_path, "LNDb-0003", "LNDb")[1]
    assert record["sphericity_class"] == 3 and record["spiculation_class"] == 2
    assert record["attribute_issues"] == []


def test_macro_summary_includes_misses_but_not_missing_grid():
    rows = [{**metadata(i), "case_id": f"case{i}", "dsc": score, "status": status}
            for i, score, status in [(1, 1.0, "scored"), (2, 0.0, "missed"),
                                     (3, None, "missing_from_evaluation_grid")]]
    summary = summarize_attributes(rows)
    assert len(summary) == 13
    size1 = summary[0]
    assert size1["n_nodules"] == 3 and size1["n_scored"] == 2
    assert size1["n_missing_from_grid"] == 1 and size1["n_missed"] == 1
    assert size1["dsc_mean"] == 0.5
    assert size1["dsc_std"] == pytest.approx(np.sqrt(0.5))
    assert summary[1]["n_nodules"] == 0 and summary[1]["dsc_mean"] is None
    assert mean_std([0.3]) == {"n": 1, "mean": 0.3, "std": None}


def save_volume(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(values.astype(np.float32), np.eye(4)), path)


def make_fixture(tmp_path):
    root = tmp_path / "LungNodule-nifti"
    annotation_root = tmp_path / "Lung_Nodule"
    config_dir = tmp_path / "run" / "configs"
    config_dir.mkdir(parents=True)
    config = {"root": str(root), "real": {}}
    for split in ("train", "test"):
        base = f"preprocessed/{split}/LNDb"
        config["real"][split] = {"LNDb": {"images": f"{base}/ct", "nod_masks": f"{base}/nodulemask",
                                          "ts_masks": f"{base}/ts_out"}}
    for split, case in (("train", "LNDb-0001"), ("test", "LNDb-0002")):
        folders = config["real"][split]["LNDb"]
        image = np.full((16, 16, 16), 0.25)
        mask = np.zeros_like(image)
        mask[4:6, 4:6, 4:6] = 1
        save_volume(root / folders["images"] / f"{case}.nii.gz", image)
        save_volume(root / folders["nod_masks"] / f"{case}.nii.gz", mask)
        save_volume(root / folders["ts_masks"] / f"{case}.nii.gz", np.full_like(mask, 28))
        folder = annotation_root / split / "LNDb" / "annotations"
        folder.mkdir(parents=True)
        np.savez(folder / f"{case}_nodule1.npz", patient_id=case, equiv_diameter_mm=1,
                 char_sphericity=[3], char_spiculation=[1])
    saved = {"train": [], "val": ["LNDb-0001"], "test": ["LNDb-0002"]}
    (config_dir / "real_split.json").write_text(json.dumps(saved))
    (config_dir / "datasets.yaml").write_text(yaml.safe_dump(config))
    training = {"data": {"expected_shape": [16, 16, 16], "in_min": 0.0, "in_max": 1.0},
                "trainer": {"precision": "32-true"}}
    (config_dir / "training.yaml").write_text(yaml.safe_dump(training))
    return root, annotation_root, config_dir, config, saved


class BackgroundModel(torch.nn.Module):
    def forward(self, image):
        return torch.cat((torch.ones_like(image), torch.zeros_like(image), torch.zeros_like(image)), dim=1)


def test_joint_reports_separate_splits_and_full_precision_recall(tmp_path):
    root, annotations, config_dir, config, saved = make_fixture(tmp_path)
    cases = evaluate.evaluation_cases(config, saved, root, "lung+nodule", annotations)
    assert [row["case_id"] for row in cases["val"]] == ["LNDb-0001"]
    for split in ("val", "test"):
        output = tmp_path / split
        summary = evaluate.evaluate_split(BackgroundModel(), cases[split], "lung+nodule", split,
                                          output, {"expected_shape": [16, 16, 16]}, torch.device("cpu"),
                                          "32-true", {"model_path": "test-model"}, save_masks=True)
        assert summary["n_cases"] == 1 and summary["n_scored_nodules"] == 1
        assert not summary["provisional"]
        for name in ("lung", "nodule"):
            for metric in ("dsc", "precision", "recall"):
                assert summary["case_metrics"][name][metric]["mean"] == 0
        rows = list(csv.DictReader((output / "case_metrics.csv").open()))
        assert int(rows[0]["target_voxels"]) == 16 ** 3 - 8  # Lung excludes nodules.
        assert len(list(csv.DictReader((output / "attribute_summary.csv").open()))) == 13
        assert len(list((output / "masks" / "LNDb").glob("*.nii.gz"))) == 2
        assert "Mean ± sample std" in (output / "report.html").read_text(encoding="utf-8")
        assert json.loads((output / "summary.json").read_text())["split"] == split


@pytest.mark.parametrize("selection", [None, "both", "val", "test"])
def test_cli_loads_actual_checkpoint_and_writes_selected_reports(tmp_path, selection):
    from models.segresnet import SegResNet

    _, _, config_dir, datasets_config, saved = make_fixture(tmp_path)
    selected = ("val", "test") if selection in (None, "both") else (selection,)
    if len(selected) == 1:
        # The unrequested split need not have a data config or manifest entry.
        datasets_config["real"].pop("test" if selection == "val" else "train")
        (config_dir / "datasets.yaml").write_text(yaml.safe_dump(datasets_config))
        manifest = {split: [{"case_id": case} for case in saved[split]] for split in selected}
        (config_dir / "selected_cases.json").write_text(json.dumps(manifest))
    config = {"spatial_dims": 3, "in_channels": 1, "out_channels": 1, "init_filters": 8,
              "blocks_down": [1, 1], "blocks_up": [1]}
    network = SegResNet(**config)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
    checkpoint = config_dir.parent / "checkpoints" / "tiny.ckpt"
    checkpoint.parent.mkdir()
    torch.save({"hyper_parameters": {"model_config": config, "classes": "nodule"},
                "state_dict": {f"network.{key}": value for key, value in network.state_dict().items()}}, checkpoint)
    output = tmp_path / "reports"
    args = ["--model", str(checkpoint), "--output-path", str(output), "--device", "cpu"]
    if selection is not None:
        args.extend(["--set", selection])
    evaluate.main(args)
    expected_folders = {"validation" if split == "val" else "test" for split in selected}
    assert {path.name for path in output.iterdir()} == expected_folders
    for split in expected_folders:
        summary = json.loads((output / split / "summary.json").read_text())
        assert summary["case_metrics"]["nodule"]["dsc"]["mean"] == 0
        assert summary["case_metrics"]["lung"]["n_unavailable"] == 1
        assert summary["case_metrics"]["lung"]["dsc"]["mean"] is None
        assert summary["attribute_metrics"][0]["dsc_mean"] == 0


def test_report_marks_merges_provisional_without_filtering_case_metrics(tmp_path):
    root, annotations, _, config, saved = make_fixture(tmp_path)
    cases = evaluate.evaluation_cases(config, saved, root, "lung+nodule", annotations)
    case = cases["val"][0]
    gt = np.zeros((16, 16, 16))
    gt[4, 4, 4], gt[6, 4, 4] = 1, 2
    save_volume(Path(case["nod_mask"]), gt)
    np.savez(Path(case["annotation_dir"]) / "LNDb-0001_nodule2.npz", patient_id="LNDb-0001",
             equiv_diameter_mm=2, char_sphericity=[4], char_spiculation=[2])

    class BridgeModel(torch.nn.Module):
        def forward(self, image):
            logits = torch.cat((torch.ones_like(image), torch.zeros_like(image), torch.zeros_like(image)), dim=1)
            logits[:, 2, 4:7, 4, 4] = 2
            return logits

    output = tmp_path / "merged_report"
    with pytest.warns(UserWarning, match="overlapping multiple GT"):
        summary = evaluate.evaluate_split(BridgeModel(), cases["val"], "lung+nodule", "val", output,
                                          {"expected_shape": [16, 16, 16]}, torch.device("cpu"),
                                          "32-true", {"model_path": "test-model"})
    assert summary["provisional"] and summary["components"]["ambiguous_count"] == 1
    assert summary["case_metrics"]["nodule"]["dsc"]["mean"] == pytest.approx(4 / 5)
    assert summary["attribute_metrics"][0]["dsc_mean"] == 0
    assert summary["attribute_metrics"][1]["dsc_mean"] == 0
    assert "PROVISIONAL" in (output / "report.html").read_text(encoding="utf-8")


def test_split_overlap_and_geometry_mismatch_are_rejected(tmp_path):
    root, annotations, _, config, saved = make_fixture(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        evaluate.evaluation_cases(config, {**saved, "train": saved["val"]}, root, "lung+nodule", annotations)
    cases = evaluate.evaluation_cases(config, saved, root, "lung+nodule", annotations)
    bad = cases["val"][0]["ts_mask"]
    img = nib.load(bad)
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), np.diag([2., 1., 1., 1.])), bad)
    with pytest.raises(RuntimeError):  # MONAI wraps the geometry validation error.
        evaluate.evaluate_split(BackgroundModel(), cases["val"], "lung+nodule", "val", tmp_path / "out",
                                 {"expected_shape": [16, 16, 16]}, torch.device("cpu"), "32-true",
                                 {"model_path": "test"})
