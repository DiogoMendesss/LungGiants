"""Regression tests for both segmentation targets and nodule-first monitoring."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import lightning.pytorch as pl
import nibabel as nib
import numpy as np
import pytest
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from monai.data import Dataset, MetaTensor
from torch.utils.data import DataLoader

from data.dataset_index import collect_cases
from data.datamodule import NoduleDataModule
from data.transforms import CheckPairGeometryd, ComposeSegmentationLabeld
from reporting import evaluate_validation_by_class, generate_training_report
from tasks import predict_labels
from train import SegmentationModule


def make_model(classes):
    return SegmentationModule(
        model_config={
            "spatial_dims": 3, "in_channels": 1, "out_channels": 99,
            "init_filters": 8, "blocks_down": [1, 1], "blocks_up": [1],
        },
        loss_config={"dice_weight": 1., "bce_weight": 1., "ce_weight": 1.},
        optimizer_config={"lr": 1e-4}, classes=classes,
    )


def test_index_requires_ts_only_in_joint_mode(tmp_path):
    folders = {"images": "ct", "nod_masks": "nod", "ts_masks": "missing_ts"}
    for key in ("images", "nod_masks"):
        directory = tmp_path / folders[key]
        directory.mkdir()
        (directory / "LIDC-IDRI-0001.nii.gz").touch()
    records = collect_cases(tmp_path, {"LIDC_IDRI": folders}, classes="nodule")
    assert records[0]["patient_id"] == "LIDC-IDRI-0001"
    assert "nod_mask" in records[0] and "ts_mask" not in records[0]
    with pytest.raises(FileNotFoundError):
        collect_cases(tmp_path, {"LIDC_IDRI": folders}, classes="lung+nodule")
    (tmp_path / "missing_ts").mkdir()
    (tmp_path / "missing_ts" / "wrong.nii.gz").touch()
    with pytest.raises(ValueError, match="ts_mask"):
        collect_cases(tmp_path, {"LIDC_IDRI": folders}, classes="lung+nodule")


def test_label_priority_and_all_lobe_ids():
    ts = MetaTensor(torch.tensor([0, 28, 29, 30, 31, 32, 33, 23]).reshape(1, 2, 2, 2))
    nod = torch.zeros_like(ts)
    nod.flatten()[0] = 7  # Outside lungs: must still be class 2.
    nod.flatten()[2] = 12  # Overlapping lung: nodule takes priority.
    data = {"nod_mask": nod, "ts_mask": ts}
    joint = ComposeSegmentationLabeld("lung+nodule")(data)
    assert joint["label"].flatten().tolist() == [2, 1, 2, 1, 1, 1, 0, 0]
    assert joint["label"].dtype == torch.long
    assert "nod_mask" not in joint and "ts_mask" not in joint
    assert torch.equal(joint["label"].affine, nod.affine)
    binary = ComposeSegmentationLabeld("nodule")({"nod_mask": nod})
    assert binary["label"].flatten().tolist() == [1, 0, 1, 0, 0, 0, 0, 0]
    assert binary["label"].dtype == torch.float32


@pytest.mark.parametrize("mask_key", ["nod_mask", "ts_mask"])
@pytest.mark.parametrize("mismatch", ["shape", "affine"])
def test_geometry_rejects_each_mask(mask_key, mismatch):
    data = {key: MetaTensor(torch.zeros((1, 4, 4, 4)))
            for key in ("image", "nod_mask", "ts_mask")}
    if mismatch == "shape":
        data[mask_key] = MetaTensor(torch.zeros((1, 3, 4, 4)))
    else:
        data[mask_key].affine[0, 3] = 10
    with pytest.raises(ValueError, match=mask_key):
        CheckPairGeometryd((4, 4, 4), "lung+nodule")(data)


def test_real_nifti_loading_and_shared_split(tmp_path):
    dataset = {"images": "ct", "nod_masks": "nod", "ts_masks": "ts"}
    for folder in dataset.values():
        (tmp_path / folder).mkdir()
    affine = np.diag([2., 2., 3., 1.])
    for case in ("LNDb-0001", "LNDb-0002", "LNDb-0003"):
        for key, folder in dataset.items():
            array = np.zeros((8, 8, 8), dtype=np.float32)
            if key == "images":
                array[:] = -500
            elif key == "nod_masks":
                array[2, 2, 2] = 7
            else:
                array[1:7, 1:7, 1:7] = 28
            nib.save(nib.Nifti1Image(array, affine), tmp_path / folder / f"{case}.nii.gz")
    # Separate test paths with another patient ID.
    test_folders = {}
    for key, folder in dataset.items():
        test_folders[key] = f"test_{folder}"
        (tmp_path / test_folders[key]).mkdir()
        original = tmp_path / folder / "LNDb-0001.nii.gz"
        (tmp_path / test_folders[key] / "LNDb-0004.nii.gz").write_bytes(original.read_bytes())
    settings = {"validation_fraction": 1/3, "seed": 42, "stratify_by_dataset": True}
    config = {
        "root": str(tmp_path), "split": settings,
        "real": {"train": {"LNDb": dataset}, "test": {"LNDb": test_folders}},
    }
    config_path, experiment_path = tmp_path / "data.yaml", tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config))
    experiment_path.write_text(yaml.safe_dump({"data": {
        "real_percent": 100, "synthetic_percent": 0, "synthetic_dataset": None,
    }}))
    split_path = tmp_path / "split.json"
    patient_ids = []
    for mode in ("nodule", "lung+nodule"):
        dm = NoduleDataModule(
            datasets_config=str(config_path), experiment_config=str(experiment_path),
            split_path=str(split_path), expected_shape=(8, 8, 8),
            classes=mode, num_workers=0, pin_memory=False,
        )
        dm.setup("fit")
        patient_ids.append([case["patient_id"] for case in dm.val_dataset.data])
        batch = next(iter(dm.val_dataloader()))
        assert batch["image"].shape == batch["label"].shape == (1, 1, 8, 8, 8)
        assert set(batch["label"].unique().tolist()) == ({0, 1} if mode == "nodule" else {0, 1, 2})
        assert torch.allclose(batch["image"], torch.full_like(batch["image"], .25))
        assert "nod_mask" not in batch and "ts_mask" not in batch
    assert patient_ids[0] == patient_ids[1]


def test_losses_decoding_and_independent_nodule_monitor():
    binary = make_model("nodule")
    labels = torch.tensor([0., 1., 0., 1.]).reshape(1, 1, 1, 2, 2)
    logits = (labels * 2 - 1) * 6
    assert torch.equal(predict_labels(logits, "nodule"), labels.long())
    assert torch.allclose(binary.compute_loss(logits, labels)[0], binary.compute_nodule_loss(logits, labels))
    joint = make_model("lung+nodule")
    target = torch.tensor([0, 1, 2, 0]).reshape(1, 1, 1, 2, 2)
    logits = torch.tensor([
        [6., -6., -6., 6.], [-6., 6., -6., -6.], [-6., -6., 6., -6.],
    ]).reshape(1, 3, 1, 2, 2).requires_grad_()
    total = joint.compute_loss(logits, target)[0]
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(logits.grad).all()
    assert all(torch.equal(score, torch.ones_like(score)) for score in joint.compute_dice(logits, target).values())
    swapped = logits.detach().clone()
    swapped[:, 0], swapped[:, 1] = logits[:, 1].detach(), logits[:, 0].detach()
    assert torch.allclose(joint.compute_nodule_loss(logits, target), joint.compute_nodule_loss(swapped, target))
    assert joint.compute_loss(swapped, target)[0] > total
    false_positive = logits.detach().clone()
    false_positive[:, 2, 0, 0, 0] = 15
    assert joint.compute_nodule_loss(false_positive, target) > joint.compute_nodule_loss(logits, target)
    missed = logits.detach().clone()
    missed[:, 2, 0, 1, 0] = -15
    assert joint.compute_nodule_loss(missed, target) > joint.compute_nodule_loss(logits, target)


class FixedJointModel(torch.nn.Module):
    def forward(self, images):
        # Image voxel values encode the desired predicted class for this fixture.
        return torch.nn.functional.one_hot(images[:, 0].long(), 3).movedim(-1, 1).float() * 10


def test_joint_report_does_not_pool_classes_or_drop_nodule_negative_cases():
    cases = []
    for i, (target, predicted) in enumerate((
        ([0, 1, 1, 2], [0, 1, 2, 2]),
        ([0, 1, 1, 0], [0, 1, 1, 0]),
    )):
        cases.append({
            "image": MetaTensor(torch.tensor(predicted).float().reshape(1, 1, 2, 2)),
            "label": MetaTensor(torch.tensor(target).reshape(1, 1, 2, 2)),
            "patient_id": str(i), "case_id": str(i), "dataset": "fixture",
        })
    results = evaluate_validation_by_class(
        FixedJointModel(), Dataset(cases), 2, torch.device("cpu"), nullcontext,
        classes="lung+nodule",
    )
    assert len(results["nodule"]["records"]) == 1
    assert results["nodule"]["excluded"] == 1
    assert len(results["lung"]["records"]) == 2
    assert len(results["background"]["records"]) == 2
    assert results["nodule"]["records"][0]["dice"] == pytest.approx(2/3)
    assert results["nodule"]["records"][0]["precision"] == pytest.approx(.5)
    assert results["lung"]["records"][0]["recall"] == pytest.approx(.5)
    assert results["background"]["records"][0]["dice"] == 1
    example = results["nodule"]["examples"]["0"]
    assert "lung_target" in example and "lung_prediction" in example
    assert example["lung_target"].shape == example["target"].shape


@pytest.mark.parametrize("classes", ["nodule", "lung+nodule"])
def test_training_checkpoint_monitors_and_report(tmp_path, classes):
    torch.set_num_threads(1)
    model = make_model(classes)
    labels = torch.zeros((1, 8, 8, 8), dtype=torch.long)
    if classes == "lung+nodule":
        labels[:, 1:7, 1:7, 1:7] = 1
        labels[:, 3:5, 3:5, 3:5] = 2
    else:
        labels[:, 3:5, 3:5, 3:5] = 1
    dataset = Dataset([{
        "image": MetaTensor(torch.rand((1, 8, 8, 8))), "label": MetaTensor(labels),
        "patient_id": "LNDb-0001", "case_id": "LNDb-0001", "dataset": "LNDb",
    }])
    loader = DataLoader(dataset, batch_size=1)
    logger = CSVLogger(str(tmp_path), name="", version="logs")
    checkpoint = ModelCheckpoint(dirpath=tmp_path / "checkpoints", monitor="val_dice_nodule", mode="max")
    stopping = EarlyStopping(monitor="val_loss_nodule", mode="min", patience=2)
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, logger=logger,
        callbacks=[checkpoint, stopping], enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, log_every_n_steps=1,
    )
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    assert "val_loss_nodule" in trainer.callback_metrics
    assert "val_dice_nodule" in trainer.callback_metrics
    restored = SegmentationModule.load_from_checkpoint(checkpoint.best_model_path, weights_only=False)
    assert restored.classes == classes
    assert restored.hparams.model_config["out_channels"] == (1 if classes == "nodule" else 3)
    datamodule = SimpleNamespace(val_dataset=dataset, hparams=SimpleNamespace(eval_batch_size=1))
    output = generate_training_report(model, datamodule, trainer, logger, checkpoint, tmp_path, classes)
    summary = json.loads((output.parent / "summary.json").read_text())
    assert summary["classes"] == classes
    assert output.read_bytes().startswith(b"\x89PNG")
    if classes == "lung+nodule":
        assert set(summary["per_class"]) == {"background", "lung", "nodule"}
        assert {path.name for path in output.parent.glob("*.png")} == {"report.png"}
        import csv
        with (output.parent / "validation_metrics.csv").open() as file:
            assert {row["class"] for row in csv.DictReader(file)} == {"background", "lung", "nodule"}
