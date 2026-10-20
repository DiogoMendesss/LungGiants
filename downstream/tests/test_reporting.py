"""Regression tests for validation-only report evaluation and PNG generation."""

import csv
import json
from contextlib import nullcontext
from types import SimpleNamespace

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from monai.data import Dataset, MetaTensor
from torch.utils.data import DataLoader, TensorDataset

from reporting import (
    axial_example, case_metrics, evaluate_validation, generate_training_report,
    read_loss_history, render_report, select_examples,
)


def make_dataset():
    cases = []
    # Five positive cases with declining voxel overlap, plus one excluded empty mask.
    for i in range(6):
        target = torch.zeros((1, 6, 7, 4))
        image = torch.full_like(target, 0.1)
        if i < 5:
            target[0, 1:3, 1:3, 2] = 1
            overlap = 4 - i
            coordinates = [(1, 1), (1, 2), (2, 1), (2, 2)]
            for x, y in coordinates[:overlap]:
                image[0, x, y, 2] = 0.9
            # Include a false positive to distinguish precision and recall.
            if i in (1, 2):
                image[0, 4, 5, 1] = 0.9
        else:
            image[:] = 0.9
        cases.append({
            "image": MetaTensor(image, affine=torch.diag(torch.tensor([1., 2., 3., 1.]))),
            "label": MetaTensor(target, affine=torch.diag(torch.tensor([1., 2., 3., 1.]))),
            "patient_id": f"synthetic/case_{i}",
            "case_id": f"case_{i}",
            "dataset": "synthetic",
        })
    return Dataset(cases)


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.observed = []

    def forward(self, images):
        self.observed.append((self.training, torch.is_inference_mode_enabled(), len(images)))
        return images - 0.5


class TinyCheckpointModel(pl.LightningModule):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.threshold = torch.nn.Parameter(torch.tensor(threshold))

    def forward(self, images):
        return images - self.threshold

    def training_step(self, batch, batch_idx):
        loss = self.threshold.square()
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", self.threshold.square())
        self.log("val_dice", 0.8)

    def on_save_checkpoint(self, checkpoint):
        # Reproduce metadata present in real MONAI training checkpoints.
        from monai.utils.enums import MetaKeys

        checkpoint["monai_metadata"] = {MetaKeys.AFFINE: torch.eye(4)}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


def test_voxel_metrics_and_empty_masks():
    assert case_metrics(0, 10, 0) is None
    assert case_metrics(4, 4, 4) == {"dice": 1., "precision": 1., "recall": 1.}
    assert case_metrics(0, 0, 4) == {"dice": 0., "precision": 0., "recall": 0.}
    metrics = case_metrics(2, 5, 4)
    assert metrics == pytest.approx({"dice": 4 / 9, "precision": 2 / 5, "recall": 0.5})


def test_loss_history_merges_rows_and_ignores_step_values(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text(
        "epoch,step,train_loss_step,train_loss_epoch,val_loss\n"
        "0,0,9,,\n0,9,,,0.4\n0,9,,0.5,\n"
        "1,10,8,,\n1,19,,0.3,0.2\n"
    )
    assert read_loss_history(path) == {"train": {1: 0.5, 2: 0.3}, "val": {1: 0.4, 2: 0.2}}


@pytest.mark.parametrize("count", range(7))
def test_selection_has_no_duplicate_cases_even_with_ties(count):
    records = [{"patient_id": str(i), "dice": 0.5} for i in range(count)]
    selected = select_examples(records)
    ids = [row["patient_id"] for _, row in selected]
    assert len(ids) == min(4, count)
    assert len(ids) == len(set(ids))
    assert selected == select_examples(list(reversed(records)))


def test_axial_selection_uses_affine_orientation_and_preserves_alignment():
    # Input voxel axis 0 points superior, axis 1 anterior, axis 2 left.
    affine = np.array([[0, 0, -3, 9], [0, 2, 0, 0], [4, 0, 0, 0], [0, 0, 0, 1.]])
    target = np.zeros((4, 5, 6), dtype=bool)
    target[2, 1:4, 2:5] = True
    example = axial_example(target.astype(float), target, target, affine)
    assert example["slice_index"] == 2
    assert example["image"].shape == (5, 6)
    assert np.array_equal(example["image"], example["target"])
    assert np.array_equal(example["target"], example["prediction"])
    assert example["aspect"] == pytest.approx(2 / 3)
    assert example["image"].base is None


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_validation_pass_excludes_empty_cases_and_retains_only_extremes(batch_size):
    model = RecordingModel()
    dataset = make_dataset()
    records, examples, excluded = evaluate_validation(
        model, dataset, batch_size, torch.device("cpu"), nullcontext,
    )
    assert excluded == 1
    assert len(records) == 5
    assert sum(n for _, _, n in model.observed) == 5
    assert all(not training and inference for training, inference, _ in model.observed)
    assert [row["dice"] for row in records] == pytest.approx([1., 0.75, 4 / 7, 0.4, 0.])
    assert records[1]["precision"] == pytest.approx(0.75)
    assert records[1]["recall"] == pytest.approx(0.75)
    assert records[-1]["precision"] == 0
    assert set(examples) == {
        "synthetic/case_0", "synthetic/case_1", "synthetic/case_3", "synthetic/case_4",
    }


def test_empty_validation_can_render_without_nan(tmp_path):
    summary = {
        "experiment": "no_nodules", "eligible_cases": 0, "excluded_empty_cases": 2,
        "mean_metrics": dict.fromkeys(("dice", "precision", "recall")),
        "checkpoint_monitor": "val_dice", "checkpoint_mode": "max",
    }
    output = tmp_path / "empty.png"
    render_report(output, {"train": {}, "val": {}}, summary, [], {})
    assert output.read_bytes().startswith(b"\x89PNG")


def test_real_checkpoint_reload_csv_logging_and_report_artifacts(tmp_path):
    model = TinyCheckpointModel()
    logger = CSVLogger(str(tmp_path), name="", version="logs")
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path / "checkpoints", monitor="val_dice", mode="max",
    )
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=2, logger=logger,
        callbacks=[checkpoint], enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, log_every_n_steps=1,
    )
    loader = DataLoader(TensorDataset(torch.zeros((1, 1))), batch_size=1)
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    # Deliberately change final weights: the report must reload the checkpoint.
    with torch.no_grad():
        model.threshold.fill_(2)
    datamodule = SimpleNamespace(
        val_dataset=make_dataset(), hparams=SimpleNamespace(eval_batch_size=2),
    )
    output = generate_training_report(
        model, datamodule, trainer, logger, checkpoint, tmp_path, "synthetic",
    )
    summary = json.loads((output.parent / "summary.json").read_text())
    assert output.read_bytes().startswith(b"\x89PNG")
    assert summary["checkpoint"] == checkpoint.best_model_path
    assert summary["eligible_cases"] == 5
    assert summary["excluded_empty_cases"] == 1
    assert summary["mean_metrics"]["dice"] == pytest.approx((1 + .75 + 4 / 7 + .4) / 5)
    assert summary["mean_metrics"]["precision"] == pytest.approx((1 + .75 + 2 / 3 + 1) / 5)
    assert summary["mean_metrics"]["recall"] == pytest.approx((1 + .75 + .5 + .25) / 5)
    assert len(summary["examples"]) == 4
    assert read_loss_history(tmp_path / "logs" / "metrics.csv") == {
        "train": {1: 0.25, 2: 0.25}, "val": {1: 0.25, 2: 0.25},
    }
    with (output.parent / "validation_metrics.csv").open(newline="") as file:
        assert len(list(csv.DictReader(file))) == 5


def test_lung_overlays_follow_nodule_slice_and_orientation():
    affine = np.array([[0, 0, -3, 9], [0, 2, 0, 0], [4, 0, 0, 0], [0, 0, 0, 1.]])
    target = np.zeros((4, 5, 6), dtype=bool)
    target[2, 1:3, 2:4] = True
    lung = np.zeros_like(target)
    lung[0] = True  # Largest lung slice differs from the selected nodule slice.
    lung[2, 3:5, 1:5] = True
    lung_prediction = np.zeros_like(target)
    lung_prediction[2, 0:2, 1:3] = True
    example = axial_example(
        target.astype(float), target, target, affine,
        lung_target=lung, lung_prediction=lung_prediction,
    )
    assert example["slice_index"] == 2
    # RAS swaps axis 0 with axis 2 and flips the original left-pointing axis 2.
    np.testing.assert_array_equal(example["lung_target"], lung[2, :, ::-1])
    np.testing.assert_array_equal(example["lung_prediction"], lung_prediction[2, :, ::-1])
    assert example["lung_target"].base is None
    assert example["lung_prediction"].base is None
