"""Train full-volume SegResNet for binary lung nodule segmentation."""

import argparse
from datetime import datetime
from pathlib import Path
import shutil

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from monai.losses import DiceLoss
from torch import nn

from data.datamodule import NoduleDataModule
from models.segresnet import SegResNet


class SegmentationModule(pl.LightningModule):
    """Network, loss, metrics, and optimization logic."""

    def __init__(self, model_config, loss_config, optimizer_config):
        super().__init__()

        # Save constructor arguments in checkpoints.
        self.save_hyperparameters()

        if model_config["out_channels"] != 1:
            raise ValueError(
                "Binary segmentation requires one output channel."
            )

        self.network = SegResNet(**model_config)

        # The single output channel represents nodule foreground.
        self.dice_loss = DiceLoss(
            sigmoid=True,
            include_background=True,
            to_onehot_y=False,
            batch=False,
            reduction="mean",
        )
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.dice_weight = loss_config["dice_weight"]
        self.bce_weight = loss_config["bce_weight"]

        if (
            self.dice_weight < 0
            or self.bce_weight < 0
            or self.dice_weight + self.bce_weight == 0
        ):
            raise ValueError(
                "Loss weights must be nonnegative and not both zero."
            )

    def forward(self, images):
        """Return logits with shape [B, 1, X, Y, Z]."""
        return self.network(images)

    def compute_loss(self, logits, labels):
        """Weighted sum of soft Dice loss and binary cross-entropy."""
        logits = logits.float()
        labels = labels.float()

        dice = self.dice_loss(logits, labels)
        bce = self.bce_loss(logits, labels)

        total = self.dice_weight * dice + self.bce_weight * bce
        return total, dice, bce

    @staticmethod
    def compute_dice(logits, labels):
        """Calculate hard foreground Dice separately for each volume."""
        # Equivalent to sigmoid(logits) > 0.5.
        predictions = logits > 0
        targets = labels > 0.5

        dimensions = tuple(range(1, logits.ndim))

        intersection = (predictions & targets).sum(
            dim=dimensions,
            dtype=torch.float32,
        )
        predicted_size = predictions.sum(
            dim=dimensions,
            dtype=torch.float32,
        )
        target_size = targets.sum(
            dim=dimensions,
            dtype=torch.float32,
        )

        denominator = predicted_size + target_size

        # Both empty: 1. Only one empty: 0.
        return torch.where(
            denominator > 0,
            2.0 * intersection / denominator.clamp_min(1.0),
            torch.ones_like(denominator),
        )

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label"]

        logits = self(images)
        loss, dice_loss, bce_loss = self.compute_loss(logits, labels)

        self.log_dict(
            {
                "train_loss": loss,
                "train_dice_loss": dice_loss,
                "train_bce_loss": bce_loss,
            },
            on_step=False,
            on_epoch=True,
            batch_size=images.shape[0],
        )

        # Lightning performs backpropagation and optimizer updates.
        return loss

    def _evaluation_step(self, batch, stage):
        images = batch["image"]
        labels = batch["label"]

        logits = self(images)
        loss, _, _ = self.compute_loss(logits, labels)
        dice = self.compute_dice(logits, labels)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=images.shape[0],
        )
        self.log(
            f"{stage}_dice",
            dice.mean(),
            on_step=False,
            on_epoch=True,
            batch_size=images.shape[0],
            prog_bar=True,
        )

    def validation_step(self, batch, batch_idx):
        self._evaluation_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._evaluation_step(batch, stage="test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            **self.hparams.optimizer_config,
        )


def read_yaml(path):
    with open(path) as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/training.yaml",
    )
    parser.add_argument(
        "--datasets",
        default="configs/datasets.yaml",
    )
    parser.add_argument(
        "--experiment",
        default="configs/experiments/real_100.yaml",
    )
    parser.add_argument(
        "--split",
        default="configs/splits/real_split.json",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help=(
            "Run one training and validation batch "
            "without saving model checkpoints."
        ),
    )
    args = parser.parse_args()

    config = read_yaml(args.config)
    experiment = read_yaml(args.experiment)

    pl.seed_everything(config["seed"], workers=True)

    # Build datasets using the selected experiment.
    datamodule = NoduleDataModule(
        datasets_config=args.datasets,
        experiment_config=args.experiment,
        split_path=args.split,
        **config["data"],
    )

    # Create or validate the fixed split before launching training.
    datamodule.setup("fit")

    print(f"Experiment: {experiment['name']}")
    print(f"Training cases: {len(datamodule.train_dataset)}")
    print(f"Validation cases: {len(datamodule.val_dataset)}")

    model = SegmentationModule(
        model_config=config["model"],
        loss_config=config["loss"],
        optimizer_config=config["optimizer"],
    )

    # Create a separate output directory for each run.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        Path(config["output_dir"])
        / experiment["name"]
        / timestamp
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    # Preserve the exact configuration and patient split.
    config_dir = run_dir / "configs"
    config_dir.mkdir()

    for source, filename in (
        (args.config, "training.yaml"),
        (args.datasets, "datasets.yaml"),
        (args.experiment, "experiment.yaml"),
        (args.split, "real_split.json"),
    ):
        shutil.copy2(source, config_dir / filename)

    checkpoint_settings = config["checkpoint"]

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="epoch={epoch:03d}-dice={val_dice:.4f}",
        auto_insert_metric_name=False,
        monitor=checkpoint_settings["monitor"],
        mode=checkpoint_settings["mode"],
        save_top_k=checkpoint_settings["save_top_k"],
        save_last=checkpoint_settings["save_last"],
    )

    logger = CSVLogger(
        save_dir=str(run_dir),
        name="",
        version="logs",
    )

    trainer = pl.Trainer(
        **config["trainer"],
        default_root_dir=str(run_dir),
        callbacks=[checkpoint_callback],
        logger=logger,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule)

    if not args.fast_dev_run:
        print(f"\nRun directory: {run_dir.resolve()}")
        print(
            "Best checkpoint:",
            checkpoint_callback.best_model_path,
        )
        print(
            "Best validation Dice:",
            checkpoint_callback.best_model_score,
        )


if __name__ == "__main__":
    main()