"""Train full-volume SegResNet for nodules or lung + nodule segmentation."""

import argparse
from datetime import datetime
from pathlib import Path
import shutil

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from monai.losses import DiceLoss
from torch import nn

from data.datamodule import NoduleDataModule
from models.segresnet import SegResNet
from tasks import CLASS_CHOICES, class_ids, output_channels, predict_labels, validate_classes


class SegmentationModule(pl.LightningModule):
    """Train the selected task and monitor nodule performance independently."""

    def __init__(self, model_config, loss_config, optimizer_config, classes="nodule"):
        super().__init__()
        self.classes = validate_classes(classes)
        # Resolve before saving so checkpoint metadata describes the actual network.
        model_config = {**model_config, "out_channels": output_channels(classes)}
        self.save_hyperparameters()
        self.network = SegResNet(**model_config)

        joint = classes == "lung+nodule"
        self.dice_loss = DiceLoss(
            sigmoid=not joint, softmax=joint,
            include_background=not joint, to_onehot_y=joint,
            batch=False, reduction="mean",
        )
        self.entropy_loss = nn.CrossEntropyLoss() if joint else nn.BCEWithLogitsLoss()
        self.nodule_dice_loss = DiceLoss(sigmoid=True, reduction="mean")
        self.nodule_bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = loss_config["dice_weight"]
        self.entropy_name = "ce" if joint else "bce"
        self.entropy_weight = loss_config[f"{self.entropy_name}_weight"]
        if (
            self.dice_weight < 0 or self.entropy_weight < 0
            or self.dice_weight + self.entropy_weight == 0
        ):
            raise ValueError("Loss weights must be nonnegative and not both zero.")

    def forward(self, images):
        """Return raw logits: one channel for binary, three for multiclass."""
        return self.network(images)

    def compute_loss(self, logits, labels):
        """Global training objective: foreground Dice plus BCE or multiclass CE."""
        logits = logits.float()
        if self.classes == "nodule":
            labels = labels.float()
            entropy = self.entropy_loss(logits, labels)
        else:
            labels = labels.long()
            entropy = self.entropy_loss(logits, labels[:, 0])
        dice = self.dice_loss(logits, labels)
        total = self.dice_weight * dice + self.entropy_weight * entropy
        return total, dice, entropy

    def compute_nodule_loss(self, logits, labels):
        """Nodule-vs-rest monitoring loss, not an extra training objective.

        For joint mode, sigmoid(z_nodule - logsumexp(z_background, z_lung))
        equals the softmax nodule probability. BCEWithLogits evaluates this
        stably, penalizing false positives and false negatives on all voxels.
        This diagnostic is not an additive component of multiclass CE.
        """
        if self.classes == "nodule":
            return self.compute_loss(logits, labels)[0]
        logits = logits.float()
        nodule_log_odds = logits[:, 2:3] - torch.logsumexp(
            logits[:, :2], dim=1, keepdim=True,
        )
        targets = (labels == 2).float()
        return (
            self.dice_weight * self.nodule_dice_loss(nodule_log_odds, targets)
            + self.entropy_weight * self.nodule_bce_loss(nodule_log_odds, targets)
        )

    def compute_dice(self, logits, labels):
        """Return per-case hard Dice for each foreground class separately."""
        predictions = predict_labels(logits, self.classes)
        dimensions = tuple(range(1, labels.ndim))
        scores = {}
        for name, class_id in class_ids(self.classes).items():
            predicted = predictions == class_id
            target = labels == class_id
            intersection = (predicted & target).sum(dim=dimensions, dtype=torch.float32)
            denominator = (
                predicted.sum(dim=dimensions, dtype=torch.float32)
                + target.sum(dim=dimensions, dtype=torch.float32)
            )
            # Keep the original convention: both empty=1, only one empty=0.
            scores[name] = torch.where(
                denominator > 0, 2 * intersection / denominator.clamp_min(1),
                torch.ones_like(denominator),
            )
        return scores

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        loss, dice, entropy = self.compute_loss(self(images), labels)
        self.log_dict(
            {
                "train_loss": loss,
                "train_dice_loss": dice,
                f"train_{self.entropy_name}_loss": entropy,
            },
            on_step=True, on_epoch=True, batch_size=images.shape[0],
        )
        return loss

    def _evaluation_step(self, batch, stage):
        images, labels = batch["image"], batch["label"]
        logits = self(images)
        loss, _, _ = self.compute_loss(logits, labels)
        nodule_loss = loss if self.classes == "nodule" else self.compute_nodule_loss(logits, labels)
        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_loss_nodule": nodule_loss,
            **{
                f"{stage}_dice_{name}": values.mean()
                for name, values in self.compute_dice(logits, labels).items()
            },
        }
        self.log_dict(
            metrics, on_step=False, on_epoch=True,
            batch_size=images.shape[0], prog_bar=True,
        )

    def validation_step(self, batch, batch_idx):
        self._evaluation_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._evaluation_step(batch, stage="test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_config)


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
    parser.add_argument(
        "--classes", choices=CLASS_CHOICES, default=None,
        help="Segmentation task; overrides config.classes (default: nodule).",
    )
    args = parser.parse_args()

    config = read_yaml(args.config)
    experiment = read_yaml(args.experiment)
    classes = validate_classes(args.classes or config.get("classes", "nodule"))
    config["classes"] = classes
    config["model"]["out_channels"] = output_channels(classes)

    if not args.fast_dev_run:
        from reporting import check_report_dependencies, generate_training_report

        check_report_dependencies()

    pl.seed_everything(config["seed"], workers=True)

    if args.fast_dev_run:
        config["data"]["num_workers"] = 0
        config["data"]["pin_memory"] = False

    # Build datasets using the selected experiment.
    datamodule = NoduleDataModule(
        datasets_config=args.datasets,
        experiment_config=args.experiment,
        split_path=args.split,
        classes=classes,
        **config["data"],
    )

    # Create or validate the fixed split before launching training.
    datamodule.setup("fit")

    print(f"Experiment: {experiment['name']}")
    print(f"Classes: {classes}")
    print(f"Training cases: {len(datamodule.train_dataset)}")
    print(f"Validation cases: {len(datamodule.val_dataset)}")

    model = SegmentationModule(
        model_config=config["model"],
        loss_config=config["loss"],
        optimizer_config=config["optimizer"],
        classes=classes,
    )

    # Separate task outputs, then create a directory for this experiment/run.
    output_dir = Path(config["output_dir"])
    output_dir = output_dir.with_name(f"{output_dir.name}_{classes}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        output_dir
        / experiment["name"]
        / timestamp
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    # Preserve the exact configuration and patient split.
    config_dir = run_dir / "configs"
    config_dir.mkdir()

    for source, filename in (
        (args.config, "training_source.yaml"),
        (args.datasets, "datasets.yaml"),
        (args.experiment, "experiment.yaml"),
        (args.split, "real_split.json"),
    ):
        shutil.copy2(source, config_dir / filename)

    # Save effective CLI overrides as well as the original YAML.
    with (config_dir / "training.yaml").open("w") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    checkpoint_settings = config["checkpoint"]

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="epoch={epoch:03d}-nodule_dice={val_dice_nodule:.4f}",
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

    wandb_logger = WandbLogger(
        project="LungGiants",
        name=f"{experiment['name']}_{classes}_{timestamp}",
        save_dir=str(run_dir),
        log_model=False,
    )
    wandb_logger.log_hyperparams(config)

    early_stopping = EarlyStopping(**config["early_stop"])
    callbacks = [checkpoint_callback, early_stopping]
    if config["trainer"].get("enable_progress_bar", True):
        callbacks.append(TQDMProgressBar(refresh_rate=10))

    trainer = pl.Trainer(
        **config["trainer"],
        default_root_dir=str(run_dir),
        callbacks=callbacks,
        logger=[logger, wandb_logger],
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
            "Best checkpoint monitor value:",
            checkpoint_callback.best_model_score,
        )

        generate_training_report(
            model, datamodule, trainer, logger, checkpoint_callback,
            run_dir, experiment["name"],
        )


if __name__ == "__main__":
    main()