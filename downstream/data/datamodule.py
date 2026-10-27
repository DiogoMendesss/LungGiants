"""Lightning DataModule for full-volume nodule segmentation."""

from functools import partial

import lightning.pytorch as pl
import torch.multiprocessing as multiprocessing
from monai.data import Dataset, DataLoader
from monai.data.utils import worker_init_fn

from data.experiment_data import build_experiment_data
from data.transforms import build_transforms


def initialize_loader_worker(worker_id, sharing_strategy):
    """Configure tensor IPC in each worker, including workers started by spawn."""
    if sharing_strategy is not None:
        multiprocessing.set_sharing_strategy(sharing_strategy)
    # Preserve MONAI's initialization of randomizable dataset transforms.
    worker_init_fn(worker_id)


class NoduleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets_config="configs/datasets.yaml",
        experiment_config="configs/experiments/real_100.yaml",
        split_path="configs/splits/real_split.json",
        batch_size=1,
        eval_batch_size=1,
        num_workers=4,
        pin_memory=True,
        expected_shape=(256, 256, 128),
        hu_min=-1000,
        hu_max=1000,
        classes="nodule",
        sharing_strategy="file_system",
        prefetch_factor=1,
    ):
        super().__init__()
        if num_workers > 0:
            if (
                sharing_strategy is not None
                and sharing_strategy not in multiprocessing.get_all_sharing_strategies()
            ):
                raise ValueError(f"Unsupported tensor sharing strategy: {sharing_strategy}")
            if prefetch_factor < 1:
                raise ValueError("prefetch_factor must be at least 1.")
        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._case_lists = None

    def setup(self, stage=None):
        """Build datasets when Lightning starts training or evaluation."""
        supported_stages = (None, "fit", "validate", "test")
        if stage not in supported_stages:
            raise ValueError(f"Unsupported stage: {stage}")

        # Index paths once per DataModule instance.
        if self._case_lists is None:
            self._case_lists = build_experiment_data(
                datasets_config=self.hparams.datasets_config,
                experiment_config=self.hparams.experiment_config,
                split_path=self.hparams.split_path,
                classes=self.hparams.classes,
            )

        train_cases, val_cases, test_cases = self._case_lists

        if stage in (None, "fit"):
            if self.train_dataset is None:
                self.train_dataset = Dataset(
                    data=train_cases,
                    transform=self._make_transforms(),
                )

        if stage in (None, "fit", "validate"):
            if self.val_dataset is None:
                self.val_dataset = Dataset(
                    data=val_cases,
                    transform=self._make_transforms(),
                )

        if stage in (None, "test"):
            if self.test_dataset is None:
                self.test_dataset = Dataset(
                    data=test_cases,
                    transform=self._make_transforms(),
                )

    def _make_transforms(self):
        """Create a separate transform pipeline for each dataset."""
        return build_transforms(
            expected_shape=self.hparams.expected_shape,
            hu_min=self.hparams.hu_min,
            hu_max=self.hparams.hu_max,
            classes=self.hparams.classes,
        )

    def _make_loader(self, dataset, batch_size, shuffle):
        if dataset is None:
            raise RuntimeError(
                "Dataset has not been initialized. Call setup() first."
            )

        worker_options = {}
        if self.hparams.num_workers > 0:
            worker_options = {
                "prefetch_factor": self.hparams.prefetch_factor,
                "worker_init_fn": partial(
                    initialize_loader_worker,
                    sharing_strategy=self.hparams.sharing_strategy,
                ),
            }

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            drop_last=False,
            **worker_options,
        )

    def train_dataloader(self):
        return self._make_loader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return self._make_loader(
            dataset=self.val_dataset,
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return self._make_loader(
            dataset=self.test_dataset,
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
        )