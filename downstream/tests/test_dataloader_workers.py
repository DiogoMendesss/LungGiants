"""Exercise tensor transfers and worker initialization with spawn and no workers."""

import pytest
import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data import Dataset

from data.datamodule import NoduleDataModule, initialize_loader_worker


class WorkerProbeDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {
            "image": torch.full((1, 2, 2, 2), float(index)),
            "strategy": multiprocessing.get_sharing_strategy(),
        }


@pytest.mark.parametrize("num_workers", [0, 1])
@pytest.mark.parametrize("stage", ["train", "val", "test"])
def test_loader_tensor_transfer(stage, num_workers):
    parent_strategy = multiprocessing.get_sharing_strategy()
    dm = NoduleDataModule(num_workers=num_workers, pin_memory=False)
    setattr(dm, f"{stage}_dataset", WorkerProbeDataset())
    loader = getattr(dm, f"{stage}_dataloader")()
    if num_workers:
        # A parent-only strategy change would not reach these workers.
        loader.multiprocessing_context = "spawn"
        assert loader.prefetch_factor == 1
    else:
        assert loader.prefetch_factor is None
        assert not loader.persistent_workers
    for _ in range(2):
        batches = list(loader)
        assert sorted(int(batch["image"][0, 0, 0, 0, 0]) for batch in batches) == [0, 1, 2]
        expected_strategy = "file_system" if num_workers else parent_strategy
        assert all(batch["strategy"] == [expected_strategy] for batch in batches)
    assert multiprocessing.get_sharing_strategy() == parent_strategy


def test_worker_initialization_preserves_monai_seeding(monkeypatch):
    calls = []
    monkeypatch.setattr(multiprocessing, "set_sharing_strategy", lambda value: calls.append(value))
    monkeypatch.setattr("data.datamodule.worker_init_fn", lambda worker_id: calls.append(worker_id))
    initialize_loader_worker(2, "file_system")
    assert calls == ["file_system", 2]
    calls.clear()
    initialize_loader_worker(3, None)
    assert calls == [3]


def test_invalid_worker_settings_fail_before_loading():
    with pytest.raises(ValueError, match="sharing strategy"):
        NoduleDataModule(num_workers=1, sharing_strategy="invalid")
    with pytest.raises(ValueError, match="prefetch_factor"):
        NoduleDataModule(num_workers=1, prefetch_factor=0)
