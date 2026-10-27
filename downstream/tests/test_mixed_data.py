"""Patient-level mix selection, mask resolution, and real-only evaluation."""

import json
from pathlib import Path

import pytest
import yaml

from data.dataset_index import build_real_index
from data.experiment_data import build_experiment_data, save_case_manifest, select_synthetic_ids


def make_folders(root, prefix, ids):
    folders = {}
    for key, leaf in (("images", "ct"), ("nod_masks", "nodulemask"), ("ts_masks", "ts_out")):
        folder = root / prefix / leaf
        folder.mkdir(parents=True)
        for case_id in ids:
            (folder / f"{case_id}.nii.gz").touch()
        folders[key] = str(folder.relative_to(root))
    return folders


@pytest.fixture
def cohort(tmp_path):
    real = {"train": {}, "test": {}}
    all_ids = []
    for name in ("LIDC", "LNDb"):
        ids = [f"{name}-{i:04d}" for i in range(10)]
        test_ids = [f"{name}-9999"]
        real["train"][name] = make_folders(tmp_path, f"real/train/{name}", ids)
        real["test"][name] = make_folders(tmp_path, f"real/test/{name}", test_ids)
        all_ids.extend(ids + test_ids)
    # Include generated validation/test IDs: neither may enter training.
    conditioned = make_folders(tmp_path, "conditioned", all_ids)
    independent = make_folders(tmp_path, "independent", [f"gen-{i:04d}" for i in range(30)])
    config = {
        "root": str(tmp_path),
        "split": {"seed": 42, "validation_fraction": .2, "stratify_by_dataset": True},
        "real": real,
        "synthetic": {
            "conditioned": {**conditioned, "conditioned_on_real": True},
            "other_generator": {**conditioned, "conditioned_on_real": True},
            "independent": {**independent, "conditioned_on_real": False},
        },
    }
    config_path = tmp_path / "datasets.yaml"
    config_path.write_text(yaml.safe_dump(config))
    split_path = tmp_path / "split.json"
    index = build_real_index(config_path, split_path)

    def build(real_percent=50, synthetic_percent=50, name="conditioned", classes="nodule"):
        config_path.write_text(yaml.safe_dump(config))
        experiment = tmp_path / "experiment.yaml"
        experiment.write_text(yaml.safe_dump({"data": {
            "real_percent": real_percent, "synthetic_percent": synthetic_percent,
            "synthetic_dataset": name,
        }}))
        return build_experiment_data(config_path, experiment, split_path, classes)

    return tmp_path, config, index, build


def ids(cases):
    return {case["patient_id"] for case in cases}


@pytest.mark.parametrize("percent", [25, 50, 75])
def test_complementary_mix_and_fixed_evaluation(cohort, percent):
    _, _, index, build = cohort
    baseline = build(100, 0, None)
    train, val, test = build(percent, 100 - percent)
    real = [case for case in train if case["source"] == "real"]
    synthetic = [case for case in train if case["source"] == "synthetic"]
    assert len(real) == 16 * percent / 100
    assert len(train) == 16
    assert not ids(real) & ids(synthetic)
    assert ids(train) == ids(index["train"])
    assert not ids(train) & (ids(val) | ids(test))
    assert val == baseline[1] and test == baseline[2]
    assert all(case["source"] == "real" for case in val + test)
    assert train == build(percent, 100 - percent)[0]
    other = build(percent, 100 - percent, "other_generator")[0]
    assert ids([case for case in other if case["source"] == "synthetic"]) == ids(synthetic)
    assert {dataset: sum(case["source_dataset"] == dataset for case in real)
            for dataset in ("LIDC", "LNDb")} == {"LIDC": len(real) // 2, "LNDb": len(real) // 2}


def test_odd_complement_does_not_duplicate_rounded_case():
    cases = [{"patient_id": str(i), "dataset": "LIDC"} for i in range(5)]
    index = {"train": cases, "val": [], "test": []}
    selected = select_synthetic_ids(index, cases[:3], {"conditioned_on_real": True},
                                    50, 50, {str(i): "path" for i in range(5)}, 42)
    assert selected == ["3", "4"]


@pytest.mark.parametrize("real_percent,synthetic_percent,expected,overlap", [
    (100, 50, 24, 8), (50, 100, 24, 8), (100, 100, 32, 16), (0, 100, 16, 0),
    (0, 25, 4, 0),
])
def test_full_pool_exception_still_excludes_held_out(cohort, real_percent, synthetic_percent,
                                                   expected, overlap):
    _, _, index, build = cohort
    train, val, test = build(real_percent, synthetic_percent)
    real = [case for case in train if case["source"] == "real"]
    synthetic = [case for case in train if case["source"] == "synthetic"]
    assert len(train) == expected
    assert len(ids(real) & ids(synthetic)) == overlap
    assert ids(train) <= ids(index["train"])
    assert not ids(train) & (ids(val) | ids(test))
    assert len({case["sample_id"] for case in train}) == len(train)


def test_independent_uses_real_baseline_and_saves_provenance(cohort):
    root, _, _, build = cohort
    result = build(name="independent")
    synthetic = [case for case in result[0] if case["source"] == "synthetic"]
    assert len(synthetic) == 8  # Half of 16 real training cases, not half of 30 generated cases.
    assert all(case["source_patient_id"] == "" for case in synthetic)
    assert result == build(name="independent")
    manifest = root / "selected.json"
    save_case_manifest(manifest, result)
    saved = json.loads(manifest.read_text())
    assert saved == dict(zip(("train", "val", "test"), result))


def test_independent_rejects_insufficient_pool(cohort):
    root, config, _, build = cohort
    folder = root / config["synthetic"]["independent"]["images"]
    for path in sorted(folder.iterdir())[5:]:
        path.unlink()
    with pytest.raises(ValueError, match="only 5 available"):
        build(name="independent")


@pytest.mark.parametrize("key,description", [("images", "image"), ("nod_masks", "nod_mask"),
                                             ("ts_masks", "ts_mask")])
def test_missing_selected_files_fail_without_resampling(cohort, key, description):
    root, config, _, build = cohort
    train, _, _ = build(classes="lung+nodule")
    selected = next(case for case in train if case["source"] == "synthetic")
    (root / config["synthetic"]["conditioned"][key] / f"{selected['case_id']}.nii.gz").unlink()
    with pytest.raises(ValueError, match=f"missing {description}"):
        build(classes="lung+nodule")


def test_multiple_mask_folders_and_optional_ts(cohort):
    root, config, _, build = cohort
    settings = config["synthetic"]["conditioned"]
    for key in ("nod_masks", "ts_masks"):
        settings[key] = [config["real"]["train"][dataset][key] for dataset in ("LIDC", "LNDb")]
    train, _, _ = build(classes="lung+nodule")
    synthetic = [case for case in train if case["source"] == "synthetic"]
    assert all("real" in Path(case["nod_mask"]).parts for case in synthetic)
    settings["ts_masks"] = "nonexistent"
    build(classes="nodule")
    with pytest.raises(FileNotFoundError):
        build(classes="lung+nodule")
    settings["nod_masks"] *= 2
    with pytest.raises(ValueError, match="Duplicate mask IDs"):
        build()


def test_unknown_conditioning_id_rejected(cohort):
    root, config, _, build = cohort
    (root / config["synthetic"]["conditioned"]["images"] / "unknown.nii.gz").touch()
    with pytest.raises(ValueError, match="Unknown synthetic conditioning"):
        build()


@pytest.mark.parametrize("value", [-1, 101, float("nan"), float("inf"), "50", True])
@pytest.mark.parametrize("field", ["real_percent", "synthetic_percent"])
def test_invalid_percentages(cohort, value, field):
    with pytest.raises(ValueError, match=field):
        cohort[3](**{field: value})


def test_invalid_experiments(cohort):
    build = cohort[3]
    for real, synth, name, message in (
        (0, 0, None, "no training cases"),
        (100, 0, "conditioned", "null"),
        (50, 50, None, "configured synthetic_dataset"),
        (50, 50, "typo", "configured synthetic_dataset"),
        (25, 50, "conditioned", "must sum to 100"),
        (75, 75, "conditioned", "must sum to 100"),
    ):
        with pytest.raises(ValueError, match=message):
            build(real, synth, name)


@pytest.mark.parametrize("classes", ["nodule", "lung+nodule"])
@pytest.mark.parametrize("name", ["conditioned", "independent"])
def test_mixed_nifti_batch_loads(cohort, classes, name):
    np = pytest.importorskip("numpy")
    nib = pytest.importorskip("nibabel")
    pytest.importorskip("monai")
    from monai.data import Dataset, DataLoader
    from data.transforms import build_transforms

    _, _, _, build = cohort
    train, _, _ = build(name=name, classes=classes)
    records = [next(case for case in train if case["source"] == source)
               for source in ("real", "synthetic")]
    for case in records:
        for key in ("image", "nod_mask", "ts_mask"):
            if key not in case:
                continue
            array = np.zeros((4, 4, 4), dtype=np.float32)
            if key == "image":
                array[:] = -500
            elif key == "nod_mask":
                array[1, 1, 1] = 1  # Binary masks also work for generated cases.
            else:
                array[:3, :3, :3] = 28
            nib.save(nib.Nifti1Image(array, np.eye(4)), case[key])
    dataset = Dataset(records, build_transforms(expected_shape=(4, 4, 4), classes=classes))
    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    assert batch["source"] == ["real", "synthetic"]
    assert tuple(batch["image"].shape) == (2, 1, 4, 4, 4)
    assert set(batch["label"].unique().tolist()) == ({0, 1} if classes == "nodule" else {0, 1, 2})
