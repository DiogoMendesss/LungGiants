"""Intensity validation and optional normalization regressions."""

import nibabel as nib
import numpy as np
import pytest
import torch
from monai.data import MetaTensor

from data.datamodule import NoduleDataModule
from data.transforms import ValidateIntensityRanged, build_transforms


@pytest.mark.parametrize("values", [[0., .25, .75, 1.], [.2, .3, .4, .5]])
def test_validation_preserves_tensor_and_metadata(values):
    image = MetaTensor(torch.tensor(values), affine=torch.diag(torch.tensor([2., 3., 4., 1.])))
    original = image.clone()
    data = {"image": image, "case_id": "preprocessed"}
    assert ValidateIntensityRanged()(data) is data
    assert data["image"] is image
    assert torch.equal(image, original)
    assert torch.equal(image.affine, original.affine)


@pytest.mark.parametrize("bad_value", [-1e-6, 1.000001, float("nan"), float("inf"), -float("inf")])
def test_validation_rejects_invalid_voxels(bad_value):
    with pytest.raises(ValueError, match="bad-case"):
        ValidateIntensityRanged()({"image": torch.tensor([0., bad_value, 1.]), "case_id": "bad-case"})


@pytest.mark.parametrize("bounds", [(1, 1), (2, 1), (float("nan"), 1), (0, float("inf"))])
def test_invalid_configuration(bounds):
    with pytest.raises(ValueError, match="in_min"):
        build_transforms(in_min=bounds[0], in_max=bounds[1])


@pytest.mark.parametrize("classes", ["nodule", "lung+nodule"])
@pytest.mark.parametrize("bounds", [None, (0, 1), (-1000, 1000), (10, 20)])
def test_nifti_pipeline(tmp_path, classes, bounds):
    expected = np.array([0., 0., .25, .5, .75, 1., 1., .125], dtype=np.float32).reshape(2, 2, 2)
    source = expected.copy()
    kwargs = {} if bounds is None else dict(in_min=bounds[0], in_max=bounds[1])
    if bounds is not None and bounds != (0, 1):
        source = source * (bounds[1] - bounds[0]) + bounds[0]
        source.flat[0] = bounds[0] - 100
        source.flat[-2] = bounds[1] + 100
    affine = np.diag([2., 3., 4., 1.])
    record = {"case_id": "pipeline-case"}
    arrays = {"image": source, "nod_mask": np.zeros_like(source)}
    arrays["nod_mask"].flat[0] = 7
    if classes == "lung+nodule":
        arrays["ts_mask"] = np.full_like(source, 28)
    for key, array in arrays.items():
        path = tmp_path / f"{key}.nii.gz"
        nib.save(nib.Nifti1Image(array, affine), path)
        record[key] = str(path)
    # Exercise the DataModule forwarding as well as the complete loading pipeline.
    dm = NoduleDataModule(expected_shape=(2, 2, 2), classes=classes, num_workers=0, **kwargs)
    output = dm._make_transforms()(record)
    assert torch.equal(output["image"], torch.from_numpy(expected).unsqueeze(0))
    assert np.array_equal(output["image"].affine.numpy(), affine)
    assert output["label"].flatten()[0] == (1 if classes == "nodule" else 2)


def test_default_pipeline_rejects_out_of_range_nifti(tmp_path):
    record = {"case_id": "invalid-nifti"}
    for key, value in (("image", -500.), ("nod_mask", 0.)):
        path = tmp_path / f"{key}.nii.gz"
        nib.save(nib.Nifti1Image(np.full((2, 2, 2), value, dtype=np.float32), np.eye(4)), path)
        record[key] = str(path)
    # MONAI wraps transform errors; the original error identifies the invalid case.
    with pytest.raises(RuntimeError) as error:
        build_transforms(expected_shape=(2, 2, 2))(record)
    assert isinstance(error.value.__cause__, ValueError)
    assert "invalid-nifti" in str(error.value.__cause__)
