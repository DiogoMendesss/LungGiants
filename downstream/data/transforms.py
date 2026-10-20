"""Load aligned CT/masks and construct the requested segmentation target."""

import torch
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged,
    MapTransform, ScaleIntensityRanged,
)

from tasks import LUNG_LOBE_LABELS, mask_keys, validate_classes


class ComposeSegmentationLabeld(MapTransform):
    """Build binary nodules or 0=background, 1=lung, 2=nodule labels."""

    def __init__(self, classes="nodule"):
        super().__init__(keys=mask_keys(classes))
        self.classes = validate_classes(classes)

    def __call__(self, data):
        result = dict(data)
        nodules = result["nod_mask"] > 0
        if self.classes == "nodule":
            label = nodules.float()
        else:
            label = torch.zeros_like(result["nod_mask"], dtype=torch.long)
            lung = torch.zeros_like(nodules, dtype=torch.bool)
            for lobe_id in LUNG_LOBE_LABELS:
                lung |= result["ts_mask"] == lobe_id
            label[lung] = 1
            # Preserve every annotated nodule, including those outside TS lungs.
            label[nodules] = 2
        result["label"] = label
        for key in ("nod_mask", "ts_mask"):
            result.pop(key, None)
        return result


class CheckPairGeometryd(MapTransform):
    """Check expected CT shape and the voxel grid of every required mask."""

    def __init__(self, expected_shape, classes="nodule"):
        super().__init__(keys=mask_keys(classes))
        self.expected_shape = (1, *expected_shape)

    def __call__(self, data):
        image = data["image"]
        case_id = data.get("case_id", "unknown")
        if tuple(image.shape) != self.expected_shape:
            raise ValueError(
                f"{case_id}: image shape {tuple(image.shape)} "
                f"does not match {self.expected_shape}."
            )
        for key in self.key_iterator(data):
            mask = data[key]
            if tuple(mask.shape) != tuple(image.shape):
                raise ValueError(f"{case_id}: image and {key} shapes do not match.")
            if not torch.allclose(image.affine, mask.affine, atol=1e-3, rtol=0):
                raise ValueError(f"{case_id}: image and {key} affines do not match.")
        return data


def build_transforms(
    expected_shape=(256, 256, 128),
    hu_min=-1000,
    hu_max=1000,
    classes="nodule",
):
    """Construct labels on load; already aligned masks are not resampled."""
    if hu_min >= hu_max:
        raise ValueError("hu_min must be smaller than hu_max.")
    keys = ["image", *mask_keys(classes)]
    return Compose([
        LoadImaged(keys=keys, reader="NibabelReader", image_only=True),
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        EnsureTyped(keys=keys, dtype=torch.float32, track_meta=True),
        CheckPairGeometryd(expected_shape=expected_shape, classes=classes),
        ScaleIntensityRanged(
            keys=["image"], a_min=hu_min, a_max=hu_max,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        ComposeSegmentationLabeld(classes=classes),
    ])
