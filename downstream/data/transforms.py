"""Load aligned CT/masks and construct the requested segmentation target."""

import math

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


class ValidateIntensityRanged(MapTransform):
    """Require finite image intensities in [0, 1] without modifying the tensor."""

    def __init__(self):
        super().__init__(keys=["image"])

    def __call__(self, data):
        case_id = data.get("case_id", data.get("patient_id", "unknown"))
        for key in self.key_iterator(data):
            image = data[key]
            if not torch.isfinite(image).all():
                raise ValueError(f"{case_id}: {key} contains non-finite intensities.")
            observed_min, observed_max = image.amin().item(), image.amax().item()
            if observed_min < 0.0 or observed_max > 1.0:
                raise ValueError(
                    f"{case_id}: {key} intensities must be in [0, 1] when "
                    f"in_min=0 and in_max=1; observed [{observed_min}, {observed_max}]."
                )
        return data


def build_transforms(
    expected_shape=(256, 256, 128),
    in_min=0.0,
    in_max=1.0,
    classes="nodule",
):
    """Validate [0, 1] inputs, otherwise clip/rescale; masks are not resampled."""
    if not (math.isfinite(in_min) and math.isfinite(in_max)):
        raise ValueError("in_min and in_max must be finite.")
    if in_min >= in_max:
        raise ValueError("in_min must be smaller than in_max.")
    keys = ["image", *mask_keys(classes)]
    intensity_transform = (
        ValidateIntensityRanged()
        if in_min == 0.0 and in_max == 1.0
        else ScaleIntensityRanged(
            keys=["image"], a_min=in_min, a_max=in_max,
            b_min=0.0, b_max=1.0, clip=True,
        )
    )
    return Compose([
        LoadImaged(keys=keys, reader="NibabelReader", image_only=True),
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        EnsureTyped(keys=keys, dtype=torch.float32, track_meta=True),
        CheckPairGeometryd(expected_shape=expected_shape, classes=classes),
        intensity_transform,
        ComposeSegmentationLabeld(classes=classes),
    ])
