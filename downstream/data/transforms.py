"""Transforms for preprocessed full-volume CT segmentation."""

import torch

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    ScaleIntensityRanged,
)


class BinarizeNodulesd(MapTransform):
    """Convert positive nodule instance IDs to foreground."""

    def __call__(self, data):
        result = dict(data)

        for key in self.key_iterator(result):
            result[key] = (result[key] > 0).float()

        return result


class CheckPairGeometryd(MapTransform):
    """Check the expected shape and matching CT/mask voxel grids."""

    def __init__(self, expected_shape):
        super().__init__(keys=["image", "label"])
        self.expected_shape = (1, *expected_shape)

    def __call__(self, data):
        image = data["image"]
        label = data["label"]
        case_id = data.get("case_id", "unknown")

        if tuple(image.shape) != self.expected_shape:
            raise ValueError(
                f"{case_id}: image shape {tuple(image.shape)} "
                f"does not match {self.expected_shape}."
            )

        if tuple(label.shape) != tuple(image.shape):
            raise ValueError(
                f"{case_id}: image and mask shapes do not match."
            )

        if not torch.allclose(
            image.affine, label.affine, atol=1e-3, rtol=0
        ):
            raise ValueError(
                f"{case_id}: image and mask affines do not match."
            )

        return data


def build_transforms(
    expected_shape=(256, 256, 128),
    hu_min=-1000,
    hu_max=1000,
):
    """Load aligned volumes and prepare a binary segmentation target."""
    if hu_min >= hu_max:
        raise ValueError("hu_min must be smaller than hu_max.")

    return Compose(
        [
            LoadImaged(
                keys=["image", "label"],
                reader="NibabelReader",
                image_only=True,
            ),
            EnsureChannelFirstd(
                keys=["image", "label"],
                channel_dim="no_channel",
            ),
            EnsureTyped(
                keys=["image", "label"],
                dtype=torch.float32,
                track_meta=True,
            ),
            CheckPairGeometryd(expected_shape=expected_shape),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=hu_min,
                a_max=hu_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            BinarizeNodulesd(keys=["label"]),
        ]
    )