import argparse
import logging
from pathlib import Path

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from tqdm import tqdm

MASKS_DIR = r"Z:\nas-ctm01\datasets\public\LungNodule-nifti\MAISI\results_synth_maskct\masks_132"
LUNG_LOBE_LABELS = [28, 29, 30, 31, 32]
ANATOMY_LABELS = {
    28: "LUL",
    29: "LLL",
    30: "RUL",
    31: "RML",
    32: "RLL",
}
TUMOR_LABEL = 23


def inspect_mask(mask_path, logger):
    """Log lobe percentages in the tumor's one-voxel, 26-connected shell.

    Percentages use all shell voxels, including non-lobe labels, as the
    denominator. The input mask is only read and is never modified.
    """
    mask = np.asanyarray(nib.load(str(mask_path)).dataobj)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")

    tumor_mask = mask == TUMOR_LABEL
    tumor_count = np.count_nonzero(tumor_mask)
    shell = ndi.binary_dilation(
        tumor_mask, structure=np.ones((3, 3, 3), dtype=bool)
    ) & ~tumor_mask
    neighbours = mask[shell]
    counts = np.array([np.count_nonzero(neighbours == label)
                       for label in LUNG_LOBE_LABELS])
    percentages = counts / neighbours.size * 100 if neighbours.size else np.zeros(5)
    above_threshold = bool(np.any(percentages > 90))

    logger.info("Mask: %s", mask_path.name)
    logger.info("Tumor voxels: %d | Neighbour voxels: %d", tumor_count, neighbours.size)
    if not tumor_count:
        logger.warning("No tumor label %d present; percentages set to zero.", TUMOR_LABEL)
    elif not neighbours.size:
        logger.warning("No neighbour voxels; percentages set to zero.")
    logger.info("Lung lobe       | Neighbour count | Percentage")
    logger.info("----------------+-----------------+-----------")
    for label, count, percentage in zip(LUNG_LOBE_LABELS, counts, percentages):
        logger.info("%15s | %15d | %9.2f%%", ANATOMY_LABELS[label], count, percentage)
    logger.info("Any lung lobe above 90%%: %s\n", "yes" if above_threshold else "no")
    return above_threshold


def main():
    parser = argparse.ArgumentParser(description="Inspect tumor neighbours without converting masks.")
    parser.add_argument("--masks-dir", type=Path, default=Path(MASKS_DIR))
    parser.add_argument("--log-file", type=Path,
                        default=Path(__file__).with_name("tumor_lunglobe_inspection.txt"))
    args = parser.parse_args()
    if not args.masks_dir.is_dir():
        parser.error(f"Masks directory does not exist: {args.masks_dir}")

    mask_paths = sorted(path for path in args.masks_dir.iterdir()
                        if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz")))
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tumor_lunglobe_inspection")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(args.log_file, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    inspected = below_threshold = failed = 0
    try:
        logger.info("Masks directory: %s", args.masks_dir.resolve())
        logger.info("Percentages use all neighbours; threshold is strictly above 90%%.\n")
        for mask_path in tqdm(mask_paths, desc="Inspecting masks", unit="mask"):
            try:
                above_threshold = inspect_mask(mask_path, logger)
            except Exception:
                failed += 1
                logger.exception("Failed to inspect %s\n", mask_path.name)
                continue
            inspected += 1
            below_threshold += not above_threshold

        logger.info("SUMMARY")
        logger.info("Masks found: %d | Inspected: %d | Failed: %d", len(mask_paths), inspected, failed)
        logger.info("Masks without any lung lobe label percentage above 90%%: %d", below_threshold)
        print(f"Inspected {inspected} masks; {below_threshold} without a lobe above 90%; {failed} failed.")
        print(f"Log saved to: {args.log_file.resolve()}")
    finally:
        logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    main()
