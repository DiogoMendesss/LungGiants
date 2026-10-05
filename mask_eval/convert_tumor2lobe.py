import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from tqdm import tqdm

MASKS_DIR = r"./data"
OUTPUT_DIR = r"./converted"
LUNG_LOBE_LABELS = [28, 29, 30, 31, 32]
TUMOR_LABEL = 23


def convert_mask(mask_path, output_dir):
    """Replace all tumor voxels with the dominant lobe touching their shell."""
    image = nib.load(str(mask_path))
    mask = np.asanyarray(image.dataobj).copy()
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")

    tumor = mask == TUMOR_LABEL
    shell = ndi.binary_dilation(
        tumor, structure=np.ones((3, 3, 3), dtype=bool)
    ) & ~tumor
    neighbours = mask[shell]
    counts = np.array([np.count_nonzero(neighbours == label)
                       for label in LUNG_LOBE_LABELS])
    if not counts.sum():
        logging.info("Skipped %s: no lung contact.", mask_path.name)
        return False

    # Maximizing the count also maximizes lobe dominance. Ties use label order.
    dominant_lobe = LUNG_LOBE_LABELS[counts.argmax()]
    mask[tumor] = dominant_lobe
    converted = image.__class__(mask, image.affine, image.header.copy())
    nib.save(converted, str(output_dir / mask_path.name))
    return True


def main():
    parser = argparse.ArgumentParser(description="Assign tumors with lung contact to their dominant lobe.")
    parser.add_argument("--masks-dir", type=Path, default=Path(MASKS_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR))
    args = parser.parse_args()
    if not args.masks_dir.is_dir():
        parser.error(f"Masks directory does not exist: {args.masks_dir}")
    if args.output_dir.resolve() == args.masks_dir.resolve():
        parser.error("Output directory must differ from masks directory.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    converted = skipped = failed = 0
    mask_paths = sorted(path for path in args.masks_dir.iterdir()
                        if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz")))
    for mask_path in tqdm(mask_paths, desc="Converting masks", unit="mask"):
        try:
            if convert_mask(mask_path, args.output_dir):
                converted += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            logging.error("Failed %s: %s", mask_path.name, exc)
    logging.info("Converted: %d | Skipped: %d | Failed: %d", converted, skipped, failed)


if __name__ == "__main__":
    main()
