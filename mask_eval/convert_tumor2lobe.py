import argparse
import logging
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from tqdm import tqdm


LUNG_LOBE_LABELS = [28, 29, 30, 31, 32]
TUMOR_LABEL = 23


def convert_mask(mask_path, output_dir, overwrite=False):
    """Convert tumors touching a lobe; save skipped masks unchanged."""
    output_path = output_dir / mask_path.name
    if output_path.exists() and not overwrite:
        logging.info("Skipped %s: output already exists (overwrite=False).", mask_path.name)
        return False

    image = nib.load(str(mask_path))
    mask = np.asanyarray(image.dataobj).copy()
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")

    tumor = mask == TUMOR_LABEL
    if not tumor.any():
        shutil.copyfile(mask_path, output_path)
        logging.info("Skipped %s: no tumor label (%d); saved unchanged.",
                     mask_path.name, TUMOR_LABEL)
        return False

    shell = ndi.binary_dilation(
        tumor, structure=np.ones((3, 3, 3), dtype=bool)
    ) & ~tumor
    neighbours = mask[shell]
    counts = np.array([np.count_nonzero(neighbours == label)
                       for label in LUNG_LOBE_LABELS])
    if not counts.sum():
        shutil.copyfile(mask_path, output_path)
        logging.info("Skipped %s: no lung contact; saved unchanged.", mask_path.name)
        return False

    # Maximizing the count also maximizes lobe dominance. Ties use label order.
    dominant_lobe = LUNG_LOBE_LABELS[counts.argmax()]
    mask[tumor] = dominant_lobe
    converted = image.__class__(mask, image.affine, image.header.copy())
    nib.save(converted, str(output_path))
    return True


def main():
    parser = argparse.ArgumentParser(description="Assign tumors with lung contact to their dominant lobe.")
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files (default: skip them).")
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
            if convert_mask(mask_path, args.output_dir, overwrite=args.overwrite):
                converted += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            logging.error("Failed %s: %s", mask_path.name, exc)
    logging.info("Converted: %d | Skipped: %d | Failed: %d", converted, skipped, failed)


if __name__ == "__main__":
    main()
