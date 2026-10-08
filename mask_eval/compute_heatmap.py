import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

MASKS_DIR =  r"./converted"
LUNG_LOBE_LABELS = [28, 29, 30, 31, 32]


def main():
    parser = argparse.ArgumentParser(description="Compute the mean binary lung map across masks.")
    parser.add_argument("--masks-dir", type=Path, default=Path(MASKS_DIR))
    args = parser.parse_args()
    if not args.masks_dir.is_dir():
        parser.error(f"Masks directory does not exist: {args.masks_dir}")

    mask_paths = sorted(
        path for path in MASKS_DIR.rglob("*")
        if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz"))
    )

    if not mask_paths:
        parser.error(f"No NIfTI masks found in: {args.masks_dir}")
    else:
        print(f"Found {len(mask_paths)} mask files in {MASKS_DIR}")

    heatmap = None
    lps_orientation = nib.orientations.axcodes2ornt(("L", "P", "S"))
    for mask_path in tqdm(mask_paths, desc="Computing heatmap", unit="mask"):
        image = nib.load(str(mask_path))
        mask = np.asanyarray(image.dataobj)
        if mask.ndim != 3:
            raise ValueError(f"{mask_path.name}: expected a 3D mask, got {mask.shape}")
        transform = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(image.affine), lps_orientation
        )
        mask = nib.orientations.apply_orientation(mask, transform)
        if heatmap is None:
            heatmap = np.zeros(mask.shape, dtype=np.float32)
        elif mask.shape != heatmap.shape:
            raise ValueError(f"{mask_path.name}: shape {mask.shape} differs from {heatmap.shape}")
        heatmap += np.isin(mask, LUNG_LOBE_LABELS)

    output_dir = Path("./") #output_dir = args.masks_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap /= len(mask_paths)
    output_path = output_dir / "lung_heatmap.npy"
    np.save(output_path, heatmap)
    print(f"Heatmap saved to: {output_path.resolve()}")

    slice_index = heatmap.shape[2] // 2
    fig, ax = plt.subplots()
    plot = ax.imshow(heatmap[:, :, slice_index].T, cmap="hot", vmin=0, vmax=1)
    ax.set_title(f"Lung heatmap — axial slice {slice_index}")
    ax.axis("off")
    fig.colorbar(plot, ax=ax, label="Lung frequency")
    png_path = output_dir / "lung_heatmap.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG saved to: {png_path.resolve()}")


if __name__ == "__main__":
    main()
