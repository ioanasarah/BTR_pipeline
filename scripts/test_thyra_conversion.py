"""Test Thyra conversion on 2 datasets from Slide 11.

The Rapiflex reader expects a folder containing ONE set of:
  - .dat (spectral data)
  - _poslog.txt (coordinates)
  - _info.txt (metadata)
  - .mis (alignment, optional)

But the slide folder has all samples mixed together. So we create
temporary per-sample folders with copies, then convert each one.

Thyra only looks for .tif optical images, but our slides have .jpg,
so we add the optical image to the SpatialData manually after conversion.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import spatialdata as sd
from spatialdata.models import Image2DModel
from thyra import convert_msi

# --- Config ---
SLIDE_DIR = Path(
    "V:/Users/Cuypers_Eva/Nieste_Nadine/Zep/"
    "060326 DHB Slide 11 50 um"
)
OUTPUT_DIR = Path(
    "c:/Users/P70078823/Desktop/Ioana BTR/data/spatialdata_test"
)

# Test with one tissue sample and one matrix blank
TEST_SAMPLES = ["1 1hnr", "matrix 1"]


def prepare_sample_folder(
    slide_dir: Path, sample_name: str, tmp_root: Path
) -> Path:
    """Create a temp folder with just the files for one sample."""
    sample_dir = tmp_root / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Per-sample files
    suffixes = [
        ".dat", ".mis", ".bak",
        "_info.txt", "_msg.txt", "_poslog.txt",
    ]
    for suffix in suffixes:
        src = slide_dir / f"{sample_name}{suffix}"
        if src.exists():
            shutil.copy2(src, sample_dir / src.name)

    # Shared optical images (jpg/tif at the slide level)
    for pattern in ["*.jpg", "*.tif", "*.tiff"]:
        for img in slide_dir.glob(pattern):
            dst = sample_dir / img.name
            if not dst.exists():
                shutil.copy2(img, dst)

    return sample_dir


def add_optical_image(zarr_path: Path, slide_dir: Path):
    """Add the slide optical image (JPG) to the SpatialData.

    The .mis file references the _0000.jpg as the alignment image,
    and the TIC affine is computed in that image's coordinate space.
    We must use _0000.jpg so the overlay is correct.
    """
    from PIL import Image as PILImage

    PILImage.MAX_IMAGE_PIXELS = None  # flexImaging scans are large

    # Use the _0000.jpg -- this is the reference image for the
    # affine transform computed by Thyra from the .mis file
    candidates = sorted(slide_dir.glob("*_0000.jpg"))
    if not candidates:
        # Fall back to any jpg
        candidates = sorted(slide_dir.glob("*.jpg"))
    if not candidates:
        print("  No JPG optical images found, skipping.")
        return

    optical_path = candidates[0]
    print(f"  Adding optical image: {optical_path.name}")

    img = np.array(PILImage.open(optical_path))
    print(f"  Image size: {img.shape[1]}x{img.shape[0]} (WxH)")
    # PIL gives (H, W, C), SpatialData wants (C, Y, X)
    if img.ndim == 3:
        img = img.transpose(2, 0, 1)

    image_element = Image2DModel.parse(img)

    sdata = sd.read_zarr(str(zarr_path))
    sdata.images["optical"] = image_element
    sdata.write_element("optical", overwrite=True)
    print(f"  Optical image added: {img.shape}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="thyra_") as tmp_root:
        tmp_root = Path(tmp_root)

        for sample_name in TEST_SAMPLES:
            sample_dir = prepare_sample_folder(
                SLIDE_DIR, sample_name, tmp_root
            )
            zarr_out = OUTPUT_DIR / f"{sample_name}.zarr"

            print(f"Converting: {sample_name}")
            print(f"  Source: {sample_dir}")
            print(
                f"  Files: "
                f"{[f.name for f in sorted(sample_dir.iterdir())]}"
            )

            success = convert_msi(str(sample_dir), str(zarr_out))
            print(f"  Conversion: {'OK' if success else 'FAILED'}")

            if success:
                add_optical_image(zarr_out, SLIDE_DIR)
            print()

    # Inspect the output
    for zarr_path in sorted(OUTPUT_DIR.glob("*.zarr")):
        print(f"\n--- {zarr_path.name} ---")
        sdata = sd.read_zarr(str(zarr_path))
        print(f"  Tables:  {list(sdata.tables.keys())}")
        print(f"  Images:  {list(sdata.images.keys())}")
        print(f"  Shapes:  {list(sdata.shapes.keys())}")

        if sdata.tables:
            table_name = list(sdata.tables.keys())[0]
            t = sdata.tables[table_name]
            print(
                f"  Table '{table_name}': "
                f"{t.shape[0]} pixels x {t.shape[1]} m/z bins"
            )
            print(f"  Obs columns: {list(t.obs.columns)}")
            for col in ["spatial_x", "spatial_y", "x", "y"]:
                if col in t.obs.columns:
                    print(
                        f"    {col}: {t.obs[col].min():.1f}"
                        f" - {t.obs[col].max():.1f}"
                    )


if __name__ == "__main__":
    main()
