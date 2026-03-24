"""Batch convert all Zep Rapiflex datasets to SpatialData.

Converts all .d datasets across 4 slides, adding optical images
from the _0000.jpg reference scan. Output is one .zarr per sample.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import spatialdata as sd
from PIL import Image as PILImage
from spatialdata.models import Image2DModel
from thyra import convert_msi

PILImage.MAX_IMAGE_PIXELS = None

# --- Config ---
ZEP_DIR = Path("V:/Users/Cuypers_Eva/Nieste_Nadine/Zep")

OUTPUT_DIR = Path(
    "c:/Users/P70078823/Desktop/Ioana BTR/data/spatialdata_zep"
)

SLIDE_FOLDERS = [
    "060326 DHB Slide 11 50 um",
    "100326 Slide 4 50 um",
    "160326 DHB slide 17 50 um",
    "270226 DHB Slide 7 50 um",
]


def find_samples(slide_dir: Path) -> list[str]:
    """Find all sample names in a slide folder by looking for .dat files."""
    return sorted(
        f.stem for f in slide_dir.glob("*.dat")
    )


def prepare_sample_folder(
    slide_dir: Path, sample_name: str, tmp_root: Path
) -> Path:
    """Create a temp folder with just the files for one sample."""
    sample_dir = tmp_root / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    suffixes = [
        ".dat", ".mis", ".bak",
        "_info.txt", "_msg.txt", "_poslog.txt",
    ]
    for suffix in suffixes:
        src = slide_dir / f"{sample_name}{suffix}"
        if src.exists():
            shutil.copy2(src, sample_dir / src.name)

    # Shared optical images
    for pattern in ["*.jpg", "*.tif", "*.tiff"]:
        for img in slide_dir.glob(pattern):
            dst = sample_dir / img.name
            if not dst.exists():
                shutil.copy2(img, dst)

    return sample_dir


def add_optical_image(zarr_path: Path, slide_dir: Path):
    """Add the _0000.jpg reference optical image to the SpatialData."""
    candidates = sorted(slide_dir.glob("*_0000.jpg"))
    if not candidates:
        candidates = sorted(slide_dir.glob("*.jpg"))
    if not candidates:
        return

    optical_path = candidates[0]
    img = np.array(PILImage.open(optical_path))
    if img.ndim == 3:
        img = img.transpose(2, 0, 1)

    image_element = Image2DModel.parse(img)
    sdata = sd.read_zarr(str(zarr_path))
    sdata.images["optical"] = image_element
    sdata.write_element("optical", overwrite=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_ok = 0
    total_fail = 0

    for slide_name in SLIDE_FOLDERS:
        slide_dir = ZEP_DIR / slide_name
        if not slide_dir.exists():
            print(f"SKIP: {slide_name} (not found)")
            continue

        samples = find_samples(slide_dir)
        print(f"\n{'='*60}")
        print(f"Slide: {slide_name} ({len(samples)} samples)")
        print(f"{'='*60}")

        slide_output = OUTPUT_DIR / slide_name
        slide_output.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="thyra_") as tmp:
            tmp_root = Path(tmp)

            for i, sample_name in enumerate(samples, 1):
                zarr_out = slide_output / f"{sample_name}.zarr"

                if zarr_out.exists():
                    print(f"  [{i}/{len(samples)}] {sample_name} -- already exists, skipping")
                    total_ok += 1
                    continue

                print(f"  [{i}/{len(samples)}] {sample_name} ...", end=" ", flush=True)

                sample_dir = prepare_sample_folder(
                    slide_dir, sample_name, tmp_root
                )

                try:
                    success = convert_msi(str(sample_dir), str(zarr_out))
                except Exception as e:
                    print(f"ERROR: {e}")
                    total_fail += 1
                    continue

                if success:
                    try:
                        add_optical_image(zarr_out, slide_dir)
                    except Exception as e:
                        print(f"OK (optical failed: {e})")
                        total_ok += 1
                        continue
                    print("OK")
                    total_ok += 1
                else:
                    print("FAILED")
                    total_fail += 1

                # Clean up temp files for this sample
                shutil.rmtree(sample_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"Done: {total_ok} OK, {total_fail} failed")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
