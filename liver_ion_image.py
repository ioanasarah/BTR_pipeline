import matplotlib.pyplot as plt
import numpy as np
import spatialdata as sd
import os
import anndata as ad

# ── CONFIG ──────────────────────────────────────────────────────────────────
batch_root   = r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_zarrs\20260413_L6_C1409_DHB_30um_resample.zarr"
results_folder = r"C:\Users\i6338212\data\results\ion_images"
os.makedirs(results_folder, exist_ok=True)

def is_matrix_zarr(name: str) -> bool:
    return "matrix" in name.lower()

def read_tic_image(zarr_path: str):
    """Read a zarr and return a 2D TIC image array."""
    adata = list(sd.read_zarr(zarr_path).tables.values())[0]
    height = int(adata.obs["y"].max()) + 1
    width  = int(adata.obs["x"].max()) + 1
    tic    = np.array(adata.X.sum(axis=1)).flatten()
    x      = adata.obs["x"].astype(int).values
    y      = adata.obs["y"].astype(int).values
    img    = np.zeros(height * width)
    img[y * width + x] = tic
    return img.reshape(height, width)

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
slide_folders = [
    f for f in os.listdir(batch_root)
    if os.path.isdir(os.path.join(batch_root, f))
]

for slide_folder in sorted(slide_folders):
    slide_path = os.path.join(batch_root, slide_folder)

    zarr_files = sorted([
        f for f in os.listdir(slide_path)
        if f.endswith(".zarr") and os.path.isdir(os.path.join(slide_path, f))
        and not is_matrix_zarr(f)   # skip matrix zarr
    ])

    if not zarr_files:
        print(f"[skip] No sample zarrs found in {slide_folder}")
        continue

    print(f"[{slide_folder}] Found {len(zarr_files)} samples: {zarr_files}")

    tic_images = []
    for zarr_file in zarr_files:
        zarr_path = os.path.join(slide_path, zarr_file)
        try:
            img = read_tic_image(zarr_path)
            tic_images.append((zarr_file.replace(".zarr", ""), img))
            print(f"  ✓ {zarr_file} → shape {img.shape}")
        except Exception as e:
            print(f"  ✗ Failed to read {zarr_file}: {e}")

    if not tic_images:
        continue

    # plot all samples for this slide side by side
    n = len(tic_images)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (sample_name, img) in zip(axes, tic_images):
        im = ax.imshow(img, cmap="hot")
        ax.set_title(sample_name, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    slide_label = slide_folder.replace(" ", "_")
    fig.suptitle(slide_folder, fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(results_folder, f"TIC_{slide_label}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → Saved: {out_path}")

print("Done.")


