"""Diagnostic plots for converted SpatialData MSI datasets.

For each .zarr in the output folder, generates:
  1. TIC image + mean mass spectrum (overview)
  2. Top 10 ion images by intensity
  3. Optical image with TIC overlay (registered)
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required for batch scripts

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spatialdata as sd
from spatialdata import transformations as sdt

INPUT_DIR = Path(
    "c:/Users/P70078823/Desktop/Ioana BTR/data/spatialdata_zep"
)
OUTPUT_DIR = INPUT_DIR / "diagnostics"

SLIDE_FOLDERS = [
    "060326 DHB Slide 11 50 um",
    "100326 Slide 4 50 um",
    "160326 DHB slide 17 50 um",
    "270226 DHB Slide 7 50 um",
]


def get_mz_values(table):
    """Extract m/z values, handling different var layouts."""
    for col in ["mz", "m/z", "mass"]:
        if col in table.var.columns:
            return table.var[col].values.astype(float)
    try:
        return table.var.index.values.astype(float)
    except (ValueError, TypeError):
        pass
    print(f"  var index sample: {table.var.index[:5].tolist()}")
    print(f"  var columns: {list(table.var.columns)}")
    return np.arange(table.shape[1], dtype=float)


def build_tic_image(table):
    """Build a 2D TIC array from the table."""
    x = table.obs["x"].values.astype(int)
    y = table.obs["y"].values.astype(int)
    tic = np.asarray(table.X.sum(axis=1)).ravel()
    img = np.full((y.max() + 1, x.max() + 1), np.nan)
    img[y, x] = tic
    return img


def plot_tic_image(table, title, ax):
    """Plot the TIC as a 2D image."""
    img = build_tic_image(table)
    im = ax.imshow(img, cmap="inferno", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="TIC", shrink=0.8)


def plot_mean_spectrum(table, mz_values, title, ax):
    """Plot the mean mass spectrum."""
    mean_int = np.asarray(table.X.mean(axis=0)).ravel()
    ax.plot(mz_values, mean_int, linewidth=0.5, color="black")
    ax.set_title(title)
    ax.set_xlabel(
        "m/z" if mz_values.max() > table.shape[1] else "bin index"
    )
    ax.set_ylabel("Mean intensity")
    ax.ticklabel_format(
        axis="y", style="scientific", scilimits=(0, 0)
    )


def plot_top_ions(table, mz_values, title_prefix, fig_path):
    """Plot top 10 ion images ranked by mean intensity."""
    mean_int = np.asarray(table.X.mean(axis=0)).ravel()
    top_idx = np.argsort(mean_int)[::-1][:10]

    x = table.obs["x"].values.astype(int)
    y = table.obs["y"].values.astype(int)
    nx, ny = x.max() + 1, y.max() + 1

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(
        f"{title_prefix} -- Top 10 ions by mean intensity",
        fontsize=14,
    )

    for ax, idx in zip(axes.ravel(), top_idx):
        ion_int = np.asarray(
            table.X[:, idx].todense()
        ).ravel()
        img = np.full((ny, nx), np.nan)
        img[y, x] = ion_int

        mz = mz_values[idx]
        if mz_values.max() > table.shape[1]:
            label = f"m/z {mz:.2f}"
        else:
            label = f"bin {idx}"
        im = ax.imshow(img, cmap="viridis", interpolation="nearest")
        ax.set_title(label, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.7)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path.name}")


def plot_optical_overlay(sdata, table, name, fig_path):
    """Overlay the TIC on the optical image using the affine transform.

    The optical image (_0000.jpg) is the reference coordinate system
    (Identity transform). The TIC carries an Affine that maps its
    raster (x, y) into that same space.
    """
    if "optical" not in sdata.images:
        print("  No optical image found, skipping overlay.")
        return

    # Get optical image (C, Y, X) -> (Y, X, C) for display
    optical = sdata.images["optical"].values
    if optical.shape[0] in (3, 4):
        optical = np.moveaxis(optical, 0, -1)

    # Get the TIC image element
    tic_key = None
    for k in sdata.images:
        if "tic" in k:
            tic_key = k
            break
    if tic_key is None:
        print("  No TIC image element found, skipping overlay.")
        return

    tic_element = sdata.images[tic_key]
    tic_data = tic_element.values[0]  # (C=1, Y, X) -> (Y, X)
    tic_h, tic_w = tic_data.shape

    # Get the affine: TIC (x, y) -> optical (x, y)
    transforms = sdt.get_transformation(
        tic_element, get_all=True
    )
    transform = list(transforms.values())[0]
    affine = np.array(transform.to_affine_matrix(
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    ))

    # Map TIC corners to optical pixel coordinates
    corners = np.array([
        [0, 0, 1],
        [tic_w, 0, 1],
        [0, tic_h, 1],
        [tic_w, tic_h, 1],
    ], dtype=float)
    mapped = (affine @ corners.T).T
    x_min, x_max = mapped[:, 0].min(), mapped[:, 0].max()
    y_min, y_max = mapped[:, 1].min(), mapped[:, 1].max()

    print(
        f"  TIC extent on optical: "
        f"x=[{x_min:.0f}, {x_max:.0f}], "
        f"y=[{y_min:.0f}, {y_max:.0f}]"
    )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(
        f"{name} -- Optical + TIC overlay", fontsize=14
    )

    # Panel 1: optical only
    axes[0].imshow(optical)
    axes[0].set_title("Optical image")
    axes[0].axis("off")

    # Panel 2: TIC overlaid on optical
    # extent = [left, right, bottom, top] for imshow
    axes[1].imshow(optical)
    axes[1].imshow(
        tic_data,
        cmap="inferno",
        alpha=0.7,
        interpolation="bilinear",
        extent=[x_min, x_max, y_max, y_min],
    )
    axes[1].set_title("TIC on optical")
    axes[1].axis("off")

    # Panel 3: zoomed to TIC region
    pad_x = (x_max - x_min) * 0.15
    pad_y = (y_max - y_min) * 0.15
    axes[2].imshow(optical)
    axes[2].imshow(
        tic_data,
        cmap="inferno",
        alpha=0.7,
        interpolation="bilinear",
        extent=[x_min, x_max, y_max, y_min],
    )
    axes[2].set_xlim(x_min - pad_x, x_max + pad_x)
    axes[2].set_ylim(y_max + pad_y, y_min - pad_y)
    axes[2].set_title("Zoomed overlay")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path.name}")


def diagnose_zarr(zarr_path: Path, out_dir: Path):
    """Run all diagnostics on a single .zarr dataset."""
    name = zarr_path.stem

    # Skip if all three outputs already exist
    if all((out_dir / f"{name}_{suffix}").exists() for suffix in
           ["overview.png", "top10_ions.png", "optical_overlay.png"]):
        print(f"\n  Skipping: {name} (already done)")
        return

    print(f"\n  Diagnosing: {name}")

    sdata = sd.read_zarr(str(zarr_path))

    if not sdata.tables:
        print("    No tables found, skipping.")
        return

    table_name = list(sdata.tables.keys())[0]
    table = sdata.tables[table_name]
    print(
        f"    Table: {table_name} "
        f"({table.shape[0]} pixels x {table.shape[1]} m/z bins)"
    )

    mz_values = get_mz_values(table)

    # Figure 1: TIC + mean spectrum overview
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(name, fontsize=14)
    plot_tic_image(table, "TIC image", ax1)
    plot_mean_spectrum(table, mz_values, "Mean mass spectrum", ax2)
    overview_path = out_dir / f"{name}_overview.png"
    fig.tight_layout()
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {overview_path.name}")

    # Figure 2: Top 10 ion images
    ions_path = out_dir / f"{name}_top10_ions.png"
    plot_top_ions(table, mz_values, name, ions_path)

    # Figure 3: Optical + TIC overlay
    overlay_path = out_dir / f"{name}_optical_overlay.png"
    plot_optical_overlay(sdata, table, name, overlay_path)


def main():
    total = 0
    failed = 0

    for slide_name in SLIDE_FOLDERS:
        slide_dir = INPUT_DIR / slide_name
        if not slide_dir.exists():
            print(f"\nSKIP: {slide_name} (not found)")
            continue

        zarr_paths = sorted(slide_dir.glob("*.zarr"))
        if not zarr_paths:
            print(f"\nSKIP: {slide_name} (no .zarr files)")
            continue

        slide_output = OUTPUT_DIR / slide_name
        slide_output.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Slide: {slide_name} ({len(zarr_paths)} datasets)")
        print(f"{'='*60}")

        for zarr_path in zarr_paths:
            total += 1
            try:
                diagnose_zarr(zarr_path, slide_output)
            except Exception as e:
                print(f"  ERROR: {e}")
                failed += 1

    print(f"\n{'='*60}")
    print(f"Done: {total - failed}/{total} OK, {failed} failed")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
