"""Debug MNF: compare scaled vs unscaled input to perform_mnf.

Tests the hypothesis that the scale mismatch between noise (raw)
and data (StandardScaler-normalized) is causing the bad clustering.

Runs MNF two ways:
  A) Current code: data = matrix_scaled (smoothed + StandardScaler)
  B) Proposed fix: data = matrix_nmf (raw, unscaled)

Prints eigenvalue stats and plots the first component spatially.
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spatialdata as sd
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from dimensionality_red import (
    smooth_and_scale_matrix,
    get_pixel_coords,
)

# Use one of the converted zarr datasets (small: 13x36x57800)
ZARR_PATH = "c:/Users/P70078823/Desktop/Ioana BTR/data/spatialdata_zep/060326 DHB Slide 11 50 um/matrix 1.zarr"
OUT_DIR = Path("c:/Users/P70078823/Desktop/Ioana BTR/results/debug_mnf")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_FOLDER = str(OUT_DIR)


def load_3d_matrix_from_zarr(zarr_path: str):
    """Reconstruct 3D (H, W, mz) matrix from a SpatialData zarr table."""
    sdata = sd.read_zarr(zarr_path)
    table = list(sdata.tables.values())[0]
    x = table.obs["x"].values.astype(int)
    y = table.obs["y"].values.astype(int)
    h, w = y.max() + 1, x.max() + 1
    n_mz = table.shape[1]
    matrix_3d = np.zeros((h, w, n_mz), dtype=np.float32)
    dense = np.asarray(table.X.todense())
    matrix_3d[y, x, :] = dense
    return matrix_3d

N_COMPONENTS = 5
N_MZ_SUBSAMPLE = 200  # must be < n_pixels (457) so covariance matrix B is full rank


def _mnf_core(X, noise, n_components):
    """Core MNF: returns embedding, eigvals."""
    A = np.cov(X, rowvar=False)
    B = np.cov(noise, rowvar=False) / 2
    eigvals, eigvecs = eigh(A, B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    Z = X @ eigvecs[:, :n_components]
    Z = StandardScaler().fit_transform(Z)
    return Z, eigvals


def plot_component(Z, mask, original_shape, title, out_path):
    """Plot first MNF component as a spatial map."""
    img = np.full(original_shape, np.nan)
    img.ravel()[mask] = Z[:, 0]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(img, cmap="RdBu_r", interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main():
    print("Loading 3D matrix from zarr...")
    matrix_3d = load_3d_matrix_from_zarr(ZARR_PATH)
    height, width, n_peaks = matrix_3d.shape
    print(f"Loaded 3D matrix: {height} x {width} x {n_peaks}")

    # Noise via diagonal neighbor differences (same as load_and_preprocess_msi)
    noise_3d = np.zeros_like(matrix_3d)
    noise_3d[:-1, :-1, :] = matrix_3d[:-1, :-1, :] - matrix_3d[1:, 1:, :]

    X = matrix_3d.reshape(height * width, n_peaks)
    noise_flat = noise_3d.reshape(height * width, n_peaks)

    # Remove zero pixels
    mask = np.sum(X, axis=1) > 0
    original_shape = (height, width)
    matrix_nmf = X[mask]
    noise = noise_flat[mask]
    np.save(f"{RUN_FOLDER}\\mask.npy", mask)
    print(f"Non-zero pixels: {mask.sum()} / {mask.size}")

    # Sub-sample m/z to avoid LAPACK 32-bit integer overflow on full 57800 features
    rng = np.random.default_rng(42)
    mz_idx = rng.choice(n_peaks, size=min(N_MZ_SUBSAMPLE, n_peaks), replace=False)
    mz_idx.sort()
    matrix_nmf = matrix_nmf[:, mz_idx]
    noise = noise[:, mz_idx]
    print(f"Sub-sampled to {matrix_nmf.shape[1]} m/z bins for eigendecomposition")

    coords = get_pixel_coords(mask, original_shape)

    matrix_scaled = smooth_and_scale_matrix(
        matrix_nmf, coords, connectivity=4, run_folder=RUN_FOLDER
    )

    # --- Diagnose scale difference ---
    print(f"\n--- Scale diagnostics ---")
    print(f"matrix_nmf:    mean={matrix_nmf.mean():.4f}, std={matrix_nmf.std():.4f}, max={matrix_nmf.max():.4f}")
    print(f"matrix_scaled: mean={matrix_scaled.mean():.4f}, std={matrix_scaled.std():.4f}, max={matrix_scaled.max():.4f}")
    print(f"noise:         mean={noise.mean():.4f}, std={noise.std():.4f}, max={noise.max():.4f}")

    A_scaled = np.cov(matrix_scaled, rowvar=False)
    A_raw = np.cov(matrix_nmf, rowvar=False)
    B = np.cov(noise, rowvar=False) / 2

    print(f"\n--- Covariance matrix norms ---")
    print(f"A (scaled data): frobenius norm = {np.linalg.norm(A_scaled):.4e}")
    print(f"A (raw data):    frobenius norm = {np.linalg.norm(A_raw):.4e}")
    print(f"B (noise):       frobenius norm = {np.linalg.norm(B):.4e}")
    print(f"\nScale ratio A_scaled/B = {np.linalg.norm(A_scaled)/np.linalg.norm(B):.4e}")
    print(f"Scale ratio A_raw/B    = {np.linalg.norm(A_raw)/np.linalg.norm(B):.4e}")

    # --- Run A: current (scaled data + raw noise) ---
    print(f"\n--- Version A: current code (scaled data, raw noise) ---")
    Z_a, eigvals_a = _mnf_core(matrix_scaled, noise, N_COMPONENTS)
    print(f"Top {N_COMPONENTS} eigenvalues: {eigvals_a[:N_COMPONENTS]}")
    print(f"Eigenvalue range: {eigvals_a.min():.4e} to {eigvals_a.max():.4e}")
    print(f"Z_a variance per component: {Z_a.var(axis=0)}")
    plot_component(
        Z_a, mask, original_shape,
        "Version A: scaled data + raw noise (CURRENT)",
        OUT_DIR / "mnf_A_scaled_data_raw_noise_component1.png"
    )

    # --- Run B: proposed fix (raw data + raw noise) ---
    print(f"\n--- Version B: proposed fix (raw data, raw noise) ---")
    Z_b, eigvals_b = _mnf_core(matrix_nmf, noise, N_COMPONENTS)
    print(f"Top {N_COMPONENTS} eigenvalues: {eigvals_b[:N_COMPONENTS]}")
    print(f"Eigenvalue range: {eigvals_b.min():.4e} to {eigvals_b.max():.4e}")
    print(f"Z_b variance per component: {Z_b.var(axis=0)}")
    plot_component(
        Z_b, mask, original_shape,
        "Version B: raw data + raw noise (PROPOSED FIX)",
        OUT_DIR / "mnf_B_raw_data_raw_noise_component1.png"
    )

    # --- Eigenvalue spectrum plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MNF eigenvalue spectrum (SNR per component)", fontsize=13)
    for ax, eigvals, title in zip(
        axes,
        [eigvals_a[:20], eigvals_b[:20]],
        ["A: scaled data + raw noise (current)", "B: raw data + raw noise (fix)"]
    ):
        ax.bar(range(1, len(eigvals) + 1), eigvals)
        ax.set_title(title)
        ax.set_xlabel("Component")
        ax.set_ylabel("Eigenvalue (SNR)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mnf_eigenvalue_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: mnf_eigenvalue_comparison.png")
    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
