"""
cluster_spectra_analysis.py
----------------------------
Standalone script for manual evaluation of cluster average spectra.
Run after the pipeline has completed for a given run folder.

Usage:
    Set the CONFIG section at the bottom and run:
    poetry run python cluster_spectra_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import spatialdata as sd
import scipy.sparse


# ── CONFIG ────────────────────────────────────────────────────────────────────
# Point this at any run folder — mosaic or single sample
# run_folder = r"C:\Users\i6338212\data\results\liver_PC\OMP_spca10_kmeans4\DHB_060326_DHB_Slide_11_50_um_OMP_spca10_kmeans4_fixed"

# run_folder = r"C:\Users\i6338212\data\results\liver_PC\OMP_pca10_kmeans4\DHB_060326_DHB_Slide_11_50_um_OMP_pca10_kmeans4"
run_folder = r"C:\Users\i6338212\data\results\hippocampus_PC\OMP_spca10_kmeans4_smoothing\hippocampus_OMP_spca10_kmeans4_smoothing"

# For raw spectrum deep dive — list of zarr paths to load
# For mosaic: list all 10 sample zarrs
# For single sample: just one zarr path
# zarr_paths = [
#     r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\1 pra.zarr",
#     r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\10 pra.zarr",
#     r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\21 pra.zarr",
#     r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\28 pra.zarr",
#     r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\49 pra.zarr",

#     r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\1 1hnr.zarr",
#     r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\10 1hnr.zarr",
#     r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\21 1hnr.zarr",
#     r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\28 1hnr.zarr",
#     r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\49 1hnr.zarr"
    
   
# ]
zarr_paths = [
    r"C:\Users\i6338212\data\Ioana Test Data\Data\hippocampus.zarr"
]

# set to True if this is a mosaic run folder, False for single sample
is_mosaic = False

# output folder for plots — defaults to run_folder/cluster_spectra
output_folder = os.path.join(run_folder, "cluster_analysis")
# ──────────────────────────────────────────────────────────────────────────────


def load_run_results(run_folder: str) -> dict:
    """
    Load all saved pipeline outputs from a run folder.
    Works for both mosaic and single sample runs.
    """
    print(f"[load] Loading results from {run_folder}")

    # load cluster labels from the spca/pca results CSV
    results = {}

    # try to find the results CSV (spca, pca, nmf etc)
    for fname in ["spca_results.csv", "pca_results.csv",
                  "nmf_results.csv", "umap_results.csv"]:
        fpath = os.path.join(run_folder, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            results["labels"] = df["cluster"].values
            results["embedding"] = df.iloc[:, :-1].values
            print(f"[load] Loaded labels from {fname} — "
                  f"{len(results['labels'])} pixels, "
                  f"{len(np.unique(results['labels']))} clusters")
            break

    if "labels" not in results:
        raise FileNotFoundError(
            f"No results CSV found in {run_folder}. "
            f"Expected spca_results.csv, pca_results.csv etc."
        )

    # load mask and original shape
    mask_path = os.path.join(run_folder, "mask.npy")
    shape_path = os.path.join(run_folder, "original_shape.npy")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"mask.npy not found in {run_folder}")
    if not os.path.exists(shape_path):
        raise FileNotFoundError(f"original_shape.npy not found in {run_folder}")

    results["mask"] = np.load(mask_path)
    results["original_shape"] = tuple(np.load(shape_path))
    print(f"[load] Mask shape: {results['mask'].shape}, "
          f"Original shape: {results['original_shape']}")

    # load preprocessed matrix (fast path)
    matrix_path = os.path.join(run_folder, "matrix.npy")
    raw_path    = os.path.join(run_folder, "matrix_raw.npy")

    if os.path.exists(matrix_path):
        results["matrix_3d"] = np.load(matrix_path)
        print(f"[load] Loaded 3D matrix: {results['matrix_3d'].shape}")
    if os.path.exists(raw_path):
        results["matrix_raw"] = np.load(raw_path)
        print(f"[load] Loaded raw matrix: {results['matrix_raw'].shape}")

    # load filtered mz values (preprocessed peaks)
    mz_path = os.path.join(run_folder, "filtered_mz_values.csv")
    if os.path.exists(mz_path):
        results["filtered_mz"] = pd.read_csv(mz_path)["mz"].values
        print(f"[load] Loaded {len(results['filtered_mz'])} filtered m/z values")

    return results


def get_cluster_pixel_spectra(matrix_flat: np.ndarray,
                               labels: np.ndarray,
                               cluster_id: int) -> np.ndarray:
    """
    Extract all pixel spectra belonging to a given cluster.
    matrix_flat: (n_nonzero_pixels, n_features)
    labels: (n_nonzero_pixels,) cluster label per pixel
    Returns: (n_cluster_pixels, n_features)
    """
    mask = labels == cluster_id
    return matrix_flat[mask]


def compute_cluster_average_spectra(matrix_flat: np.ndarray,
                                     labels: np.ndarray) -> dict:
    """
    Compute the average spectrum for each cluster.
    Returns dict: {cluster_id: avg_spectrum array}
    """
    unique_clusters = np.unique(labels)
    avg_spectra = {}
    for cid in unique_clusters:
        if cid == -1:
            continue  # skip noise/background
        cluster_pixels = get_cluster_pixel_spectra(matrix_flat, labels, cid)
        avg_spectra[cid] = cluster_pixels.mean(axis=0)
        print(f"  Cluster {cid+1}: {cluster_pixels.shape[0]} pixels")
    return avg_spectra


def plot_preprocessed_spectra(avg_spectra: dict,
                               filtered_mz: np.ndarray,
                               output_folder: str) -> None:
    """
    Plot one PNG per cluster showing its average preprocessed spectrum.
    Also saves a combined overview PNG with all clusters.
    """
    os.makedirs(output_folder, exist_ok=True)
    n_clusters = len(avg_spectra)

    colours = [
        "red", "blue", "green", "orange", "purple",
        "cyan", "magenta", "brown", "pink", "lime"
    ]

    # one plot per cluster
    for cid, avg in avg_spectra.items():
        fig, ax = plt.subplots(figsize=(14, 4))
        colour = colours[cid % len(colours)]

        ax.bar(filtered_mz, avg, width=0.5, color=colour, alpha=0.8)
        ax.set_xlabel("m/z", fontsize=12)
        ax.set_ylabel("Mean TIC-normalised intensity", fontsize=12)
        ax.set_title(f"Cluster {cid+1} — average spectrum "
                     f"({len(filtered_mz)} peaks)", fontsize=14)
        ax.set_xlim(filtered_mz.min() - 10, filtered_mz.max() + 10)

        # annotate top 5 peaks
        top5_idx = np.argsort(avg)[-5:][::-1]
        for idx in top5_idx:
            ax.annotate(
                f"{filtered_mz[idx]:.2f}",
                xy=(filtered_mz[idx], avg[idx]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=colour
            )

        plt.tight_layout()
        save_path = os.path.join(output_folder,
                                  f"cluster_{cid+1}_preprocessed_spectrum.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    # combined overview — all clusters in one figure
    fig, axes = plt.subplots(n_clusters, 1,
                              figsize=(14, 4 * n_clusters),
                              sharex=True)
    if n_clusters == 1:
        axes = [axes]

    for ax, (cid, avg) in zip(axes, avg_spectra.items()):
        colour = colours[cid % len(colours)]
        ax.bar(filtered_mz, avg, width=0.5, color=colour, alpha=0.8)
        ax.set_ylabel("Intensity", fontsize=10)
        ax.set_title(f"Cluster {cid+1}", fontsize=11)

    axes[-1].set_xlabel("m/z", fontsize=12)
    plt.suptitle("Average spectra per cluster (preprocessed peaks)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    overview_path = os.path.join(output_folder,
                                  "all_clusters_preprocessed_overview.png")
    plt.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved overview: {overview_path}")


def load_raw_average_spectra_from_zarrs(zarr_paths: list,
                                         labels: np.ndarray,
                                         mask: np.ndarray,
                                         original_shape: tuple,
                                         is_mosaic: bool,
                                         sample_offset: int = 0) -> tuple:
    """
    Load raw full-resolution spectra from zarr files and compute
    per-cluster averages at full 57800-bin resolution.

    For mosaic runs: loads each zarr, figures out which pixels belong
    to which cluster using the mask and spatial offset.
    For single sample runs: loads one zarr directly.

    Returns: (mz_axis, avg_spectra_dict)
    """
    print("[raw] Loading raw spectra from zarr files...")

    # get mz axis from first zarr
    first_sd = sd.read_zarr(zarr_paths[0])
    first_adata = list(first_sd.tables.values())[0]
    mz_axis = first_adata.var["mz"].values
    n_mz = len(mz_axis)
    print(f"[raw] m/z axis: {n_mz} bins, range "
          f"{mz_axis.min():.2f}–{mz_axis.max():.2f}")

    # reconstruct which label each nonzero pixel has
    # labels are indexed over nonzero pixels only
    height, width = original_shape
    spatial_map = np.full(height * width, -1)
    spatial_map[mask.flatten()] = labels
    spatial_map = spatial_map.reshape(height, width)

    print(f"[debug] spatial_map shape: {spatial_map.shape}")
    print(f"[debug] unique cluster ids in spatial_map: {np.unique(spatial_map)}")
    print(f"[debug] sample_offset: {sample_offset}")


    # accumulate sum and count per cluster across all zarrs
    unique_clusters = [c for c in np.unique(labels) if c != -1]
    cluster_sums   = {c: np.zeros(n_mz, dtype=np.float64)
                      for c in unique_clusters}
    cluster_counts = {c: 0 for c in unique_clusters}

    # for zarr_idx, zarr_path in enumerate(zarr_paths):
    #     sample_name = os.path.basename(zarr_path).replace(".zarr", "")
    #     print(f"[raw] Processing {sample_name} "
    #           f"({zarr_idx+1}/{len(zarr_paths)})...")

    #     sd_data = sd.read_zarr(zarr_path)
    #     adata   = list(sd_data.tables.values())[0]
    #     X       = adata.X
    #     if scipy.sparse.issparse(X):
    #         X = X.toarray()

    #     x_coords = adata.obs["x"].astype(int).values
    #     y_coords = adata.obs["y"].astype(int).values

    #     print(f"[debug] first zarr x range: {x_coords.min()} – {x_coords.max()}")
    #     print(f"[debug] first zarr y range: {y_coords.min()} – {y_coords.max()}")
    #     print(f"[debug] mosaic height: {height}, width: {width}")


    #     for pix_idx, (xi, yi) in enumerate(zip(x_coords, y_coords)):
    #         if is_mosaic:
    #             # in mosaic, sample pixels start at sample_offset rows
    #             mosaic_row = yi + sample_offset
    #             mosaic_col = xi
    #         else:
    #             mosaic_row = yi
    #             mosaic_col = xi

    #         # bounds check
    #         if (mosaic_row >= height or mosaic_col >= width or
    #                 mosaic_row < 0 or mosaic_col < 0):
    #             continue

    #         cluster_id = spatial_map[mosaic_row, mosaic_col]
    #         if cluster_id == -1:
    #             continue

    #         cluster_sums[cluster_id]   += X[pix_idx]
    #         cluster_counts[cluster_id] += 1
    #     print(f"[debug] cluster_counts after first zarr: {cluster_counts}")

    #     print(f"[raw]   Done. Processed {len(x_coords)} pixels.")

    # # compute averages
    # avg_spectra = {}
    # for cid in unique_clusters:
    #     if cluster_counts[cid] > 0:
    #         avg_spectra[cid] = cluster_sums[cid] / cluster_counts[cid]
    #         print(f"  Cluster {cid+1}: {cluster_counts[cid]} raw pixels averaged")
    #     else:
    #         print(f"  Cluster {cid+1}: no pixels found — skipping")


    for zarr_idx, zarr_path in enumerate(zarr_paths):
        sample_name = os.path.basename(zarr_path).replace(".zarr", "")
        print(f"[raw] Processing {sample_name} "
            f"({zarr_idx+1}/{len(zarr_paths)})...")

        sd_data = sd.read_zarr(zarr_path)
        adata   = list(sd_data.tables.values())[0]
        X       = adata.X  # keep as sparse — do NOT call .toarray()

        x_coords = adata.obs["x"].astype(int).values
        y_coords = adata.obs["y"].astype(int).values

        n_pixels = X.shape[0]
        chunk_size = 1000  # process 1000 pixels at a time

        for chunk_start in range(0, n_pixels, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_pixels)

            # extract chunk as dense — only 1000 × 460517 × 8 bytes ≈ 3.5 GB
            # still too big — use row-by-row for very large zarrs
            if scipy.sparse.issparse(X):
                X_chunk = X[chunk_start:chunk_end].toarray()
            else:
                X_chunk = X[chunk_start:chunk_end]

            for pix_idx_local, pix_idx_global in enumerate(
                    range(chunk_start, chunk_end)):

                xi = x_coords[pix_idx_global]
                yi = y_coords[pix_idx_global]

                if is_mosaic:
                    mosaic_row = yi + sample_row_offset
                    mosaic_col = xi + sample_col_offset
                else:
                    mosaic_row = yi
                    mosaic_col = xi

                if (mosaic_row >= height or mosaic_col >= width or
                        mosaic_row < 0 or mosaic_col < 0):
                    continue

                cluster_id = spatial_map[mosaic_row, mosaic_col]
                if cluster_id == -1:
                    continue

                cluster_sums[cluster_id]   += X_chunk[pix_idx_local]
                cluster_counts[cluster_id] += 1

            if chunk_start % 10000 == 0:
                print(f"[raw]   {chunk_end}/{n_pixels} pixels...", flush=True)

    print(f"[raw]   Done. Processed {n_pixels} pixels.")



    return mz_axis, avg_spectra


def plot_raw_spectra_interactive(mz_axis: np.ndarray,
                                  avg_spectra: dict,
                                  output_folder: str) -> None:
    """
    Create one interactive Plotly HTML per cluster showing its
    full-resolution average spectrum (57800 bins).
    """
    os.makedirs(output_folder, exist_ok=True)

    colours = [
        "red", "blue", "green", "orange", "purple",
        "cyan", "magenta", "brown", "pink", "lime"
    ]

    for cid, avg in avg_spectra.items():
        colour = colours[cid % len(colours)]

        # find top 20 peaks for annotation
        top20_idx = np.argsort(avg)[-20:][::-1]

        fig = go.Figure()

        # main spectrum line
        fig.add_trace(go.Scatter(
            x=mz_axis,
            y=avg,
            mode="lines",
            name=f"Cluster {cid+1}",
            line=dict(color=colour, width=1),
            hovertemplate="m/z: %{x:.4f}<br>intensity: %{y:.4f}<extra></extra>"
        ))

        # mark top 20 peaks as scatter points
        fig.add_trace(go.Scatter(
            x=mz_axis[top20_idx],
            y=avg[top20_idx],
            mode="markers+text",
            name="Top 20 peaks",
            marker=dict(color="black", size=6, symbol="triangle-up"),
            text=[f"{mz_axis[i]:.2f}" for i in top20_idx],
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate="m/z: %{x:.4f}<br>intensity: %{y:.4f}<extra></extra>"
        ))

        fig.update_layout(
            title=f"Cluster {cid+1} — full resolution average spectrum",
            xaxis_title="m/z",
            yaxis_title="Mean intensity (raw counts)",
            height=500,
            width=1400,
            legend=dict(x=1.01, y=1),
            hovermode="x unified"
        )

        # add range slider so you can zoom into regions of interest
        fig.update_xaxes(rangeslider_visible=True)

        save_path = os.path.join(output_folder,
                                  f"cluster_{cid+1}_raw_spectrum.html")
        fig.write_html(save_path)
        print(f"  Saved interactive plot: {save_path}")

    # also save a combined figure with all clusters as subplots
    n_clusters = len(avg_spectra)
    fig_all = make_subplots(
        rows=n_clusters, cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Cluster {cid+1}" for cid in avg_spectra.keys()]
    )

    for row, (cid, avg) in enumerate(avg_spectra.items(), 1):
        colour = colours[cid % len(colours)]
        fig_all.add_trace(
            go.Scatter(
                x=mz_axis, y=avg,
                mode="lines",
                name=f"Cluster {cid+1}",
                line=dict(color=colour, width=1),
                hovertemplate="m/z: %{x:.4f}<br>intensity: %{y:.4f}<extra></extra>"
            ),
            row=row, col=1
        )

    fig_all.update_layout(
        title="All clusters — full resolution average spectra",
        height=400 * n_clusters,
        width=1400,
        hovermode="x unified"
    )
    fig_all.update_xaxes(rangeslider_visible=False)

    all_path = os.path.join(output_folder, "all_clusters_raw_overview.html")
    fig_all.write_html(all_path)
    print(f"  Saved combined interactive plot: {all_path}")


def run_cluster_spectrum_analysis(run_folder: str,
                                   zarr_paths: list,
                                   is_mosaic: bool,
                                   output_folder: str) -> None:

    os.makedirs(output_folder, exist_ok=True)

    # load pipeline outputs
    results = load_run_results(run_folder)
    labels         = results["labels"]
    mask           = results["mask"]
    original_shape = results["original_shape"]

    # get sample offset for mosaic (where tissue starts below matrix block)
    sample_offset = 0
    offset_path = os.path.join(run_folder, "sample_offset.npy")
    if os.path.exists(offset_path):
        sample_offset = int(np.load(offset_path)[0])
        print(f"[main] Mosaic sample offset: {sample_offset} rows")

    # ── PREPROCESSED SPECTRA (fast) ───────────────────────────────────────────
    if "matrix_raw" in results and "filtered_mz" in results:
        print("\n[main] Computing cluster averages on preprocessed matrix...")
        matrix_raw  = results["matrix_raw"]   # (n_nonzero_pixels, n_peaks)
        filtered_mz = results["filtered_mz"]

        avg_preprocessed = compute_cluster_average_spectra(matrix_raw, labels)

        print("[main] Plotting preprocessed spectra...")
        plot_preprocessed_spectra(
            avg_preprocessed,
            filtered_mz,
            os.path.join(output_folder, "preprocessed")
        )
    else:
        print("[main] WARNING: matrix_raw.npy or filtered_mz_values.csv not "
              "found — skipping preprocessed spectra.")

    # ── RAW SPECTRA (deep dive) ───────────────────────────────────────────────
    if zarr_paths:
        print("\n[main] Loading raw spectra from zarr files "
              "(this may take a few minutes)...")
        mz_axis, avg_raw = load_raw_average_spectra_from_zarrs(
            zarr_paths=zarr_paths,
            labels=labels,
            mask=mask,
            original_shape=original_shape,
            is_mosaic=is_mosaic,
            sample_offset=sample_offset
        )

        print("[main] Plotting raw interactive spectra...")
        plot_raw_spectra_interactive(
            mz_axis,
            avg_raw,
            os.path.join(output_folder, "raw")
        )
    else:
        print("[main] No zarr paths provided — skipping raw spectra.")

    print(f"\n[main] All done. Results saved to {output_folder}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_cluster_spectrum_analysis(
        run_folder=run_folder,
        zarr_paths=zarr_paths,
        is_mosaic=is_mosaic,
        output_folder=output_folder
    )

