import os
import time
import numpy as np
import spatialdata as sd
import anndata as ad
# import spatialdata_plot  # noqa: F401
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matchms 
from matchms import Spectrum
from sklearn.linear_model import OrthogonalMatchingPursuit
import pandas as pd
import os
import scipy.sparse
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from scipy.sparse import issparse
# import msiwarp as mw

print("Loaded packages for preprocessing")

# results_folder = r"C:\Ioana\_uni\BTR_pipeline_code\results" # change folder path as needed
# # results_folder = r"C:\Users\i6338212\data\results"
# preprocessing_run_name = "small_computer_xenium_omp"
# run_folder = os.path.join(results_folder, preprocessing_run_name)
# os.makedirs(run_folder, exist_ok=True)

start_time = time.perf_counter()

def reading_data(path):
    print("reading data...")
    # path = r"C:\Ioana\_uni\btr\zarr\MALDI-MSI Mouse Brain.zarr\MALDI-MSI Mouse Brain.zarr"
    spatial_data = sd.read_zarr(path)
    print(f"data read in {time.perf_counter() - start_time:.2f} seconds")    
    return spatial_data  
# how can i load this once and store it so i dont have to run this every time? 


def compute_average_spectrum(
        spatial_data
        ):
    
    print("computing average spectrum...")
    # data = spatial_data["MALDI-MSI_z0"] # for maldi msi mouse brain zarr
    data = list(spatial_data.tables.values())[0]
    mz = data.var["mz"].values
    avg_intensity = data.uns["average_spectrum"] # unstructured annotation within anndata object
    # average intensity at each m/z across all pixels


    average_spectrum = Spectrum(
        mz=mz, 
        intensities = avg_intensity,
        metadata={"id": "average_spectrum"})
    
    print(f"average spectrum computed in {time.perf_counter() - start_time:.2f} seconds")
    
    # plt.plot(mz, avg_intensity)
    # plt.xlabel("m/z")
    # plt.ylabel("Average Intensity")
    # plt.title("Average Mass Spectrum")
    # plt.show()
    return data, mz, avg_intensity, average_spectrum





def check_mass_drift(data, run_folder, n_pixels=1000):
    """
    Compare average spectra from first vs last N pixels to visualise mass drift.
    """
    mz_axis = data.var["mz"].values
    X = data.X

    # compute average for first and last n_pixels
    first = np.array(X[:n_pixels].mean(axis=0)).flatten()
    last = np.array(X[-n_pixels:].mean(axis=0)).flatten()

    # plot overlaid
    plt.figure(figsize=(14, 4))
    plt.plot(mz_axis, first, label=f"First {n_pixels} pixels", alpha=0.7)
    plt.plot(mz_axis, last, label=f"Last {n_pixels} pixels", alpha=0.7)
    plt.xlabel("m/z")
    plt.ylabel("Average Intensity")
    plt.title("Mass Drift Check: First vs Last Pixels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "mass_drift_check_full.png"), dpi=150)
    plt.show()

    # zoom into a high-intensity region to see shifts more clearly
    # find the top peak in the overall average
    overall_avg = np.array(X.mean(axis=0)).flatten()
    top_peak_idx = np.argmax(overall_avg)
    top_peak_mz = mz_axis[top_peak_idx]

    zoom_window = 2.0  # Da either side
    zoom_mask = np.abs(mz_axis - top_peak_mz) < zoom_window

    plt.figure(figsize=(10, 4))
    plt.plot(mz_axis[zoom_mask], first[zoom_mask], label=f"First {n_pixels} pixels", alpha=0.7)
    plt.plot(mz_axis[zoom_mask], last[zoom_mask], label=f"Last {n_pixels} pixels", alpha=0.7)
    plt.xlabel("m/z")
    plt.ylabel("Average Intensity")
    plt.title(f"Zoom around top peak at m/z ≈ {top_peak_mz:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "mass_drift_check_zoom.png"), dpi=150)
    plt.show()

    # print the shift estimate at the top peak
    first_peak_mz = mz_axis[zoom_mask][np.argmax(first[zoom_mask])]
    last_peak_mz = mz_axis[zoom_mask][np.argmax(last[zoom_mask])]
    shift = abs(first_peak_mz - last_peak_mz)
    print(f"Top peak position: first={first_peak_mz:.4f}, last={last_peak_mz:.4f}")
    print(f"Estimated drift at top peak: {shift:.4f} Da")
    print("Verdict:", "significant drift (consider alignment)" if shift > 0.1 else "drift likely negligible")

    return shift

def linear_recalibration(data, reference_mz, reference_intensity, run_folder,
                          n_landmarks=5, mz_tolerance=0.3):
    """
    Align each pixel spectrum to a reference via linear m/z recalibration.
    
    data: AnnData object containing MSI spectra
    reference_mz: m/z axis of the average spectrum
    reference_intensity: intensity values of the average spectrum
    n_landmarks: number of anchor peaks used for recalibration 
    mz_tolerance: search window (Da) around each landmark
    """
    
    print("performing linear recalibration...")
    # pick landmark peaks from the avg spectrum
    top_idxs = np.argsort(reference_intensity)[-n_landmarks * 3:]  # oversample
    # np.argsort returns indices sorted by intensity
    # selecting only top peaks by intensity 
    # *3 bcs some peaks may be missing from some pixels, we want to have enough landmarks to find matches in each pixel
    landmark_mz = reference_mz[top_idxs] # getting m/z values of the top peaks
    
    mz_axis = data.var["mz"].values


    total_pixels = data.shape[0] # number of spectra = pixels 
    X = data.X # pixels by m/z bins

    # #only selecting a subsection of pixels to test on --> to change, set n_pixels to total pixels
    # max_pixels = 200 # set a maximum number of pixels to process for recalibration to save time (can adjust as needed)
    # n_pixels = min(max_pixels, total_pixels)
    nonzero_pixels = np.where(X.getnnz(axis=1) > 0)[0]
    n_pixels = len(nonzero_pixels)
    
    
    corrected_rows = np.zeros((n_pixels, len(mz_axis)), dtype=np.float32) # initialize array to store corrected spectra for each pixel
    
    # checking shit 
    # print("Total nonzero entries:", data.X.nnz)
    # print("Total pixels:", data.shape[0])

    # selecting non-zero pixels to run subset 
    # nonzero_pixels = np.where(data.X.getnnz(axis=1) > 0)[0]
    # selected_pixels = nonzero_pixels[:n_pixels]
    # for i_idx, i in enumerate(selected_pixels):
    #     if i % 100 == 0:
    #         print(f"Processed {i}/{n_pixels} pixels")

    #     pixel = np.array(X[i].todense()).flatten() if issparse(X) else X[i] # get each pixel spectrum as a dense array
    #     # print(pixel.max())
    #     # 1D array of intensities for the current pixel
        
    #     measured_peaks = [] # observed m/z in pixel
    #     reference_peaks = [] # expected m/z in reference
        
    #     for lm in landmark_mz:
    #         # find the window around landmark in the pixel
    #         window = np.abs(mz_axis - lm) < mz_tolerance # boolean mask for mz values within +/- tolerance of landmark
    #         if window.sum() == 0:
    #             continue # safety check - if no bins in the window then skip
    #         local_intensities = pixel[window] # get intensities near the expected landmark 
    #         if local_intensities.max() == 0:
    #             continue  # if peak absent in this pixel
            
    #         # take the mz of the local maximum
    #         local_mz = mz_axis[window] # first find index
    #         measured_peaks.append(local_mz[np.argmax(local_intensities)]) # then get the mz value at the local max intensity
    #         reference_peaks.append(lm) # store expected mz for this landmark peak
    #         # have a matched pair (observed mz, expected mz) for this landmark in this pixel
        
    #     # at least two points are required to fit a line
    #     if len(measured_peaks) < 2:
    #         # corrected_rows.append(pixel)  # cant fit a line, leave spectrum uncorrected
    #         corrected_rows[i] = pixel.astype(np.float32)
    #         continue
        
    #     # fit linear correction: mz_true = a * mz_measured + b
    #     a, b = np.polyfit(measured_peaks, reference_peaks, deg=1)
    #     # print(a, b)
    #     corrected_mz = a * mz_axis + b
    #     # resample pixel onto original mz_axis grid --> go through this again 

    
    #     corrected_pixel = np.interp(mz_axis, corrected_mz, pixel)
    #     corrected_rows[i_idx] = corrected_pixel.astype(np.float32) # store corrected spectrum for this pixel
    

# uncomment this for normal, all pixel analysis 
    for i_idx, i in enumerate(nonzero_pixels): # iterate through pixels
        if i_idx % 100 == 0:
            print(f"Processed {i_idx}/{n_pixels} pixels")

        pixel = np.array(X[i].todense()).flatten() if issparse(X) else X[i] # get each pixel spectrum as a dense array
        # print(pixel.max())
        # 1D array of intensities for the current pixel
        
        measured_peaks = [] # observed m/z in pixel
        reference_peaks = [] # expected m/z in reference
        
        for lm in landmark_mz:
            # find the window around landmark in the pixel
            window = np.abs(mz_axis - lm) < mz_tolerance # boolean mask for mz values within +/- tolerance of landmark
            if window.sum() == 0:
                continue # safety check - if no bins in the window then skip
            local_intensities = pixel[window] # get intensities near the expected landmark 
            if local_intensities.max() == 0:
                continue  # if peak absent in this pixel
            
            # take the mz of the local maximum
            local_mz = mz_axis[window] # first find index
            measured_peaks.append(local_mz[np.argmax(local_intensities)]) # then get the mz value at the local max intensity
            reference_peaks.append(lm) # store expected mz for this landmark peak
            # have a matched pair (observed mz, expected mz) for this landmark in this pixel
        
        # at least two points are required to fit a line
        if len(measured_peaks) < 2:
            # corrected_rows.append(pixel)  # cant fit a line, leave spectrum uncorrected
            corrected_rows[i_idx] = pixel.astype(np.float32)
            continue
        
        # fit linear correction: mz_true = a * mz_measured + b
        a, b = np.polyfit(measured_peaks, reference_peaks, deg=1)
        # print(a, b)
        corrected_mz = a * mz_axis + b
        # resample pixel onto original mz_axis grid --> go through this again 

    
        corrected_pixel = np.interp(mz_axis, corrected_mz, pixel)
        corrected_rows[i_idx] = corrected_pixel.astype(np.float32) # store corrected spectrum for this pixel
    

    #  save corrected spectra as a new AnnData object or overwrite existing one
    np.savez_compressed(
    os.path.join(run_folder, "recalibrated_spectra.npz"),
    spectra=np.array(corrected_rows, dtype=np.float32),
    mz_axis=mz_axis)
    print(f"Recalibrated spectra saved to {os.path.join(run_folder, 'recalibrated_spectra.npz')}. Took {time.perf_counter() - start_time:.2f} seconds")
    # returns corrected spectra matrix and unchanged m/z axis
    
    aligned_matrix = np.array(corrected_rows)

    print(f"Linear recalibration completed in {time.perf_counter() - start_time:.2f} seconds")
    return aligned_matrix, mz_axis



def msiwarp_recalibration(data, reference_mz, reference_intensity):
    spectra =  ... # code to load the unaligned spectra
    reference_spectrum =  ... 

    # setup the node placement parameters
    params = mw.params_uniform(...)
    epsilon = 1.0 # peak matching threshold, relative to peak width
    n_cores = 4

    # find an m/z recalibration function for each spectrum
    recal_funcs = mw.find_optimal_warpings_uni(spectra, reference_spectrum, params, epsilon, n_cores)

    # use the recalibration functions to warp the spectra
    warped_spectra = [mx.warp_peaks_unique(s_i, r_i) for (s_i, r_i) in zip(spectra, recal_funcs)]

# ... code to store the warped spectra


def identify_matrix_peaks(
                          corner_fraction: float,
                           sample_zarr_paths: list,
                           candidate_mz: np.ndarray,
                           run_folder: str,
                           matrix_zarr_path: str = None,
                           top_n_images: int = 50,
                           tol: float = 0.1) -> pd.DataFrame:
    # load the first sample zarr to get the m/z axis (assuming all samples share the same m/z axis)
    s_adata_first = list(sd.read_zarr(sample_zarr_paths[0]).tables.values())[0]
    mz_axis = s_adata_first.var["mz"].values

    # load matrix zarr and get average spectrum
    if matrix_zarr_path is not None:
        msd = sd.read_zarr(matrix_zarr_path)
        m_adata = list(msd.tables.values())[0]
        mz_axis = m_adata.var["mz"].values
        matrix_avg = m_adata.uns["average_spectrum"]  # shape (57800,)
        
        # to use for ion images
        m_x = m_adata.obs["x"].astype(int).values
        m_y = m_adata.obs["y"].astype(int).values
        m_width  = m_x.max() + 1
        m_height = m_y.max() + 1
        X_matrix_source = m_adata  # used below for ion image extraction
    else:
        print(f"[identify_matrix_peaks] No matrix zarr provided. "
              f"Extracting top-left {corner_fraction:.0%} corner as matrix region.")
        # if no matrix zarr provided, we can still generate ion images from the original data by assuming the matrix is in the top-left corner
        corner_sums  = np.zeros(len(mz_axis), dtype=np.float64)
        corner_count = 0
        s_adata= list(sd.read_zarr(sample_zarr_paths[0]).tables.values())[0]
        x_all = s_adata_first.obs["x"].astype(int).values
        y_all = s_adata_first.obs["y"].astype(int).values
        global_width  = x_all.max() + 1
        global_height = y_all.max() + 1

        x_threshold = int(global_width  * corner_fraction)
        y_threshold = int(global_height * corner_fraction)
        print(f"  Corner region: x < {x_threshold}, y < {y_threshold} "
              f"(image is {global_width} x {global_height})")
        # collect corner pixels across all sample zarrs
        # (if matrix is embedded in one zarr, just pass that one zarr)
        for zarr_path in sample_zarr_paths:
            sd_data = sd.read_zarr(zarr_path)
            adata   = list(sd_data.tables.values())[0]
            x_coords = adata.obs["x"].astype(int).values
            y_coords = adata.obs["y"].astype(int).values

            corner_mask = (x_coords < x_threshold) & (y_coords < y_threshold)
            n_corner = corner_mask.sum()
            print(f"  {os.path.basename(zarr_path)}: {n_corner} corner pixels found")

            if n_corner == 0:
                continue

            X = adata.X
            if scipy.sparse.issparse(X):
                corner_spectra = np.array(X[corner_mask].toarray(), dtype=np.float64)
            else:
                corner_spectra = X[corner_mask].astype(np.float64)

            corner_sums  += corner_spectra.sum(axis=0)
            corner_count += n_corner
        if corner_count == 0:  
            raise ValueError("No corner pixels found across all zarrs. Cannot estimate matrix spectrum.")   
        matrix_avg = corner_sums / corner_count
        print(f"  Matrix average computed from {corner_count} corner pixels.")
        
        # for ion image generation in the no-matrix-zarr path:
        # use the corner pixels of the first zarr
        m_x = x_all[
            (x_all < x_threshold) & (y_all < y_threshold)
        ]
        m_y = y_all[
            (x_all < x_threshold) & (y_all < y_threshold)
        ]
        m_width  = x_threshold
        m_height = y_threshold
        X_matrix_source = None  # handled separately below
    
    
    # load each samples average spectrum and average them together
    sample_avgs = []
    for zarr_path in sample_zarr_paths:
        sd_data = sd.read_zarr(zarr_path)
        s_adata = list(sd_data.tables.values())[0]
        sample_avgs.append(s_adata.uns["average_spectrum"]) 

    # mean across all 10 samples — shape (57800,)
    sample_avg = np.mean(sample_avgs, axis=0)
    
    
    # for each candidate peak, find the closest bin in the raw mz axis
    results_rows = []
    for mz_val in candidate_mz:
        col_idx = np.argmin(np.abs(mz_axis - mz_val))
        results_rows.append({
            "mz":         mz_val,
            "matrix_avg": matrix_avg[col_idx],
            "sample_avg": sample_avg[col_idx],
            "ratio":      matrix_avg[col_idx] / (sample_avg[col_idx] + 1e-9)
        })

    results_df = pd.DataFrame(results_rows).sort_values(
        "ratio", ascending=False
    ).reset_index(drop=True)
    results_df["rank"] = results_df.index + 1

    print(f"[identify_matrix_peaks] Evaluating {len(candidate_mz)} peaks")
    print(results_df[["rank", "mz", "ratio"]].head(60).to_string())

    csv_path = os.path.join(run_folder, "matrix_peak_candidates.csv")
    results_df.to_csv(csv_path, index=False)



    # # each m/z bin we want to see how much higher it is in matrix vs samples
    # ratio = matrix_avg / (sample_avg + 1e-9) # 1e-9 to avoid division by 0

    # # build df
    # results_df = pd.DataFrame({
    #     "mz":         mz_axis,      
    #     "matrix_avg": matrix_avg,   # avg intensity in matrix pixels
    #     "sample_avg": sample_avg,   # avg intensity in sample pixels
    #     "ratio":      ratio         # matrix_avg / sample_avg
    # }).sort_values("ratio", ascending=False).reset_index(drop=True)
    # results_df["rank"] = results_df.index + 1  # rank 1 = most matrix-like


    # # checking hings 
    # print("Matrix avg max:", matrix_avg.max())
    # print("Matrix avg mean:", matrix_avg.mean())
    # print("Sample avg max:", sample_avg.max())
    # print("Sample avg mean:", sample_avg.mean())

    # print(results_df[["rank", "mz", "ratio"]].head(60).to_string())


    # # save the full ranked list as csv
    # csv_path = os.path.join(run_folder, "matrix_peak_candidates.csv")
    # results_df.to_csv(csv_path, index=False)


    # make subfolder fo ion images
    ion_image_folder = os.path.join(run_folder, "matrix_peak_ion_images")
    os.makedirs(ion_image_folder, exist_ok=True)

    # # get the x,y pixel coordinates of the matrix zarr pixels
    # m_x = m_adata.obs["x"].astype(int).values
    # m_y = m_adata.obs["y"].astype(int).values
    # # work out the spatial dimensions of the matrix block
    # m_width  = m_x.max() + 1
    # m_height = m_y.max() + 1


    n_to_plot = min(top_n_images, len(results_df))
    top_col_idxs = [
        np.argmin(np.abs(mz_axis - results_df.iloc[r]["mz"]))
        for r in range(n_to_plot)
    ]

    # load X_matrix only here, only for ion image generation
    # we only need the top N columns so extract those only
    if matrix_zarr_path is not None:
        # matrix zarr exists — load from m_adata
        X_matrix = m_adata.X
        if scipy.sparse.issparse(X_matrix):
            X_matrix_subset = np.array(X_matrix[:, top_col_idxs].todense())
        else:
            X_matrix_subset = X_matrix[:, top_col_idxs]
    else:
        # no matrix zarr — load corner pixels from first sample zarr
        s_adata_corner = list(sd.read_zarr(sample_zarr_paths[0]).tables.values())[0]
        x_all_c = s_adata_corner.obs["x"].astype(int).values
        y_all_c = s_adata_corner.obs["y"].astype(int).values
        corner_mask = (x_all_c < x_threshold) & (y_all_c < y_threshold)

        X_corner = s_adata_corner.X
        if scipy.sparse.issparse(X_corner):
            X_matrix_subset = np.array(X_corner[corner_mask][:, top_col_idxs].todense())
        else:
            X_matrix_subset = X_corner[corner_mask][:, top_col_idxs]


    first_sample_avg = sample_avgs[0]
    first_sample_name = os.path.basename(sample_zarr_paths[0]).replace(".zarr", "")

    for rank in range(n_to_plot):
        row = results_df.iloc[rank]
        mz_val = row["mz"]
        col_idx = np.argmin(np.abs(mz_axis - mz_val))

        ion_image = np.zeros((m_height, m_width), dtype=np.float32)
        for i, (xi, yi) in enumerate(zip(m_x, m_y)):
            ion_image[yi, xi] = X_matrix_subset[i, rank]

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        im = axes[0].imshow(ion_image, cmap="hot", interpolation="nearest")
        plt.colorbar(im, ax=axes[0], label="Intensity")
        axes[0].set_title(f"Matrix spatial ion image\nm/z {mz_val:.4f}")
        axes[0].axis("off")

        axes[1].bar(["Matrix avg", "All samples avg"],
                    [row["matrix_avg"], row["sample_avg"]],
                    color=["red", "steelblue"])
        axes[1].set_title(f"All-sample comparison\nRatio: {row['ratio']:.1f}x")
        axes[1].set_ylabel("Average intensity (raw counts)")

        first_sample_intensity = first_sample_avg[col_idx]
        axes[2].bar(["Matrix avg", f"{first_sample_name} avg"],
                    [row["matrix_avg"], first_sample_intensity],
                    color=["red", "darkorange"])
        first_ratio = row["matrix_avg"] / (first_sample_intensity + 1e-9)
        axes[2].set_title(f"Single sample\nRank {rank+1} | ratio: {first_ratio:.1f}x")
        axes[2].set_ylabel("Average intensity (raw counts)")

        plt.suptitle(f"m/z {mz_val:.4f}  —  rank {rank+1}  —  mean ratio {row['ratio']:.1f}x",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        save_path = os.path.join(ion_image_folder,
                                  f"rank{rank+1:02d}_mz{mz_val:.2f}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[identify_matrix_peaks] Ion images saved to {ion_image_folder}")
    return results_df

def filter_nonphysical_peaks(peak_mz: np.ndarray, mz_cutoff: float = 600.0,
                              tol: float = 0.1) -> np.ndarray:
    """
    Remove peaks whose m/z fractional part is suspiciously non-integer.
    Real MALDI ions have fractional m/z parts near 0 (singly charged).
    Doubly-charged / ghost OMP peaks cluster around 0.3-0.9.
    """
    frac = peak_mz - np.floor(peak_mz)
    is_low_mz = peak_mz < mz_cutoff
    is_ghost = (frac >= tol) & (frac <= (1.0 - tol))

    remove = is_low_mz & is_ghost
    keep = ~remove
    # # keep peaks where fractional part is within tol of 0 or 1
    # keep = (frac < tol) | (frac > (1.0 - tol))
    # n_removed = (~keep).sum()
    print(f"[filter_nonphysical] Removed {remove.sum()} ghost peaks (m/z < {mz_cutoff})")
    return peak_mz[keep]


def filter_matrix_peaks(peak_mz: np.ndarray,
                         matrix_peaks_df: pd.DataFrame,
                         ratio_threshold: float,
                         tol: float = 0.1) -> tuple:
    """
    Remove peaks from peak_mz that match flagged matrix peaks above ratio_threshold.
    Returns filtered peak_mz and a list of removed peaks.
    """
    flagged_mz = matrix_peaks_df[
        matrix_peaks_df["ratio"] >= ratio_threshold
    ]["mz"].values

    keep = []
    removed = []
    for mz in peak_mz:
        if np.any(np.abs(flagged_mz - mz) <= tol):
            removed.append(mz)
        else:
            keep.append(mz)

    print(f"[filter_matrix_peaks] Removed {len(removed)} matrix peaks, "
          f"kept {len(keep)} peaks (threshold={ratio_threshold}x)")
    return np.array(keep), np.array(removed)





def stack_matrix_spatially(sample_matrix_3d: np.ndarray,
                            matrix_zarr_path: str,
                            filtered_mz: np.ndarray,
                            mz_axis: np.ndarray,
                            gap: int=5):
    msd = sd.read_zarr(matrix_zarr_path)
    matrix_adata = list(msd.tables.values())[0]

    m_x = matrix_adata.obs["x"].astype(int)
    m_y = matrix_adata.obs["y"].astype(int)

    m_width = m_x.max() + 1
    m_height = m_y.max() + 1
    n_mz = sample_matrix_3d.shape[2]

    # build matrix onl spatial blocl
    matrix_block = np.zeros((m_height, m_width, n_mz), dtype=sample_matrix_3d.dtype)
    X = matrix_adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray()

    # select only the columns matching the filtered m/z values from preprocessing 
    peak_idxs = [np.argmin(np.abs(mz_axis - mz)) for mz in filtered_mz] 
    X = X[:, peak_idxs] 
    # now shape is (n_matrix_pixels, 226) 
    
    # TIC normalise matrix pixels to match sample normalisation 
    tic = X.sum(axis=1) 
    tic = np.where(tic == 0, 1, tic) 
    X = X / tic[:, np.newaxis]



    for i, (xi, yi) in enumerate(zip(m_x, m_y)):
        matrix_block[yi, xi, :] = X[i]
    

    # build combined array
    s_height, s_width, _ = sample_matrix_3d.shape
    new_height = m_height + gap + s_height
    new_width = max(m_width, s_width)

    combined = np.zeros((new_height, new_width, n_mz), dtype=sample_matrix_3d.dtype)

    # matrix block o top left
    combined [:m_height, :m_width, :] = matrix_block
    # sample below gap
    offset = m_height + gap
    combined[offset:offset + s_height, :s_width, :] = sample_matrix_3d


    print(f"[stack_matrix_spatially] Matrix block: rows 0–{m_height}, cols 0–{m_width}") 
    print(f"[stack_matrix_spatially] Sample block: rows {offset}–{offset + s_height}") 
    print(f"[stack_matrix_spatially] Combined shape: {combined.shape}")

    return combined, offset
    



# def harmonise_mz_axes(sample_matrices: list, 
#                     sample_mz_lists: list,
#                     full_mz_axes: list,
#                     tol: float = 0.01,
#                     min_presence=0.7) -> tuple:
#     """
#     Find the intersection of all filtered_mz lists (within tolerance),
#     then reindex each sample matrix to the common axis.
#     Returns: (list of reindexed 3D matrices, common_mz array)
#     """
#     print("[harmonise] Finding common m/z axis (intersection)...")

#     # start with first sample's mz list as reference
#     common_mz = sample_mz_lists[0].copy()

#     for i, mz_list in enumerate(sample_mz_lists):
#         print(f"  Sample {i+1}: {len(mz_list)} peaks, range {mz_list.min():.2f}–{mz_list.max():.2f}")

#     n_samples = len(sample_mz_lists)
#     min_count = int(np.ceil(min_presence * n_samples))

#     all_peaks = np.sort(np.concatenate(sample_mz_lists))

#     bins = []
#     i = 0 
#     while i < len(all_peaks):
#         cluster = [all_peaks[i]]
#         j = i+1 
#         while j < len(all_peaks) and all_peaks[j] - all_peaks[i] <= tol:
#             cluster.append(all_peaks[j])
#             j=+1
#         bins.append((np.mean(cluster), len(cluster)))
#         i=j


#     # common_mz = sample_mz_lists[0].copy()
#     common_mz = []
#     # print(f"  Starting with {len(common_mz)} peaks from sample 1")

#     for bin_center, _ in bins:
#         count = sum(
#             1 for mz_list in sample_mz_lists
#             if np.any(np.abs(mz_list - bin_center) <=tol)
#         )
#         if count >= min_count:
#             common_mz.append(bin_center)


#     common_mz = np.array(common_mz)
#     for i, mz_list in enumerate(sample_mz_lists[1:], 2):
#         matched = []
#         for mz in common_mz:
#             # keep only if this mz appears in the other sample within tolerance
#             if np.any(np.abs(mz_list - mz) <= tol):
#                 matched.append(mz)
#         common_mz = np.array(matched)

#     print(f"[harmonise] Common m/z peaks: {len(common_mz)}")

#     # reindex each sample matrix to common_mz
#     reindexed = []
#     for i, (matrix_3d, filtered_mz) in enumerate(zip(sample_matrices, sample_mz_lists)):
#         h, w, _ = matrix_3d.shape
#         new_matrix = np.zeros((h, w, len(common_mz)), dtype=matrix_3d.dtype)
#         for new_idx, target_mz in enumerate(common_mz):
#             # find closest peak in this sample's filtered_mz
#             old_idx = np.argmin(np.abs(filtered_mz - target_mz))
#             if np.abs(filtered_mz[old_idx] - target_mz) <= tol:
#                 new_matrix[:, :, new_idx] = matrix_3d[:, :, old_idx]
#             # else leave as zeros (peak absent in this sample)
#         reindexed.append(new_matrix)

#     return reindexed, common_mz

def harmonise_mz_axes(sample_matrices, sample_mz_lists, full_mz_axes,
                       tol=0.05, min_presence=0.2):
    print("[harmonise] Building common m/z axis...")
    for i, mz_list in enumerate(sample_mz_lists):
        print(f"  Sample {i+1}: {len(mz_list)} peaks, range {mz_list.min():.2f}–{mz_list.max():.2f}")

    n_samples = len(sample_mz_lists)
    min_count = int(np.ceil(min_presence * n_samples))

    # pool and sort all peaks
    all_peaks = np.sort(np.concatenate(sample_mz_lists))

    # greedily bin close peaks into cluster centres
    bin_centers = []
    i = 0
    while i < len(all_peaks):
        j = i + 1
        while j < len(all_peaks) and all_peaks[j] - all_peaks[i] <= tol:
            j += 1
        bin_centers.append(np.mean(all_peaks[i:j]))
        i = j
    bin_centers = np.array(bin_centers)
    print(f"[harmonise] Total candidate bins: {len(bin_centers)}")

    # count presence per bin — process one sample at a time, no large matrix
    counts = np.zeros(len(bin_centers), dtype=int)
    for mz_list in sample_mz_lists:
        mz_sorted = np.sort(mz_list)
        for k, center in enumerate(bin_centers):
            # binary search — O(log n) instead of O(n)
            idx = np.searchsorted(mz_sorted, center)
            found = False
            for offset in [-1, 0, 1]:  # check nearest neighbours only
                j = idx + offset
                if 0 <= j < len(mz_sorted) and abs(mz_sorted[j] - center) <= tol:
                    found = True
                    break
            if found:
                counts[k] += 1

    common_mz = bin_centers[counts >= min_count]
    print(f"[harmonise] Common peaks (>={min_presence*100:.0f}% of samples): {len(common_mz)}")

    # reindex each sample matrix to common_mz — one sample at a time
    reindexed = []
    for i, (matrix_3d, filtered_mz) in enumerate(zip(sample_matrices, sample_mz_lists)):
        h, w, _ = matrix_3d.shape
        new_matrix = np.zeros((h, w, len(common_mz)), dtype=np.float32)  # float32 saves memory
        mz_sorted = np.sort(filtered_mz)
        mz_sorted_idx = np.argsort(filtered_mz)

        for new_idx, target_mz in enumerate(common_mz):
            idx = np.searchsorted(mz_sorted, target_mz)
            for offset in [-1, 0, 1]:
                j = idx + offset
                if 0 <= j < len(mz_sorted) and abs(mz_sorted[j] - target_mz) <= tol:
                    orig_idx = mz_sorted_idx[j]
                    new_matrix[:, :, new_idx] = matrix_3d[:, :, orig_idx]
                    break

        print(f"  Sample {i+1}: reindexed to {len(common_mz)} common peaks", flush=True)
        reindexed.append(new_matrix)
        # free original matrix from memory immediately
        sample_matrices[i] = None

    return reindexed, common_mz

def batch_correct_by_sample(reindexed_matrices: list) -> list:
    """
    Remove inter-sample intensity offsets using per-sample median-centering.
    Clips to 0 to preserve non-negativity (mask compatibility).
    """
    corrected = []
    for i, matrix_3d in enumerate(reindexed_matrices):
        h, w, n_mz = matrix_3d.shape
        flat = matrix_3d.reshape(-1, n_mz)
        
        nonzero = flat.sum(axis=1) > 0
        flat_nz = flat[nonzero]
        
        # subtract per-feature median (robust to outliers, preserves relative differences)
        median = np.median(flat_nz, axis=0)
        flat_nz = flat_nz - median
        
        # clip negatives to 0 — keeps mask valid, removes offset only
        flat_nz = np.clip(flat_nz, 0, None)
        
        flat[nonzero] = flat_nz
        corrected.append(flat.reshape(h, w, n_mz))
        print(f"[batch_correct] Sample {i+1} median-centred and clipped.")
    
    return corrected




def build_slide_mosaic(sample_matrices: list, n_pra: int,
                        matrix_zarr_path: str, common_mz: np.ndarray,
                        full_mz_axis: np.ndarray, gap: int = 10) -> tuple:
    """
    Arrange samples in 2 rows (pra top, 1hnr bottom), with matrix block top-left.
    Returns combined 3D mosaic and a dict of sample offsets for provenance.
    """

    print("Buildin slide mosaic")
    pra_matrices = sample_matrices[:n_pra]
    hnr_matrices = sample_matrices[n_pra:]

    def make_row(matrices, gap):
        # pad all to same height, then concatenate horizontally with gaps
        max_h = max(m.shape[0] for m in matrices)
        n_mz  = matrices[0].shape[2]
        row_parts = []
        for m in matrices:
            h, w, _ = m.shape
            # pad height if needed
            pad = np.zeros((max_h - h, w, n_mz), dtype=m.dtype)
            m_padded = np.concatenate([m, pad], axis=0)
            row_parts.append(m_padded)
            # add horizontal gap
            row_parts.append(np.zeros((max_h, gap, n_mz), dtype=m.dtype))
        return np.concatenate(row_parts, axis=1)

    pra_row = make_row(pra_matrices, gap)
    hnr_row = make_row(hnr_matrices, gap)

    # pad rows to same width
    n_mz   = common_mz.shape[0]
    max_w  = max(pra_row.shape[1], hnr_row.shape[1])

    def pad_width(row, target_w):
        if row.shape[1] < target_w:
            pad = np.zeros((row.shape[0], target_w - row.shape[1], n_mz), dtype=row.dtype)
            return np.concatenate([row, pad], axis=1)
        return row

    pra_row = pad_width(pra_row, max_w)
    hnr_row = pad_width(hnr_row, max_w)

    # vertical gap between rows
    row_gap = np.zeros((gap, max_w, n_mz), dtype=pra_row.dtype)
    sample_block = np.concatenate([pra_row, row_gap, hnr_row], axis=0)



    # add matrix block top-left
    if matrix_zarr_path:
        sample_block, sample_offset = stack_matrix_spatially(
            sample_block, matrix_zarr_path,
            filtered_mz=common_mz,
            mz_axis=full_mz_axis,
            gap=gap
        )
    else:
        sample_offset = 0

    print(f"[build_slide_mosaic] Final mosaic shape: {sample_block.shape}")
    return sample_block, sample_offset



def median_filter_spectrum(intensity, kernel_size=5):
    """
    Apply median filter to smooth spectrum before peak detection.

    kernel_size must be odd (e.g., 3, 5, 7).
    """
    print(f"Applying median filter (kernel size = {kernel_size})...")
    filtered = medfilt(intensity, kernel_size=kernel_size)
    return filtered 


def savgol_filter_spectrum(intensity, window_length=11, polyorder=3):
    print(f"Applying Savitzky-Golay filter (window={window_length}, poly={polyorder})...")
    filtered = savgol_filter(intensity, window_length=window_length, polyorder=polyorder)
    filtered = np.clip(filtered, 0, None)  # remove negative values from smoothing
    return filtered

def gaussian_filter_spectrum(intensity, sigma=1.0):
    print(f"Applying Gaussian filter (sigma={sigma})...")
    filtered = gaussian_filter(intensity, sigma=sigma)
    return filtered


def peak_detection_mad(
        mz, 
        avg_intensity, 
        window_size=20, 
        snr=2
        ):

    print("detecting peaks using MADestimation...")

    # compute mad noise
    median = np.median(avg_intensity)
    mad_noise = np.median(np.abs(avg_intensity - median))

    # intensity is twice higher than the noise level, where the intensity of noise was determined by MADestimation
    threshold = snr * mad_noise
    print(f"threshold for peak detection: {threshold:.2f}")

    peak_idxs = []

    # intensity is maximum in local window size (set to 20)
    for i in range(window_size, len(avg_intensity) - window_size):
        local_window = avg_intensity[i-window_size:i+window_size+1]

        if avg_intensity[i] == np.max(local_window):
            if avg_intensity[i] > threshold:
                peak_idxs.append(i)

    peak_mz = mz[peak_idxs]
    peak_intensities = avg_intensity[peak_idxs]
    print(f"Detected {len(peak_mz)} peaks in {time.perf_counter() - start_time:.2f} seconds") # should have below 800 peaks ish
    return peak_mz, peak_intensities


def no_matrix_peaks(avg_intensity: np.ndarray,
                    mz_axis: np.ndarray,
                    matrix_peaks_df: pd.DataFrame,
                    ratio_threshold: float,
                    tol: float = 0.1):

    flagged_mz = matrix_peaks_df[matrix_peaks_df["ratio"] >= ratio_threshold]["mz"].values
    
    suppressed = avg_intensity.copy()
    for mz in flagged_mz:
        window = np.abs(mz_axis - mz) <= tol
        suppressed[window] = 0.0
    
    n_suppressed= np.sum(suppressed == 0) - np.sum(avg_intensity == 0)
    print(f"[no_matrix_peaks] zeroed out {len(flagged_mz)} matrix peaks")
    print(f"[no_matrix_peaks] {n_suppressed} bins suppressed")
    return suppressed
    

def peak_detection_omp(mz, 
        avg_intensity,
        run_folder, 
        window_size=0.07, 
        non_zero_coefs=700, 
        candidate_intensity=None):
    print("detecting peaks using OMP...")



    intensity_for_candidates = candidate_intensity if candidate_intensity is not None else avg_intensity
    # use mad to get candidate peaks for OMP
    median = np.median(intensity_for_candidates)
    mad_noise = np.median(np.abs(intensity_for_candidates - median))
    # threshold = 2 * mad_noise # 2 is the snr 
    threshold = 1.5 * mad_noise

    candidate_peak_idxs = []
    window = 20 # index-based window for local max check
    for i in range(window, len(intensity_for_candidates) - window):
        local_window = intensity_for_candidates[i-window:i+window+1]
        if intensity_for_candidates[i] == np.max(local_window) and intensity_for_candidates[i] > threshold:
            candidate_peak_idxs.append(i)
    # this loop checks each m/z peak to see if its the highest intensity in a +/- 20 idx window AND if its above noise threshold
    # if it is, then it gets appended to the candidate peaks -- which will be used for omp

    candidate_idxs = np.array(candidate_peak_idxs)
    candidate_mz = mz[candidate_idxs]
    print(f"MAD found {len(candidate_mz)} candidates for OMP dictionary")


    # starts OMP peak detection using the candidate peaks as the dictionary

    X = np.zeros((len(mz), len(candidate_mz))) # initialize a matrix X with rows corresponding to m/z values and columns corresponding to candidate peaks
    for i, cm in enumerate(candidate_mz):
        X[:, i] = np.exp(-0.5 * ((mz - cm) / window_size) ** 2)
    # Creates a matrix X where each column is a Gaussian "peak shape" centred at one candidate m/z. This represents what a perfect peak would look like at that position
# e^(-0.5 * ((mz - cm) / window_size) ** 2) is the formula for a Gaussian function, where cm is the center of the peak and window_size controls the width of the peak. This creates a bell-shaped curve that peaks at cm and falls off as you move away from cm, with the rate of falloff determined by window_size.
    n_coefs = min(non_zero_coefs, len(candidate_mz)) #
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_coefs)
    omp.fit(X, avg_intensity) # fits the OMP model to find the best combination of candidate peaks that can reconstruct the average spectrum with a sparse set of coefficients (non_zero_coefs controls how many peaks we want to keep)

    coef = omp.coef_
    selected_idxs = coef.nonzero()[0]
    peak_mz = candidate_mz[selected_idxs]
    peak_intensities = avg_intensity[candidate_idxs[selected_idxs]]

    pd.DataFrame({"mz": peak_mz}).to_csv(f"{run_folder}\\peak_mz_values.csv", index=False)

    print(f"Detected {len(peak_mz)} peaks in {time.perf_counter() - start_time:.2f} seconds")
    return peak_mz, peak_intensities


def peak_binning(
        peak_mz, 
        run_folder,
        tolerance=0.005
):
    print("binning peaks...")

    sorted_mz_idxs = np.argsort(peak_mz) # returns indices that would sort the peak masses
    peak_mz = peak_mz[sorted_mz_idxs] # reorders masses in ascending order

    bins= []
    current_bin = [peak_mz[0]] # start with the first mass as the first bin
    
    for mz_val in peak_mz[1:]: 
        mean_mass = np.mean(current_bin) # compute mean mass of the current bin
        if abs(mz_val - mean_mass) / mean_mass < tolerance: 
            # if the new mass is within the tolerance of the mean mass, add it to the current bin
        #    checks that peak is within 0.5% of mean mass of the bin
            current_bin.append(mz_val) # then appends it to the bin = its a peak from the same compound
        else:
            bins.append(np.mean(current_bin)) # if not, it starts a new bin with the new mass
            current_bin = [mz_val] # start new bin with the new mass

    bins.append(np.mean(current_bin)) # add the last bin after the loop
    print(f"Binned peaks into {len(bins)} bins in {time.perf_counter() - start_time:.2f} seconds")
    pd.DataFrame({"mz": bins}).to_csv(f"{run_folder}\\binned_mz_values.csv", index=False)
    return np.array(bins)

def pooling(
        aligned_matrix, 
        mz_recalibrated,
        bins
): # All appeared ions are pooled into a peak-list
    print("pooling spectra...")
    mz_axis = mz_recalibrated # m/z values for each ion

# for each consensus mass find closest index in mz axis and extract column from matrix
#  --> only select meaningful peaks ?
    peak_idxs = [
        np.argmin(np.abs(mz_axis - peak))
        for peak in bins
        ]
    pooled_spectra = aligned_matrix[:, peak_idxs] # extract columns corresponding to the binned peaks
    pooled_mz = mz_axis[peak_idxs]

   
    print(f"Pooled spectra with {len(bins)} peaks in {time.perf_counter() - start_time:.2f} seconds")
    return pooled_spectra, pooled_mz


def filtering(
        pooled_spectra, 
        pooled_mz, 
        run_folder
):
    print("filtering spectra...")
    presence = (pooled_spectra > 0).sum(axis=0) # count how many pixels each peak appears
    threshold = 0.01 * pooled_spectra.shape[0] # set threshold as 1% of total number of spectra
    # threshold = 50

    valid_peaks = np.array(presence).flatten() > threshold # keep peaks that are present in more than 1% of spectra
    filtered_spectra = pooled_spectra[:, valid_peaks] # filter the spectra to keep only valid peaks
    filtered_mz = pooled_mz[valid_peaks]
    pd.DataFrame({"mz": filtered_mz}).to_csv(
        f"{run_folder}\\filtered_mz_values.csv",
        index=False
    )
    # print("Threshold:", threshold)
    # print("Max presence:", np.max(presence))
    # print("Median presence:", np.median(np.array(presence).flatten()))

    #  print(f"shape: {filtered_spectra.shape}")
    # print(filtered_spectra)
    return presence, filtered_spectra, filtered_mz


def tic_normalization(filtered_spectra: np.ndarray, run_folder:str,
                      target: float = 1.0):
    print("performing TIC normalization...")
    # densify before normalisation
    if issparse(filtered_spectra):
        print("  [tic_normalization] densifying sparse matrix...")
        filtered_spectra = filtered_spectra.toarray()
    
    tic = filtered_spectra.sum(axis=1) # total ion current for each spectrum / for each pixel
    tic = np.where(tic == 0, 1, tic) # avoid division by zero
    normalised_matrix = (filtered_spectra / tic[:, np.newaxis]) * target
    
    # TIC should be aprox 1.0 for all non-zero pixels
    non_zero_tics = normalised_matrix.sum(axis=1)
    non_zero_tics = non_zero_tics[non_zero_tics > 0]
    if scipy.sparse.issparse(normalised_matrix):
        normalised_matrix = normalised_matrix.toarray()
    np.save(os.path.join(run_folder, "normalised_matrix.npy"), normalised_matrix)
    print(f"Post-normalisation TIC — mean: {non_zero_tics.mean():.4f}, std: {non_zero_tics.std():.6f}")
    print(f"TIC normalization complete in {time.perf_counter() - start_time:.2f} seconds")
    
    return normalised_matrix

def reshaping_to_3d_matrix(
        data, 
        filtered_spectra
        ):
    print("reshaping to 3D matrix...")
    x_coords = data.obs["x"].values
    y_coords = data.obs["y"].values

    x_int = x_coords.astype(int)
    y_int = y_coords.astype(int)
    width = int(x_int.max()) + 1
    height = int(y_int.max()) + 1
    print(f"Reshaping to 3D matrix with dimensions: ({height}, {width}, {filtered_spectra.shape[1]})")

    if issparse(filtered_spectra):
        spectra_array = filtered_spectra.toarray()
    else:
        spectra_array = filtered_spectra
    matrix = np.zeros((height, width, spectra_array.shape[1]), dtype=spectra_array.dtype)
    for i, (xi, yi) in enumerate(zip(x_int, y_int)):
        matrix[yi, xi, :] = spectra_array[i]
    # matrix = filtered_spectra.reshape(height, width, -1)
    print(f"Reshaped matrix shape: {matrix.shape} in {time.perf_counter() - start_time:.2f} seconds")
    
    return matrix

# spatial_data = reading_data()
# data, mz, avg_intensity, average_spectrum = compute_average_spectrum(spatial_data)
# print(f"mz data type: {type(mz)}")
# # print("Average spectrum m/z values:", mz)
# print("Mean difference between adjacent m/z values:", np.diff(mz).mean())
# pd.DataFrame({"mz": peak_mz}).to_csv(f"{results_folder}\\peak_mz_values_omp.csv", index=False)


# if __name__ == "__main__":
#     # zarr_path = r"C:\Users\i6338212\data\Ioana Test Data\Data\xenium.zarr"
#     zarr_path=r"C:\Users\i6338212\data\Ioana Test Data\Data\hippocampus.zarr"
#     spatial_data = reading_data(zarr_path)
#     AnnData, mz, avg_intensity, average_spectrum = compute_average_spectrum(spatial_data)
#     print(f"data type: {type(AnnData)}, mz type: {type(mz)}, avg_intensity type: {type(avg_intensity)}")
#     shift = check_mass_drift(AnnData, n_pixels=1000)

#     # aligned_matrix, mz_axis = linear_recalibration(AnnData, mz, avg_intensity)

#     # aligned_matrix, mz = linear_recalibration(AnnData, mz, avg_intensity)
#     # print(f"data type after recalibration: {type(aligned_matrix)}, mz type: {type(mz)}")
#     # # print first 10 values of aligned matrix and mz to check
#     # print("First 10 m/z values after recalibration:", mz[:10])
#     # print("First 10 intensity values of the first pixel after recalibration:", aligned_matrix[0, :10])
#     # # recompute average on aigned spectra
#     # avg_intensity = aligned_matrix.mean(axis=0)
#     # print(f"Average intensity after recalibration: {avg_intensity[:10]}") # print first 10 values to check")

#     peak_mz, peak_intensities = peak_detection_omp(mz, avg_intensity, non_zero_coefs=700)
#     # peak_mz_mad, peak_intensities_mad = peak_detection_mad(mz, avg_intensity, window_size=20, snr=2)
#     # print(f"mz data type: {type(mz)}")
#     # # print("Average spectrum m/z values:", mz)
#     # print("Mean difference between adjacent m/z values:", np.diff(mz).mean())
#     pd.DataFrame({"mz": peak_mz}).to_csv(f"{run_folder}\\peak_mz_values.csv", index=False)

#     bins = peak_binning(peak_mz, tolerance=0.005)
#     pd.DataFrame({"mz": bins}).to_csv(f"{run_folder}\\binned_mz_values.csv", index=False)
#     print(f"Binned m/z values saved to {run_folder}\\binned_mz_values.csv")

#     pooled_spectra, pooled_mz = pooling(aligned_matrix=AnnData.X, mz_recalibrated=mz, bins=bins)
#     presence, filtered_spectra, filtered_mz = filtering(pooled_spectra, pooled_mz)
#     pd.DataFrame({"mz": filtered_mz}).to_csv(
#         f"{run_folder}\\filtered_mz_values.csv",
#         index=False
#     )
#     print(f"Filtered m/z values saved to {run_folder}\\filtered_mz_values.csv")

#     normalised_matix = tic_normalization(filtered_spectra)
#     if issparse(normalised_matix):
#         normalised_matix = normalised_matix.toarray()


#     matrix = reshaping_to_3d_matrix(AnnData, normalised_matix)
# # # print("Nonzeros:", np.count_nonzero(matrix))
# # plt.imshow(matrix[:, :, 94], cmap="hot", interpolation="nearest") # visualize the ion image for the peak at index 98 (closest to 647.47 m/z)
# # plt.colorbar()
# # plt.title("Ion Image for Peak 0")
# # plt.show()
# # print(np.max(matrix[:, :, 0]))

#     np.save("msi_matrix.npy", matrix)


# # # Reshaping to 3D matrix with dimensions: (1469, 1007, 142)
# # # Reshaped matrix shape: (1469, 1007, 142) in 13.56 seconds



def preprocess_single_sample(zarr_path: str, 
                            params: dict, 
                            run_folder: str,
                            matrix_peaks_df=None) -> tuple:
    """
    Run preprocessing on one sample zarr.
    Returns the 3D normalised matrix and its filtered_mz axis.
    """

    sample_name = os.path.basename(zarr_path).replace(".zarr", "")
    sample_folder = os.path.join(run_folder, "per_sample", sample_name)
    os.makedirs(sample_folder, exist_ok=True)

    
    spatial_data = reading_data(zarr_path)
    data, mz, avg_intensity, _ = compute_average_spectrum(spatial_data)

    # DEFINE FILTERING
    if params.get("filtering") == "median":
        avg_intensity = median_filter_spectrum(avg_intensity, kernel_size=5)
    elif params.get("filtering") == "savgol":
        avg_intensity = savgol_filter_spectrum(avg_intensity, window_length=11, polyorder=3)
    elif params.get("filtering") == "gaussian":
        avg_intensity = gaussian_filter_spectrum(avg_intensity, sigma=1.0)

    # DEFINE PEAK DETECTION
    if params["peak_method"] == "OMP":
        peak_mz, _ = peak_detection_omp(mz, avg_intensity, run_folder,
                                         non_zero_coefs=params["omp_coefs"])
     
    else:
        peak_mz, _ = peak_detection_mad(mz, avg_intensity, window_size=20, snr=2)

    peak_mz = filter_nonphysical_peaks(peak_mz, tol=0.15)

    if matrix_peaks_df is not None and params.get("matrix_ratio_threshold"):
        peak_mz, removed = filter_matrix_peaks(
            peak_mz,
            matrix_peaks_df,
            ratio_threshold=params["matrix_ratio_threshold"],
            tol=0.2
        )

        print(f"[preprocess_single_sample] peaks before matrix filter: {len(peak_mz) + len(removed)}")
        print(f"[preprocess_single_sample] peaks after matrix filter: {len(peak_mz)}")
        print(f"[preprocess_single_sample] removed matrix peaks: {removed}")
        pd.DataFrame({"mz": removed}).to_csv(
            os.path.join(sample_folder, "removed_matrix_peaks.csv"), index=False
        )



    bins = peak_binning(peak_mz, run_folder, tolerance=params["bin_tol"])
    pooled_spectra, pooled_mz = pooling(data.X, mz, bins)
    _, filtered_spectra, filtered_mz = filtering(pooled_spectra, pooled_mz, run_folder)
    normalised = tic_normalization(filtered_spectra, run_folder)
    matrix_3d = reshaping_to_3d_matrix(data, normalised)

    return matrix_3d, filtered_mz, mz


def run_preprocessing(params, run_folder):
    """Slide-level preprocessing: preprocess each sample, harmonise, build mosaic."""  

    sample_zarr_paths = params.get("sample_zarr_paths")

    # if running in old single-sample mode, fall back gracefully
    if not sample_zarr_paths:
        sample_zarr_paths = [params.get("zarr_path")]

    print(f"[preprocessing] Processing {len(sample_zarr_paths)} samples...")

    sample_matrices = []
    sample_mz_lists = []
    full_mz_axes    = []


    if params.get("batch_mode", False):
        # run omp on first sample to get candidat mz
        ref_zarr = sample_zarr_paths[0]
        ref_sd = reading_data(ref_zarr)
        ref_data, ref_mz, ref_avg, _ = compute_average_spectrum(ref_sd)
        ref_peak_mz, _ = peak_detection_omp(
            ref_mz, ref_avg, run_folder,
            non_zero_coefs=params["omp_coefs"]
        )
        print(f"[preprocessing] Reference OMP found {len(ref_peak_mz)} candidate peaks")

    # identify matrix peaks first if matrix zarr available
        
        matrix_peaks_df = None
        matrix_zarr_path = params.get("matrix_zarr_path")
        if matrix_zarr_path and params.get("matrix_ratio_threshold"):
            matrix_peaks_df = identify_matrix_peaks(
                matrix_zarr_path=matrix_zarr_path,
                sample_zarr_paths=params["sample_zarr_paths"],
                candidate_mz=ref_peak_mz,
                run_folder=run_folder,
                top_n_images=20
            )
        elif params.get("matrix_zarr_path") or params.get("matrix_ratio_threshold"):
            # always generate ion images even if threshold not set yet
            matrix_peaks_df = identify_matrix_peaks(
                matrix_zarr_path=matrix_zarr_path,
                sample_zarr_paths=params["sample_zarr_paths"],
                candidate_mz=ref_peak_mz,
                run_folder=run_folder,
                top_n_images=20
            )
            print("[preprocessing] matrix_ratio_threshold not set — peaks identified "
              "but not filtered. Check ion images and set threshold in params.")

        for i, zarr_path in enumerate(sample_zarr_paths):
            print(f"----------------------------------------------- Sample {i+1}/{len(sample_zarr_paths)}: {os.path.basename(zarr_path)}-----------------------------------------------------------")
            matrix_3d, filtered_mz, full_mz = preprocess_single_sample(zarr_path, params, run_folder, matrix_peaks_df)
            sample_matrices.append(matrix_3d)
            sample_mz_lists.append(filtered_mz)
            full_mz_axes.append(full_mz)

        # harmonise to common m/z axis
        reindexed_matrices, common_mz = harmonise_mz_axes(
            sample_matrices, sample_mz_lists, full_mz_axes, tol=0.05
        )

        # save common mz for feature selection downstream
        pd.DataFrame({"mz": common_mz}).to_csv(
            f"{run_folder}/filtered_mz_values.csv", index=False
        )

        # build mosaic
        mosaic, sample_offset = build_slide_mosaic(
            reindexed_matrices,
            n_pra=params.get("n_pra", len(reindexed_matrices) // 2),
            matrix_zarr_path=params.get("matrix_zarr_path"),
            common_mz=common_mz,
            full_mz_axis=full_mz_axes[0],  # all share same raw mz axis
            gap=10
        )
        if sample_offset > 0:
            mosaic[:sample_offset, :, :] = 0.0   # exclude matrix block from clustering
            print(f"[run_preprocessing] Matrix block excluded: rows 0–{sample_offset}")

        np.save(f"{run_folder}/sample_offset.npy", np.array([sample_offset]))
        np.save(f"{run_folder}/matrix.npy", mosaic)

        # np.save(f"{run_folder}/sample_offset.npy", np.array([sample_offset]))
        # np.save(f"{run_folder}/matrix.npy", mosaic)

        print(f"[preprocessing] Mosaic saved. Shape: {mosaic.shape}")

        return {
            "matrix_path": f"{run_folder}/matrix.npy",
            "n_features":  mosaic.shape[-1]
        }
    else:
        # ── SINGLE SAMPLE MODE ────────────────────────────────────────────
        print("[preprocessing] Running in single sample mode...")
        spatial_data = reading_data(params["zarr_path"])
        AnnData, mz, filtered_avg_intensity, _ = compute_average_spectrum(spatial_data)

        if params.get("filtering") == "median":
            avg_intensity = median_filter_spectrum(filtered_avg_intensity, kernel_size=5)
        elif params.get("filtering") == "savgol":
            avg_intensity = savgol_filter_spectrum(filtered_avg_intensity, window_length=11, polyorder=3)
        elif params.get("filtering") == "gaussian":
            avg_intensity = gaussian_filter_spectrum(filtered_avg_intensity, sigma=1.0)

        if params["peak_method"] == "OMP":
            peak_mz, _ = peak_detection_omp(
                mz, filtered_avg_intensity, run_folder,
                non_zero_coefs=params["omp_coefs"]
            )
        else:
            peak_mz, _ = peak_detection_mad(
                mz, filtered_avg_intensity, window_size=20, snr=2
            )

        # peak_mz = filter_nonphysical_peaks(peak_mz, tol=0.15)
        
        # print(f"[debug] len(spatial_data.var['mz'].values): {len(spatial_data.var['mz'].values)}")
        matrix_zarr_path = params.get("matrix_zarr_path")
        # if matrix_zarr_path or params.get("matrix_ratio_threshold"):
        
        matrix_peaks_df = identify_matrix_peaks(
                # matrix_zarr_path=matrix_zarr_path,
                corner_fraction=0.2,
                sample_zarr_paths=sample_zarr_paths,
                candidate_mz=peak_mz,
                run_folder=run_folder,
                matrix_zarr_path=matrix_zarr_path,
                top_n_images=20
            )
        print(matrix_peaks_df)
        # if matrix_peaks_df is not None and params.get("matrix_ratio_threshold"):
        #     peak_mz, removed = filter_matrix_peaks(
        #         peak_mz,
        #         matrix_peaks_df,
        #         ratio_threshold=params["matrix_ratio_threshold"],
        #         tol=0.2
        #     )

        bins = peak_binning(peak_mz, run_folder, tolerance=params["bin_tol"])
        pooled_spectra, pooled_mz = pooling(AnnData.X, mz, bins)
        _, filtered_spectra, filtered_mz = filtering(
            pooled_spectra, pooled_mz, run_folder
        )
        normalized_matrix = tic_normalization(filtered_spectra, run_folder)
        matrix = reshaping_to_3d_matrix(AnnData, normalized_matrix)

        np.save(f"{run_folder}/matrix.npy", matrix)

        return {
            "matrix_path": f"{run_folder}/matrix.npy",
            "n_features":  matrix.shape[-1]
        }

