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
    data = spatial_data["msi_dataset_z0"]
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
        if i % 100 == 0:
            print(f"Processed {i}/{n_pixels} pixels")

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
    

def preprocess_single_sample(zarr_path: str, params: dict, run_folder: str):
    """
    Run preprocessing on one sample zarr.
    Returns the 3D normalised matrix and its filtered_mz axis.
    """

    sample_name = os.path.basename(zarr_path).replace(".zarr", "")
    sample_folder = os.path.join(run_folder, "per_sample", sample_name)
    os.makedirs(sample_folder, exist_ok=True)

    
    spatial_data = reading_data(zarr_path)
    data, mz, avg_intensity, _ = compute_average_spectrum(spatial_data)

    if params.get("filtering") == "median":
        avg_intensity = median_filter_spectrum(avg_intensity, kernel_size=5)

    if params["peak_method"] == "OMP":
        peak_mz, _ = peak_detection_omp(mz, avg_intensity, run_folder,
                                         non_zero_coefs=params["omp_coefs"])
    else:
        peak_mz, _ = peak_detection_mad(mz, avg_intensity, window_size=20, snr=2)

    bins = peak_binning(peak_mz, run_folder, tolerance=params["bin_tol"])
    pooled_spectra, pooled_mz = pooling(data.X, mz, bins)
    _, filtered_spectra, filtered_mz = filtering(pooled_spectra, pooled_mz, run_folder)
    normalised = tic_normalization(filtered_spectra, run_folder)
    matrix_3d = reshaping_to_3d_matrix(data, normalised)

    return matrix_3d, filtered_mz, mz


def harmonise_mz_axes(sample_matrices: list, sample_mz_lists: list,
                       full_mz_axes: list, tol: float = 0.01) -> tuple:
    """
    Find the intersection of all filtered_mz lists (within tolerance),
    then reindex each sample matrix to the common axis.
    Returns: (list of reindexed 3D matrices, common_mz array)
    """
    print("[harmonise] Finding common m/z axis (intersection)...")

    # start with first sample's mz list as reference
    common_mz = sample_mz_lists[0].copy()

    for i, mz_list in enumerate(sample_mz_lists):
        print(f"  Sample {i+1}: {len(mz_list)} peaks, range {mz_list.min():.2f}–{mz_list.max():.2f}")


    common_mz = sample_mz_lists[0].copy()
    print(f"  Starting with {len(common_mz)} peaks from sample 1")

    for i, mz_list in enumerate(sample_mz_lists[1:], 2):
        matched = []
        for mz in common_mz:
            # keep only if this mz appears in the other sample within tolerance
            if np.any(np.abs(mz_list - mz) <= tol):
                matched.append(mz)
        common_mz = np.array(matched)

    print(f"[harmonise] Common m/z peaks: {len(common_mz)}")

    # reindex each sample matrix to common_mz
    reindexed = []
    for matrix_3d, filtered_mz, full_mz in zip(sample_matrices, sample_mz_lists, full_mz_axes):
        h, w, _ = matrix_3d.shape
        new_matrix = np.zeros((h, w, len(common_mz)), dtype=matrix_3d.dtype)
        for new_idx, target_mz in enumerate(common_mz):
            # find closest peak in this sample's filtered_mz
            old_idx = np.argmin(np.abs(filtered_mz - target_mz))
            if np.abs(filtered_mz[old_idx] - target_mz) <= tol:
                new_matrix[:, :, new_idx] = matrix_3d[:, :, old_idx]
            # else leave as zeros (peak absent in this sample)
        reindexed.append(new_matrix)

    return reindexed, common_mz


def build_slide_mosaic(sample_matrices: list, n_pra: int,
                        matrix_zarr_path: str, common_mz: np.ndarray,
                        full_mz_axis: np.ndarray, gap: int = 10) -> tuple:
    """
    Arrange samples in 2 rows (pra top, 1hnr bottom), with matrix block top-left.
    Returns combined 3D mosaic and a dict of sample offsets for provenance.
    """
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


def run_preprocessing(params, run_folder):
    """Slide-level preprocessing: preprocess each sample, harmonise, build mosaic."""

    sample_zarr_paths = params.get("sample_zarr_paths")

    # if running in old single-sample mode, fall back gracefully
    if not sample_zarr_paths:
        sample_zarr_paths = [params["zarr_path"]]

    print(f"[preprocessing] Processing {len(sample_zarr_paths)} samples for slide mosaic...")

    sample_matrices = []
    sample_mz_lists = []
    full_mz_axes    = []

    for i, zarr_path in enumerate(sample_zarr_paths):
        print(f"[preprocessing] Sample {i+1}/{len(sample_zarr_paths)}: {os.path.basename(zarr_path)}")
        matrix_3d, filtered_mz, full_mz = preprocess_single_sample(zarr_path, params, run_folder)
        sample_matrices.append(matrix_3d)
        sample_mz_lists.append(filtered_mz)
        full_mz_axes.append(full_mz)

    # harmonise to common m/z axis
    reindexed_matrices, common_mz = harmonise_mz_axes(
        sample_matrices, sample_mz_lists, full_mz_axes, tol=0.01
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

    np.save(f"{run_folder}/sample_offset.npy", np.array([sample_offset]))
    np.save(f"{run_folder}/matrix.npy", mosaic)

    print(f"[preprocessing] Mosaic saved. Shape: {mosaic.shape}")

    return {
        "matrix_path": f"{run_folder}/matrix.npy",
        "n_features":  mosaic.shape[-1]
    }

def median_filter_spectrum(intensity, kernel_size=5):
    """
    Apply median filter to smooth spectrum before peak detection.

    kernel_size must be odd (e.g., 3, 5, 7).
    """
    print(f"Applying median filter (kernel size = {kernel_size})...")
    
    filtered = medfilt(intensity, kernel_size=kernel_size)

    
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




def peak_detection_omp(mz, 
        avg_intensity,
        run_folder, 
        window_size=0.07, 
        non_zero_coefs=700):
    print("detecting peaks using OMP...")

    # use mad to get candidate peaks for OMP
    median = np.median(avg_intensity)
    mad_noise = np.median(np.abs(avg_intensity - median))
    threshold = 2 * mad_noise # 2 is the snr 

    candidate_peak_idxs = []
    window = 20 # index-based window for local max check
    for i in range(window, len(avg_intensity) - window):
        local_window = avg_intensity[i-window:i+window+1]
        if avg_intensity[i] == np.max(local_window) and avg_intensity[i] > threshold:
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
    threshold = 0.00005 * pooled_spectra.shape[0] # set threshold as 0.005% of total number of spectra
    # threshold = 50

    valid_peaks = np.array(presence).flatten() > threshold # keep peaks that are present in more than 0.5% of spectra
    filtered_spectra = pooled_spectra[:, valid_peaks] # filter the spectra to keep only valid peaks

    filtered_spectra = pooled_spectra[:, valid_peaks]
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
    tic = filtered_spectra.sum(axis=1) # total ion current for each spectrum / for each pixel
    tic = np.where(tic == 0, 1, tic) # avoid division by zero
    normalised_matrix = (filtered_spectra / tic) * target # divide each spectrum by its TIC to normalize for differences in total intensity between spectra
    
    # TIC should be aprox 1.0 for all non-zero pixels
    non_zero_tics = normalised_matrix.sum(axis=1)
    non_zero_tics = non_zero_tics[non_zero_tics > 0]
    np.save(f"{run_folder}\\normalised_matrix.npy", normalised_matrix)
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

    width = len(np.unique(x_coords))
    height = len(np.unique(y_coords))
    print(f"Reshaping to 3D matrix with dimensions: ({height}, {width}, {filtered_spectra.shape[1]})")

    if issparse(filtered_spectra):
        spectra_array = filtered_spectra.toarray()
    else:
        spectra_array = filtered_spectra
    matrix = spectra_array.reshape(height, width, -1)
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

def run_preprocessing(params, run_folder):
    if params.get("batch_mode", False):
        print("Running batch mode")
        sample_zarr_paths = params["sample_zarr_paths"]
        
        sample_matrices = []
        sample_mz_lists = []
        full_mz_axes = []
        for i, zarr_path in enumerate(sample_zarr_paths):
            matrix_3d, filtered_mz, full_mz = preprocess_single_sample(
                zarr_path, params, run_folder
            )
            sample_matrices.append(matrix_3d)
            sample_mz_lists.append(filtered_mz)
            full_mz_axes.append(full_mz)

        # harmonise to common mz axis 
        reindexed_matrices, common_mz = harmonise_mz_axes(
            sample_matrices, 
            sample_mz_lists,
            full_mz_axes, 
            tol=0.01
        )

        pd.DataFrame({"mz": common_mz}).to_csv(
            f"{run_folder}/filtered_mz_values.csv", index=False
        )

        # buil mosaic 
        mosaic, sample_offset = build_slide_mosaic(
            reindexed_matrices,
            n_pra = params.get("n_pra", len(reindexed_matrices) // 2),
            matrix_zarr_path = params.get("matrix_zarr_path"),
            common_mz=common_mz,
            full_mz_axis=full_mz_axes[0],
            gap=10
        )

        np.save(f"{run_folder}/sample_offset.npy", np.array([sample_offset]))
        np.save(f"{run_folder}/matrix.npy", mosaic)
        print(f"Mosaic shape: {mosaic.shape}")

        return {
            "matrix_path": f"{run_folder}/matrix.npy",
            "n_features": mosaic.shape[-1]
        }

    else:
        spatial_data = reading_data(params["zarr_path"])
        AnnData, mz, filtered_avg_intensity, _ = compute_average_spectrum(spatial_data)


        if params["filtering"] == "median":
            filtered_avg_intensity = median_filter_spectrum(
                filtered_avg_intensity,
                kernel_size=5  # tune this!
            )

            print("filtering...")
                #  debug plot
            # plt.figure(figsize=(12, 4))
            # plt.plot(mz, filtered_avg_intensity, label="Original", alpha=0.5)
            # plt.plot(mz, filtered_avg_intensity, label="Median filtered", linewidth=2)
            # plt.legend()
            # plt.title("Median filtering effect")
            # plt.savefig(f"{run_folder}/median_filter_debug.png", dpi=150)
            # plt.close()
        else:
            pass

        if params["peak_method"] == "OMP":
            peak_mz, peak_intensities = peak_detection_omp(
                mz, filtered_avg_intensity , run_folder, non_zero_coefs=params["omp_coefs"]
            )

        else:
            peak_mz, peak_intensities = peak_detection_mad(
            mz, 
            filtered_avg_intensity, 
            window_size=20, 
            snr=2
            )

        bins = peak_binning(peak_mz, run_folder, tolerance=params["bin_tol"])
        pooled_spectra, pooled_mz = pooling(AnnData.X, mz, bins)
        _, filtered_spectra, filtered_mz = filtering(pooled_spectra, pooled_mz, run_folder)

        normalized_matrix = tic_normalization(filtered_spectra, run_folder)

        matrix = reshaping_to_3d_matrix(AnnData, normalized_matrix)

        matrix_zarr_path = params.get("matrix_zarr_path")
        print(f"[preprocessing] about to call stack_matrix_spatially with: {matrix_zarr_path}")
        if matrix_zarr_path:
            matrix, sample_offset = stack_matrix_spatially(matrix, 
            matrix_zarr_path, 
            filtered_mz,
            mz_axis=mz,
            gap=5)
            np.save(f"{run_folder}/sample_offset.npy", np.array([sample_offset]))
        else:
            print("[preprocessing] WARNING: matrix_zarr_path is None, skipping stacking")

        np.save(f"{run_folder}/matrix.npy", matrix) 
        return { 
            "matrix_path": f"{run_folder}/matrix.npy", 
            "n_features": matrix.shape[-1]
            }
