import os
import time
import numpy as np
import spatialdata as sd
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
from scipy.sparse import issparse
# import msiwarp as mw

print("Loaded packages! Starting preprocessing...")

# results_folder = r"C:\Ioana\_uni\BTR_pipeline_code\results" # change folder path as needed
results_folder = r"C:\Users\i6338212\data\results"
preprocessing_run_name = "small_computer_xenium_omp"
run_folder = os.path.join(results_folder, preprocessing_run_name)
os.makedirs(run_folder, exist_ok=True)

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



def check_mass_drift(data, n_pixels=1000):
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

def linear_recalibration(data, reference_mz, reference_intensity, 
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
        print(pixel.max())
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
        pooled_mz
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


def tic_normalization(filtered_spectra: np.ndarray, 
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

    matrix = filtered_spectra.toarray().reshape(height, width, -1)
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
    spatial_data = reading_data(params["zarr_path"])
    AnnData, mz, avg_intensity, _ = compute_average_spectrum(spatial_data)

    peak_mz, peak_intensities = peak_detection_omp(
        mz, avg_intensity, non_zero_coefs=params["omp_coefs"]
    )

    bins = peak_binning(peak_mz, tolerance=params["bin_tol"])
    pooled_spectra, pooled_mz = pooling(AnnData.X, mz, bins)
    _, filtered_spectra, filtered_mz = filtering(pooled_spectra, pooled_mz)

    normalized_matrix = tic_normalization(filtered_spectra)

    matrix = reshaping_to_3d_matrix(AnnData, normalized_matrix)

    np.save(f"{run_folder}/matrix.npy", matrix)

    return {
        "matrix_path": f"{run_folder}/matrix.npy",
        "n_features": matrix.shape[-1]
    }
