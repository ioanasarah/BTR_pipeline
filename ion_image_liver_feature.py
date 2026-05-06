import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def reconstruct_and_plot_ion_images(
        matrix_scaled,  
        mask, 
        original_shape, 
        mz_values, 
        feature,
        # rf_dict, 
        run_folder
):
    height, width = original_shape
    # print(original_shape)
    # print(mask.shape)
    # print(mask.dtype)


    top_mz = feature
    col_idx = int(np.argmin(np.abs(mz_values - top_mz))) # finds closest value to the feature defined 

    ion_flat = np.full(height * width, np.nan)
    ion_flat[mask] = matrix_scaled[:, col_idx]
    # ion_flat[mask] = col_idx
    ion_image = ion_flat.reshape(height, width)

    plt.figure(figsize=(8, 6))
    plt.imshow(ion_image, cmap="hot", interpolation="nearest")
    plt.colorbar(label="intensity")
    plt.title(f"Ion image: m/z {top_mz:.3f}")
    plt.savefig(f"{run_folder}\\ion_image_{feature}.png", dpi=150)
    plt.show()


run_folder = r"C:\Users\i6338212\data\results\liver_mosaic_PC\OMP700_filtering0.005_spca10_hierarchical6_savgol_spectralfiltering\DHB_060326_DHB_Slide_11_50_um_OMP700_filtering0.005_spca10_hierarchical6_savgol_spectralfiltering"
reduction_name = "DHB_060326_DHB_Slide_11_50_um_OMP700_filtering0.005_spca10_hierarchical6_savgol_spectralfiltering"
matrix = np.load(f"{run_folder}\\matrix.npy")
# mask = np.load(f"{run_folder}\\mask.npy")
labels = pd.read_csv(f"{run_folder}\\spca_results.csv").iloc[:, -1]  # assuming last column is 'cluster'
spatial_map = np.load(f"{run_folder}\\spatial_map_matrix_{reduction_name}.npy")
matrix_scaled = np.load(f"{run_folder}\\matrix_scaled.npy")
mz_values = pd.read_csv(f"{run_folder}\\filtered_mz_values.csv")["mz"].values
mask = np.load(f"{run_folder}\\mask.npy")
original_shape = np.load(f"{run_folder}\\original_shape.npy")
name_of_run = "OMP700_filtering0.005_spca10_hierarchical6_savgol_spectralfiltering"


h, w, n_peaks = matrix.shape
matrix_flat = matrix.reshape(h * w, n_peaks)

feature = 377.13


reconstruct_and_plot_ion_images(
    matrix_scaled,
    mask,
    original_shape,
    mz_values, 
    feature,
    # rf_dict, 
    run_folder
)
# poetry run python ion_image_liver_feature.py