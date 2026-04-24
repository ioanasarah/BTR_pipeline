import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def reconstruct_and_plot_ion_images(
        matrix_scaled,  
        mask, 
        original_shape, 
        mz_values, 
        feature,
        rf_dict, 
        run_folder):
    

    height, width = original_shape

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

# poetry run python plot_ion_images.py
if __name__ == "__main__":
    feature = 639.56
    # results_folder = r"C:\Ioana\_uni\BTR_pipeline_code\results\hippocampus_laptop\OMP_pca10_hierarchical4_label_matrix_smoothing\hippocampus_OMP_pca10_hierarchical4_label_matrix_smoothing"
    results_folder = r"C:\Ioana\_uni\BTR_pipeline_code\results\mosaic_hippocampus_theos_comp\OMP_pca10_kmeans4\hippocampus_OMP_pca10_kmeans4"
    original_shape = np.load(f"{results_folder}\\original_shape.npy")
    matrix_scaled = np.load(f"{results_folder}\\matrix_scaled.npy")
    mask = np.load(f"{results_folder}\\mask.npy")
    mz_values = pd.read_csv(f"{results_folder}\\filtered_mz_values.csv")["mz"].values

    reconstruct_and_plot_ion_images(
        matrix_scaled=matrix_scaled,
        mask=mask,
        original_shape=original_shape, 
        mz_values=mz_values,
        feature=feature,
        rf_dict=None,
        run_folder=results_folder)


