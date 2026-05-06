import matplotlib.pyplot as plt
import numpy as np
import spatialdata as sd

adata = list(sd.read_zarr(r"C:\Ioana\_uni\btr\zarr\MALDI-MSI_Mouse_Brain.zarr\MALDI-MSI Mouse Brain.zarr").tables.values())[0]
results_folder = r"C:\Ioana\_uni\BTR_pipeline_code\results\xenium_laptop"
original_shape = (adata.obs["y"].max() + 1, adata.obs["x"].max() + 1)
height, width = original_shape
tic = np.array(adata.X.sum(axis=1)).flatten()
tic_image = np.zeros(height * width)
x = adata.obs["x"].astype(int).values
y = adata.obs["y"].astype(int).values
tic_image[y * width + x] = tic
plt.imshow(tic_image.reshape(height, width), cmap="hot")
plt.colorbar()
plt.title("TIC image")
plt.savefig(f"{results_folder}/tic_image.png", dpi=150)

# import anndata as ad
# import os

# tables_path = r"C:\Ioana\_uni\btr\zarr\20260413_L6_C1409_DHB_30um_resample.zarr\tables"
# table_name = [f for f in os.listdir(tables_path) if f != "zarr.json"][0]
# adata = ad.read_zarr(os.path.join(tables_path, table_name))

# print("x range:", adata.obs["x"].min(), "–", adata.obs["x"].max())
# print("y range:", adata.obs["y"].min(), "–", adata.obs["y"].max())
# print("spatial_x range:", adata.obs["spatial_x"].min(), "–", adata.obs["spatial_x"].max())
# print("spatial_y range:", adata.obs["spatial_y"].min(), "–", adata.obs["spatial_y"].max())

# import matplotlib.pyplot as plt
# import numpy as np

# tic = np.array(adata.X.sum(axis=1)).flatten()

# # swap x and y here
# x = adata.obs["y"].astype(int).values  # swapped
# y = adata.obs["x"].astype(int).values  # swapped

# height = y.max() + 1
# width = x.max() + 1
# tic_image = np.zeros(height * width)
# tic_image[y * width + x] = tic
# plt.imshow(tic_image.reshape(height, width), cmap="hot")
# plt.colorbar()
# plt.title("TIC image - swapped axes")
# plt.savefig("tic_swapped.png", dpi=150)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# matrix_scaled = np.load(r"C:\Ioana\_uni\BTR_pipeline_code\results\mosaic_hippocampus_laptop\OMP_pca10_kmeans4_smoothing_savgol_filtering\mosaic_hippocampus_OMP_pca10_kmeans4_smoothing_savgol_filtering\matrix_scaled.npy")
# mask = np.load(r"C:\Ioana\_uni\BTR_pipeline_code\results\mosaic_hippocampus_laptop\OMP_pca10_kmeans4_smoothing_savgol_filtering\mosaic_hippocampus_OMP_pca10_kmeans4_smoothing_savgol_filtering\mask.npy")
# original_shape = np.load(r"C:\Ioana\_uni\BTR_pipeline_code\results\mosaic_hippocampus_laptop\OMP_pca10_kmeans4_smoothing_savgol_filtering\mosaic_hippocampus_OMP_pca10_kmeans4_smoothing_savgol_filtering\original_shape.npy")
# pca_results = pd.read_csv(r"C:\Ioana\_uni\BTR_pipeline_code\results\mosaic_hippocampus_laptop\OMP_pca10_kmeans4_smoothing_savgol_filtering\mosaic_hippocampus_OMP_pca10_kmeans4_smoothing_savgol_filtering\pca_results.csv")  

# import pandas as pd
# pca_df = pd.read_csv(r"C:\Ioana\_uni\BTR_pipeline_code\results\mosaic_hippocampus_laptop\OMP_pca10_kmeans4_smoothing_savgol_filtering\mosaic_hippocampus_OMP_pca10_kmeans4_smoothing_savgol_filtering\pca_results.csv")
# embedding = pca_df.iloc[:, :-1].values  # all cols except cluster

# height, width = original_shape
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# for i in range(3):
#     img = np.full(height * width, np.nan)
#     img[mask.flatten()] = embedding[:, i]
#     axes[i].imshow(img.reshape(height, width), cmap="RdBu_r")
#     axes[i].set_title(f"PC{i+1}")
# plt.savefig("pca_components.png", dpi=150)