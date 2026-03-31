# # import spatialdata as sd
# # import numpy as np
# # import pandas as pd
# # import os 

# # liver = sd.read_zarr(r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\1 1hnr.zarr")
# # matrix = sd.read_zarr(r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\matrix 1.zarr")

# # # Check the AnnData key names
# # print("Liver keys:", list(liver.tables.keys()))
# # print("Matrix keys:", list(matrix.tables.keys()))

# # # Grab the first table from each
# # liver_adata = list(liver.tables.values())[0]
# # matrix_adata = list(matrix.tables.values())[0]

# # print("\nLiver AnnData shape:", liver_adata.shape)  # (n_pixels, n_mz)
# # print("Matrix AnnData shape:", matrix_adata.shape)

# # # Check m/z axes
# # liver_mz = liver_adata.var["mz"].values
# # matrix_mz = matrix_adata.var["mz"].values

# # print("\nLiver m/z range:", liver_mz.min(), "–", liver_mz.max(), "| n bins:", len(liver_mz))
# # print("Matrix m/z range:", matrix_mz.min(), "–", matrix_mz.max(), "| n bins:", len(matrix_mz))
# # print("Same m/z axis?", np.allclose(liver_mz, matrix_mz, atol=1e-4))

# import spatialdata as sd
# import numpy as np

# # change these to one sample + one matrix zarr from the same slide
# sample = sd.read_zarr(r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\1 1hnr.zarr")
# matrix = sd.read_zarr(r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\matrix 1.zarr")

# s = list(sample.tables.values())[0]
# m = list(matrix.tables.values())[0]

# print("Sample x range:", s.obs["x"].min(), "–", s.obs["x"].max())
# print("Sample y range:", s.obs["y"].min(), "–", s.obs["y"].max())
# print("Matrix x range:", m.obs["x"].min(), "–", m.obs["x"].max())
# print("Matrix y range:", m.obs["y"].min(), "–", m.obs["y"].max())
# print("Sample shape:", s.shape)
# print("Matrix shape:", m.shape

import os 
slide_path = r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um"

for f in sorted(os.listdir(slide_path)):
    print(f)