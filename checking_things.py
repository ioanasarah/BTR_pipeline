import spatialdata as sd
import numpy as np
import pandas as pd
import os 

liver = sd.read_zarr(r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\1 1hnr.zarr")
matrix = sd.read_zarr(r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\matrix 1.zarr")

# Check the AnnData key names
print("Liver keys:", list(liver.tables.keys()))
print("Matrix keys:", list(matrix.tables.keys()))

# Grab the first table from each
liver_adata = list(liver.tables.values())[0]
matrix_adata = list(matrix.tables.values())[0]

print("\nLiver AnnData shape:", liver_adata.shape)  # (n_pixels, n_mz)
print("Matrix AnnData shape:", matrix_adata.shape)

# Check m/z axes
liver_mz = liver_adata.var["mz"].values
matrix_mz = matrix_adata.var["mz"].values

print("\nLiver m/z range:", liver_mz.min(), "–", liver_mz.max(), "| n bins:", len(liver_mz))
print("Matrix m/z range:", matrix_mz.min(), "–", matrix_mz.max(), "| n bins:", len(matrix_mz))
print("Same m/z axis?", np.allclose(liver_mz, matrix_mz, atol=1e-4))
