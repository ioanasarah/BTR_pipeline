# import sys 
# print(sys.executable)


import thyra
print(thyra.__version__)
import spatialdata as sd
print(sd.__version__)

from thyra import convert_msi
import inspect
import zarr 





# success = convert_msi(
#     r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1\20260413_L6_C1409_DHB_30um.d",
#     r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_new\20260413_L6_C1409_DHB_30um_resample.zarr",
#     dataset_id="hippocampus_mosaic",
#     reader_options={
#         "use_calibrated_state": True,
#         "intensity_threshold": 500.0,
#     },
# )
# #  poetry run python convert_files.py


import zarr
import anndata as ad

# Open the zarr store directly (bypassing spatialdata's OME check)
store = zarr.open(
    r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_new\20260413_L6_C1409_DHB_30um_resample.zarr",
    mode="r"
)

# Read the table directly as AnnData
adata = ad.read_zarr(
    r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_new\20260413_L6_C1409_DHB_30um_resample.zarr\tables\hippocampus_mosaic_z0"
)

print(adata)
print(adata.var["mz"].values[:10])
print(adata.obs.columns.tolist())

# try these in order to find docs
# help(convert_msi)

# # also check if there's a resampling config class
# try:
#     from thyra import ResamplingConfig
#     help(ResamplingConfig)
# except ImportError:
#     print("no ResamplingConfig class")

# try:
#     from thyra.config import ResamplingConfig
#     help(ResamplingConfig)
# except ImportError:
#     print("not in thyra.config either")

# # check what's available in the thyra module
# import thyra
# print(dir(thyra))
