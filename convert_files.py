# import sys 
# print(sys.executable)


import thyra
import spatialdata as sd
print(sd.__version__)

from thyra import convert_msi
import zarr 


# sdata = thyra.read_bruker(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1\20260413_L6_C1409_DHB_30um.d")
# sdata.write("20260413_L6_C1409_DHB_30um.zarr")
# print("successfully convered!")

# z = zarr.open(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_zarr\20260413_L6_C1409_DHB_30um.zarr")
# print(dict(z.attrs))


# success = convert_msi(
#     r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1\20260413_L6_C1409_DHB_30um.d",
#     r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_zarr\20260413_L6_C1409_DHB_30um_new.zarr",
#     dataset_id="hippocampus_mosaic",
# ) 

# # sample 1
# sdata = thyra.convert_msi(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1\20260413_L6_C1409_DHB_30um.d")
# print(thype(sdata))

# sdata.write(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_zarr\20260413_L6_C1409_DHB_30um_new.zarr")

# z = zarr.open(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_zarr\20260413_L6_C1409_DHB_30um_new.zarr")
# print(dict(z.attrs))

# # sample 2
# sdata1 = thyra.convert_msi(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1\20260413_L6_C1411_DHB_30um.d")
# print(thype(sdata1))

# sdata1.write(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_zarr\20260413_L6_C1411_DHB_30um_new.zarr")

# z1 = zarr.open(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_zarr\20260413_L6_C1411_DHB_30um_new.zarr")
# print(dict(z1.attrs))

# # sample 3
# sdata2 = thyra.convert_msi(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1\20260413_L6_C2601_DHB_30um.d")
# print(thype(sdata2))

# sdata2.write(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_zarr\20260413_L6_C2601_DHB_30um_new.zarr")

# z2 = zarr.open(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_zarr\20260413_L6_C2601_DHB_30um_new.zarr")
# print(dict(z2.attrs))

# success1 = convert_msi(
#     r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1\20260413_L6_C1411_DHB_30um.d",
#     r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1\20260413_L6_C1411_DHB_30um.zarr",
#     dataset_id="hippocampus_mosaic",
# )

# success2 = convert_msi(  
#     r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1\20260413_L6_C2601_DHB_30um.d",
#     r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1\20260413_L6_C2601_DHB_30um.zarr",
#     dataset_id="hippocampus_mosaic",
# )

import os 
# print(os.listdir(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_zarr\20260413_L6_C1411_DHB_30um.zarr"))
# print(os.listdir(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1\20260413_L6_C1409_DHB_30um.d"))

import json
import anndata as ad
# with open(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_zarr\20260413_L6_C1409_DHB_30um.zarr\zarr.json") as f:
#     print(json.dumps(json.load(f), indent=2))

for folder in ['images', 'shapes', 'tables']:
    path = r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_zarr\20260413_L6_C1409_DHB_30um.zarr" + "\\" + folder
    print(folder, os.listdir(path))


# sdata = sd.read_zarr(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_zarr\20260413_L6_C1409_DHB_30um.zarr")
# print("ok")
# print(list(sdata.tables.keys()))
# print(list(sdata.images.keys()))

store = zarr.open(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_zarr\20260413_L6_C1409_DHB_30um.zarr", mode="r")
adata = ad.read_zarr(r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_zarr\20260413_L6_C1409_DHB_30um.zarr\tables\hippocampus_mosaic_z0")


print(adata)
print(adata.var["mz"].values[:10])
print(adata.obs.columns.tolist())
#  poetry run python convert_files.py    