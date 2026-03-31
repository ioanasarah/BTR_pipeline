import os
import re

# def parse_slide_metadata(slide_folder_name: str) -> dict:
#     """
#     Extract matrix and slide info from a slide folder name.
#     Example: '2024-01-15 DHB Slide 4 50 um'
#     Returns: {"matrix": "DHB", "slide_name": "2024-01-15 DHB Slide 4 50 um"}
#     """
#     # Look for known matrix names in the folder name
#     known_matrices = ["DHB", "DAN", "CHCA"]
#     matrix = "unknown"
#     for m in known_matrices:
#         if m.upper() in slide_folder_name.upper():
#             matrix = m
#             break

#     return {
#         "matrix":     matrix,
#         "slide_name": slide_folder_name,
#     }

matrix = "DHB"


def is_matrix_zarr(name:str) -> bool:
    return "matrix" in name.lower()


def is_pra_zarr(name:str) -> bool:
    return "pra" in name.lower()


def collect_batch_params(batch_root: str, slide_filter: str, base_params: dict) -> list[dict]:
    """
    Walk spatialdata_zep/slide/sample.zarr and return a list of
    param dicts — one per zarr file — ready to pass to run_pipeline().

    Skips anything that isn't a .zarr directory.
    """
    all_params = []

    slide_folders = [
        f for f in os.listdir(batch_root)
        if os.path.isdir(os.path.join(batch_root, f))
    ]

    # filter to one slide if specified
    if slide_filter:
        slide_folders = [f for f in slide_folders if slide_filter in f]
        if not slide_folders:
            raise ValueError(f"No slide folders matched filter: '{slide_filter}'")
        print(f"[batch_runner] Slide filter '{slide_filter}' matched: {slide_folders}")


    if not os.path.isdir(batch_root):
        raise FileNotFoundError(f"Batch root not found: {batch_root}")

    # slide_folders = [
    #     f for f in os.listdir(batch_root)
    #     if os.path.isdir(os.path.join(batch_root, f))
    # ]

    for slide_folder in sorted(slide_folders):
        slide_path = os.path.join(batch_root, slide_folder)
        # slide_meta = parse_slide_metadata(slide_folder)

        zarr_files = sorted([
            f for f in os.listdir(slide_path)
            if f.endswith(".zarr") and os.path.isdir(os.path.join(slide_path, f))
        ])

        # separate matrix from sample files 
        matrix_zarrs = [os.path.join(slide_path, f) for f in zarr_files if is_matrix_zarr(f)]
        sample_zarrs = [f for f in zarr_files if not is_matrix_zarr(f)]


        # split samples into pra and 1hnr
        pra_zarrs = sorted([f for f in sample_zarrs if is_pra_zarr(f)])
        hnr_zarrs = sorted([f for f in sample_zarrs if not is_pra_zarr(f)])


        matrix_zarr_path = matrix_zarrs[0] if matrix_zarrs else None

        print(f"[batch_runner] Slide: {slide_folder}") 
        print(f" pra samples: {pra_zarrs}") 
        print(f" 1hnr samples: {hnr_zarrs}") 
        print(f" matrix zarr: {matrix_zarr_path}")




        # if matrix_zarr_path:
        #     print(f"[batch_runner] using matrix zarr: {matrix_zarrs[0]}")
        # else:
            # print(f"[batch_runner] No matrix zarr found for slide: {slide_folder}")

        # for zarr_file in sample_zarrs:
        #     sample_name = zarr_file.replace(".zarr", "")
        #     zarr_path   = os.path.join(slide_path, zarr_file)

            # Build a param dict for this specific zarr
            params = { **base_params, # list of full paths, pra first then 1hnr 
            "batch_mode" = True, 
            "sample_zarr_paths": [os.path.join(slide_path, f) for f in pra_zarrs] + [os.path.join(slide_path, f) for f in hnr_zarrs], 
            "sample_names": [f.replace(".zarr", "") for f in pra_zarrs] + [f.replace(".zarr", "") for f in hnr_zarrs], 
            "n_pra": len(pra_zarrs), # so preprocessing knows row split 
            "matrix_zarr_path": matrix_zarr_path, 
            "dataset": f"{matrix}_{slide_folder}".replace(" ", "_"), 
            "slide": slide_folder, 
            "matrix": matrix, 
            "sample_name": slide_folder, # used for run_id 
            }

            print(f"[batch_runner] matrix_zarr_path for {sample_name}: {matrix_zarr_path}")
            all_params.append(params)

    print(f"[batch_runner] Found {len(all_params)} zarr files across {len(slide_folders)} slides.")
    return all_params



if __name__ == "__main__":
    all_params = collect_batch_params(
        batch_root = r"C:\Users\i6338212\data\spatialdata_zep",
        slide_filter = "DHB Slide 11 50 um",
        base_params = {
    "tissue": "hippocampus",
    "dataset": "xenium",
    "computer": "PC",
    "experiment": "liver_PC",
    # "zarr_path": r"C:\Ioana\_uni\btr\zarr\MALDI-MSI Mouse Brain.zarr\MALDI-MSI Mouse Brain.zarr",
    "zarr_path": r"C:\Users\i6338212\data\Ioana Test Data\Data\hippocampus.zarr",
    # "zarr_path": r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\1 1hnr.zarr",

    "smoothing": "8_connectivity",
    # "smoothing": None,
    "filtering": None,
    "peak_method": "OMP",
    "normalisation": "TIC",
    "omp_coefs": 700,
    "bin_tol": 0.005,

    "dimred": "spca", 
    "n_components": 10,

    "clustering": "kmeans",
    "n_clusters": 4

    # "run_id": "OMP_pca10_k3_no_smoothing",
}
    )
    print(all_params[0]["matrix_zarr_paths"])