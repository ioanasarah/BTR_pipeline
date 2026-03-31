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
matrix_keyword = "matrix"


def is_matrix_zarr(name:str) -> bool:
    return any(matrix_keyword in name.lower())




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

        zarr_files = [
            f for f in os.listdir(slide_path)
            if f.endswith(".zarr") and os.path.isdir(os.path.join(slide_path, f))
        ]

        # separate matrix from sample files 
        matrix_zarrs = [os.path.join(slide_path, f) for f in zarr_files if is_matrix_zarr(f)]
        sample_zarrs = [f for f in zarr_files if not is_matrix_zarr(f)]

        for zarr_file in sorted(zarr_files):
            sample_name = zarr_file.replace(".zarr", "")
            zarr_path   = os.path.join(slide_path, zarr_file)

            # Build a param dict for this specific zarr
            params = {
                **base_params,
                "zarr_path":   zarr_path,
                "matrix_zarr_paths": matrix_zarrs,
                "dataset":     f"{matrix}_{sample_name}".replace(" ", "_"),
                "slide":       slide_folder,
                "matrix":      matrix,
                "sample_name": sample_name,
            }
            all_params.append(params)

    print(f"[batch_runner] Found {len(all_params)} zarr files across {len(slide_folders)} slides.")
    return all_params