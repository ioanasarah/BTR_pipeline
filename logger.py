import os 
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"


import pandas as pd
import time
import numpy as np
from preprocessing import run_preprocessing
from dimensionality_red import run_dimensionality_reduction
from clustering_metrics import run_clustering_metrics
from feature_selection import run_feature_selection
# from feature_selection import perform_anova_test, perform_fdr_correction, volcano_plot_plotly, run_random_forest, reconstruct_and_plot_ion_images, combine_anova_rf

# batch_mode = False
# batch_mode = True
slide_filter = None # None to run all slides
#  DHB Slide 4 50 um


# results_folder = r"C:\Users\i6338212\data\results"
results_folder = r"C:\Ioana\_uni\BTR_pipeline_code\results"
# results_folder = r"C:\Users\i6338212\data\results"
results_csv = os.path.join(results_folder, "experiment_results.csv")

# batch_root = r"C:\Users\i6338212\data\spatialdata_zep"
# batch_root = r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\Slide1_zarr"

# reduction_name = "OMP_pca10_k3_no_smoothing"

# run_folder = os.path.join(results_folder, preprocessing_run_name, reduction_name)
# os.makedirs(run_folder, exist_ok=True)


single_params= {
    "batch_mode": False,
    "tissue": "hippocampus",
    "dataset": "hippocampus",
    "computer": "laptop",
    "experiment": "hippocampus_laptop",
    # "zarr_path": r"C:\Ioana\_uni\btr\zarr\MALDI-MSI Mouse Brain.zarr\MALDI-MSI Mouse Brain.zarr",
    "zarr_path": r"C:\Ioana\_uni\btr\zarr\hippocampus.zarr",
    # "zarr_path": r"C:\Users\i6338212\data\datasets\mosaic_hippocampus\zarr_files1\20260413_L6_C1409_DHB_30um_new.zarr",
    # "zarr_path": r"C:\Users\i6338212\data\Ioana Test Data\Data\hippocampus.zarr",
    # "zarr_path": r"C:\Users\i6338212\data\spatialdata_zep\060326 DHB Slide 11 50 um\1 1hnr.zarr",

    "smoothing": "8_connect",

    # "smoothing": None,
    "filtering": "guided", # None, "median", "savgol", "gaussian", "guided"
    "peak_method": "OMP",
    "normalisation": "TIC",
    "omp_coefs": 700,
    "bin_tol": 0.005,
    "matrix_ratio_threshold": None, 
    "matrix_zarr_path": None,

    "dimred": "pca", 
    "n_components": 10,

    "clustering": "kmeans",
    "n_clusters":4

    # "run_id": "OMP_pca10_k3_no_smoothing",
}

def generate_method_name(params):
    """Just the method combo — no dataset. Used as the grouping folder."""
    parts = [
        params["peak_method"],
        params["dimred"].lower() + str(params["n_components"]),
        params["clustering"].lower() + str(params["n_clusters"]),
    ]
    if params.get("smoothing"):
        parts.append("smoothing")

    if params.get("filtering"):
        parts.append(f"{params['filtering']}_filtering")
    
    return "_".join(parts)

def generate_run_name(params):
    """Full run ID including dataset — used as the leaf folder."""
    return f"{params['dataset']}_{generate_method_name(params)}"
# logging results
def log_experiment(results_csv, row_dict):
    df_new = pd.DataFrame([row_dict])

    if os.path.exists(results_csv):
        df_existing = pd.read_csv(results_csv)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(results_csv, index=False)



def run_pipeline(params: dict):
    params["run_id"] = generate_run_name(params)
    folder_name = f"{params['tissue']}_{params['computer']}"
    run_folder = os.path.join(
            results_folder,
            f"{params['tissue']}_{params['computer']}",
            generate_method_name(params),
            generate_run_name(params), 
            "newOMP"
        )
    os.makedirs(run_folder, exist_ok=True)

    print(f"Results from {params['run_id']} will be saved to {run_folder}")

    start_time = time.perf_counter()

    # PREPROCESSING + DIM REDUCTION
    preprocessing_output = run_preprocessing(params, run_folder)
    # preprocessing_output = {"n_features": "144"}
    dimensionality_red_output = run_dimensionality_reduction(
        # preprocessing_output["matrix_path"],
        f"{run_folder}/matrix.npy",
        # r"C:\Users\i6338212\data\msi_matrix_hippocampus_omp.npy",
        params, 
        run_folder)
    # run_feature_selection(params, run_folder)

    feature_selection_output = run_feature_selection(
        dimensionality_red_output=dimensionality_red_output,
        run_folder=run_folder,
        params=params,
    )

    metrics_output = run_clustering_metrics(
        dimensionality_red_output,
        run_folder,
        params
    )

    #updating results

    results_row = {
        **params,

        # preprocessing
        "n_features": preprocessing_output["n_features"],

        # clustering
        "n_clusters_found": dimensionality_red_output["n_clusters_found"],

        # metrics
        **metrics_output,

        # runtime
        "runtime_sec": time.perf_counter() - start_time
    }

    log_experiment(results_csv, results_row)



if __name__ == "__main__":
    if single_params["batch_mode"]:
        print("Running batch mode!")
        from batch_runner import collect_batch_params
        print("Imported batch runner things ")
        all_params = collect_batch_params(batch_root, slide_filter, single_params)
        # print(f"Batch mode: {len(all_params)} runs found")
        for i, p in enumerate(all_params, 1):
            # print(f"-------------------------------- Starting sample {i}/{len(all_params)}: {p['sample_name']}-----------------------------------------------------------------")
            run_pipeline(p)
    else:
        run_pipeline(single_params)


if __name__ == "__main__":
    print("Full run completed and logged.")