import os 
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"


print("here1")
import pandas as pd
import time
from preprocessing import run_preprocessing
from dimensionality_red import run_dimensionality_reduction
from clustering_metrics import run_clustering_metrics

# results_folder = r"C:\Users\i6338212\data\results"
results_folder = r"C:\Ioana\_uni\BTR_pipeline_code\results"
results_csv = os.path.join(results_folder, "experiment_results.csv")


# reduction_name = "OMP_pca10_k3_no_smoothing"

# run_folder = os.path.join(results_folder, preprocessing_run_name, reduction_name)
# os.makedirs(run_folder, exist_ok=True)

params = {
    "dataset": "xenium",
    "computer": "laptop",
    "zarr_path": r"C:\Ioana\_uni\btr\zarr\MALDI-MSI Mouse Brain.zarr\MALDI-MSI Mouse Brain.zarr",
    # "zarr_path": r"C:\Users\i6338212\data\Ioana Test Data\Data\hippocampus.zarr",

    "smoothing": "3x3_just_mask",
    "peak_method": "OMP",
    "normalisation": "TIC",
    "omp_coefs": 700,
    "bin_tol": 0.005,

    "dimred": "pca", 
    "n_components": 10,

    "clustering": "kmeans",
    "n_clusters": 4

    # "run_id": "OMP_pca10_k3_no_smoothing",
}

def generate_run_name(params):
    parts = [
        params["dataset"],
        params["peak_method"],
        params["dimred"].lower() + str(params["n_components"]),
        "k" + str(params["n_clusters"]),
    ]

    if params["smoothing"]:
        # BUG FIX: use the actual smoothing value rather than the hardcoded string
        # "3x3_just_mask_smoothing".  The old code produced identical run folder names
        # for any non-falsy smoothing setting, so different smoothing configs would
        # silently overwrite each other's results.
        parts.append(f"{params['smoothing']}_smoothing")

    return "_".join(parts)


# running everything
params["run_id"] = generate_run_name(params)
folder_name = f"{params['dataset']}_{params['computer']}"
run_folder = os.path.join(
        results_folder,
        folder_name,
        params["run_id"]
    )
os.makedirs(run_folder, exist_ok=True)

print(f"Results from {params['run_id']} will be saved to {run_folder}")

start_time = time.perf_counter()
# preprocessing_output = run_preprocessing(params, run_folder)
preprocessing_output = {"n_features": "144"}
dimensionality_red_output = run_dimensionality_reduction(
    r"C:\Ioana\_uni\BTR_pipeline_code\msi_matrix_omp.npy",
    # r"C:\Users\i6338212\data\msi_matrix_hippocampus_omp.npy",
    params, 
    run_folder)
metrics_output = run_clustering_metrics(dimensionality_red_output, run_folder, params)

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

# logging results
def log_experiment(results_csv, row_dict):
    df_new = pd.DataFrame([row_dict])

    if os.path.exists(results_csv):
        df_existing = pd.read_csv(results_csv)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(results_csv, index=False)

log_experiment(results_csv, results_row)

print("Full run completed and logged.")