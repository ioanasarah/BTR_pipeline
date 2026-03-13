import pandas as pd
import numpy as np
import time
import os
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import plotly.express as px


results_folder = r"C:\Ioana\_uni\BTR_pipeline_code\results"
name_of_run = "k8_kmeans++_5x5_smoothing_omp"
run_folder = os.path.join(results_folder, name_of_run)
os.makedirs(run_folder, exist_ok=True)


start_time = time.perf_counter()

def load_things(raw_matrix_file_path: str,
                umap_file_path: str):
    # load matrix
    matrix_scaled = np.load(raw_matrix_file_path)
    print(f"Matrix loaded in {time.perf_counter() - start_time:.2f} seconds")

    # load umap 

    umap_transformed = pd.read_csv(umap_file_path)
    labels = umap_transformed["cluster"]


    return matrix_scaled, umap_transformed, labels


def perform_anova_test(matrix: np.ndarray, labels: pd.Series) -> pd.DataFrame:
    print("Performing ANOVA test for feature selection...")
    unique_labels = labels.unique()
    p_values_per_feature = []

    for i in range(matrix.shape[1]): # for each m/z value (feature)
        groups = []
        # groups = [matrix[labels == label, i] for label in unique_labels]
        for label in unique_labels:
            groups.append(matrix[labels == label, i]) #creates a boolean mask for current label (True if 1==1, False otherwise)

        f_stat, p_value = f_oneway(*groups)
        p_values_per_feature.append(p_value)
    p_values = np.array(p_values_per_feature)

    # how many features are stat significant at alpha = 0.05 before FDR correction
    print(f"Number of features with p-value < 0.05 before FDR correction: {np.sum(p_values < 0.05)}")
    print(f"Minimum p-value before FDR correction: {np.min(p_values)}")
    print(f"Median p-value before FDR correction: {np.median(p_values)}")

    print("ANOVA test complete. Took {:.2f} seconds".format(time.perf_counter() - start_time))

    return p_values


def perform_fdr_correction(p_values: np.ndarray):
  
    reject, pvals_corrected, _, _ = multipletests(
    p_values,
    method='fdr_bh'
    )

    # reject is a boolean array: true if feature is sinificant after FDR correction, false otherwise
    # pvals_corrected is an array of adjusted p-values after FDR correction = q values


    print(f"Minimum adjusted p-value: {np.min(pvals_corrected)}")
    print(f"Number of significant features after FDR correction: {np.sum(reject)}")
    print("FDR correction complete. Took {:.2f} seconds".format(time.perf_counter() - start_time))
    return reject, pvals_corrected


def volcano_plot_plotly(matrix, 
                        labels, 
                        p_values,
                        mz_values):
    
    
    cluster0 = matrix[labels == 0]
    cluster1 = matrix[labels == 1]

    mean0 = np.mean(cluster0, axis=0)
    mean1 = np.mean(cluster1, axis=0)

    log2_fc = np.log2((mean1 + 1e-9) / (mean0 + 1e-9))
    log2_fc = np.nan_to_num(log2_fc)

    # fix p-values
    p_values = np.clip(p_values, 1e-300, None)
    neg_log_p = -np.log10(p_values)

    significant = (p_values < 0.05) & (np.abs(log2_fc) > 1)

    df = pd.DataFrame({
        "m/z": mz_values,
        "log2FC": log2_fc,
        "-log10(p)": neg_log_p,
        "significant": significant
    })

    fig = px.scatter(
        df,
        x="log2FC",
        y="-log10(p)",
        color="significant",
        title="Volcano Plot",
        opacity=0.7, 
        hover_name="m/z", 
        hover_data=["log2FC", "-log10(p)"]
    )

    fig.add_hline(y=-np.log10(0.05), line_dash="dash")
    fig.add_vline(x=1, line_dash="dash")
    fig.add_vline(x=-1, line_dash="dash")

    fig.show()
    # fig.save(f"{results_folder}\\volcano_plot_k2_5x5_smoothing_raw_matrix.html")
    # save figure 
    fig.write_html(f"{run_folder}\\volcano_plot_{name_of_run}.html")
    print(f"Volcano plot saved to {run_folder}\\volcano_plot_{name_of_run}.html")


if __name__ == "__main__":
    matrix_scaled, umap_transformed, labels = load_things(
        raw_matrix_file_path=f"{run_folder}\\matrix_raw.npy",
        umap_file_path=f"{run_folder}\\umap_results.csv")

    p_values = perform_anova_test(matrix=matrix_scaled, labels=labels)
   
    reject, pvals_corrected = perform_fdr_correction(p_values)
    save_path = f"{run_folder}\\anova_results.csv"
    anova_results_df = pd.DataFrame({
        "mz": pd.read_csv(r"C:\Ioana\_uni\BTR_pipeline_code\results_segmentation_omp\filtered_mz_values.csv")["mz"].values,
        "p_value": p_values,
        "adjusted_p_value": pvals_corrected,
        "significant_after_fdr": reject
    })
    anova_results_df.to_csv(save_path, index=False)
    print(f"ANOVA results saved to {save_path}")
    
    mz_values = pd.read_csv(r"C:\Ioana\_uni\BTR_pipeline_code\results_segmentation_omp\filtered_mz_values.csv")["mz"].values
    volcano_plot_plotly(matrix_scaled, labels, p_values, mz_values)

# we want to know how many features are significant after FDR correction, and what is the minimum adjusted p-value. This will give us an idea of how many features we can select for downstream analysis.
