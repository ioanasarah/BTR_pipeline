import pandas as pd
import numpy as np
import time
import os
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import plotly.express as px


print("Loaded packages for feature selection")



def load_things(raw_matrix_file_path: str,
                file_path: str):
    # load matrix
    matrix_scaled = np.load(raw_matrix_file_path)
    print(f"Matrix loaded in {time.perf_counter() - start_time:.2f} seconds")

    # load pca 

    pca_df = pd.read_csv(file_path)
    labels = pca_df["cluster"]
    pca_transformed = pca_df.iloc[:, :-1].values
    print(f"PCA results loaded. Shape: {pca_transformed.shape}, Clusters: {labels.nunique()}")
    return matrix_scaled, pca_transformed, labels


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

    # print("ANOVA test complete. Took {:.2f} seconds".format(time.perf_counter() - start_time))

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
    # print("FDR correction complete. Took {:.2f} seconds".format(time.perf_counter() - start_time))
    return reject, pvals_corrected


def volcano_plot_plotly(matrix, 
                        labels, 
                        p_values,
                        run_folder, 
                        mz_values,
                        # run_folder: str,
                        name_of_run: str,
                        cluster_a: int = 0,
                        cluster_b: int = 1):
    # BUG FIX: run_folder and name_of_run are now explicit parameters.
    # Previously they referenced module-level variables that are not
    # defined when this function is called from run_dimensionality_reduction
    # or any other context -- a silent NameError waiting to happen.
    #
    # BUG FIX: cluster_a / cluster_b parameters replace the hard-coded 0/1.
    # A volcano plot compares exactly two groups (fold-change is only
    # meaningful between a pair).  With k>2 clusters the original code
    # silently ignored all clusters except 0 and 1.  Callers should now
    # explicitly choose which pair to compare, or loop over all pairs.
    cluster0 = matrix[labels == cluster_a]
    cluster1 = matrix[labels == cluster_b]

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

def run_random_forest(
                    matrix: np.ndarray,
                    labels: pd.Series,
                    mz_values: np.ndarray,
                    run_folder: str,
                    name_of_run: str,
                    n_estimators: int = 100,
                    n_jobs: int = -1,
                    top_n_features: int = 30,
                    cross_validate: bool = True,
                    ) -> dict:
    print("Starting random forest feature selection...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",   
        max_samples = 0.1,   
        class_weight="balanced",  # handles unequal cluster sizes
        random_state=42,
        n_jobs=n_jobs,
    )
    rf.fit(matrix, labels)
    print(f"Random Forest trained ({n_estimators} trees).")

    # extracting and ranking feature importance scores
    importances = rf.feature_importances_
    # print(len(importances))
    # print(len(mz_values))

    # finding standard dev across trees 

    # std_list =[]
    # for tree in rf.estimators_:
    #     std = np.std(tree.feature_importances_, axis=0)
    #     std_list.append(std)
    # print(len(std_list))

    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    # print(len(std)) 

    importances_df = pd.DataFrame({
        "mz": mz_values,
        "importance": importances,
        "importance_std": std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(importances_df.head())


# add a column rank and rank the features in order of importance 
    importances_df["rank"] = importances_df.index + 1
   
#    save importance dataframe 
    save_path = f"{run_folder}\\rf_feature_importances_{name_of_run}.csv"
    top_features = importances_df["mz"].values[:top_n_features]

    importances_df.to_csv(save_path, index=False)
    print(f"Feature importances saved to {save_path}")

    # 5 fold cross validation??

    # plotting 
    top_df = importances_df.head(top_n_features)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        top_df["mz"].astype(str),
        top_df["importance"],
        xerr=top_df["importance_std"],
        align="center",
        color="steelblue",
        ecolor="gray",
        capsize=3,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean Decrease in Importance")
    ax.set_title(f"Top {top_n_features} m/z Features — {name_of_run}")
    plt.tight_layout()
    plot_path = f"{run_folder}\\rf_top_features_{name_of_run}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Feature importance plot saved to {plot_path}")


    # print("Random forest classification complete. Took {:.2f} seconds".format(time.perf_counter() - start_time))

    return {
        "importances_df": importances_df,
        "top_features": top_features,
        "top_feature": top_features[0]
        # "cv_scores": cv_scores,
        # "rf_model": rf,
    }

def combine_anova_rf(
        anova_results_df: pd.DataFrame,
        rf_importances_df: pd.DataFrame,
        top_n_rf: int,
        run_folder: str,
        name_of_run: str
): 
    sig_anova = anova_results_df[anova_results_df["significant_after_fdr"] == True]["mz"]
    top_rf = rf_importances_df.head(top_n_rf)["mz"]

    # which features are both stat sign and high importance 
    intersect = set(sig_anova).intersection(set(top_rf))
    print(f"\nConsensus features (ANOVA sig + top-{top_n_rf} RF): {len(intersect)}")

    merged = rf_importances_df[rf_importances_df["mz"].isin(intersect)].copy()
    merged = merged.merge(
        anova_results_df[["mz", "adjusted_p_value"]],
        on="mz", how="left"
    ).sort_values("importance", ascending=False)

    save_path = f"{run_folder}\\consensus_features_{name_of_run}.csv"
    merged.to_csv(save_path, index=False)
    print(f"Consensus features saved to {save_path}")

    return merged


def reconstruct_and_plot_ion_images(
        matrix_scaled,  
        mask, 
        original_shape, 
        mz_values, 
        feature,
        rf_dict, 
        run_folder
):
    height, width = original_shape
    # print(original_shape)
    # print(mask.shape)
    # print(mask.dtype)


    top_mz = feature
    col_idx = int(np.argmin(np.abs(mz_values - top_mz))) # finds closest value to the feature defined 

    ion_flat = np.full(height * width, np.nan)
    ion_flat[mask] = matrix_scaled[:, col_idx]
    # ion_flat[mask] = col_idx
    ion_image = ion_flat.reshape(height, width)

    plt.figure(figsize=(8, 6))
    plt.imshow(ion_image, cmap="hot", interpolation="nearest")
    plt.colorbar(label="intensity")
    plt.title(f"Ion image: m/z {top_mz:.3f}")
    plt.savefig(f"{run_folder}\\ion_image_{feature}.png", dpi=150)
    plt.show()



if __name__ == "__main__":
    start_time = time.perf_counter()
    results_folder = r"C:\Ioana\_uni\BTR_pipeline_code\results" # change folder path as needed
    preprocessing_run_name = "xenium_laptop"
    # reduction_name = "xenium_OMP_pca_umap10_k5_smoothing" # good segm but bg weird
    reduction_name = "xenium_OMP_pca10_k4_3x3_smoothing" # good segm 
    # reduction_name = "xenium_OMP_pca10_k4" # bad segm
    run_folder = os.path.join(results_folder, preprocessing_run_name, reduction_name)
    mz_values = pd.read_csv(f"{run_folder}\\filtered_mz_values.csv")["mz"].values


    os.makedirs(run_folder, exist_ok=True)
    matrix_scaled, pca_transformed, labels = load_things(
        raw_matrix_file_path=f"{run_folder}\\matrix_scaled.npy",
        file_path=f"{run_folder}\\pca_results.csv")
    
    p_values = perform_anova_test(
    matrix = matrix_scaled, 
        # f"{run_folder}\\{params['dimred']}_results.csv"
    labels = labels)
    reject, pvals_corrected = perform_fdr_correction(
        p_values=p_values
    )

    anova_results_df = pd.DataFrame({
            "mz": pd.read_csv(f"{run_folder}\\filtered_mz_values.csv")["mz"].values,
            "p_value": p_values,
            "adjusted_p_value": pvals_corrected,
            "significant_after_fdr": reject
        })
        
    rf_dict = run_random_forest(matrix_scaled, 
                                labels,
                                mz_values, 
                                run_folder, 
                                reduction_name)
    
    mask = np.load(r"C:\Ioana\_uni\BTR_pipeline_code\results\xenium_laptop\xenium_OMP_pca10_k4_3x3_smoothing\mask.npy")
    for cluster_id in labels.unique():
        mask = labels == cluster_id
        cluster_means = matrix_scaled[mask].mean(axis=0)
        overall_means = matrix_scaled.mean(axis=0)
        enrichment = cluster_means / (overall_means + 1e-9)
        top_idx = np.argsort(enrichment)[::-1][:5]
        print(f"Cluster {cluster_id} top enriched m/z: {mz_values[top_idx]}")


    mask = np.load(r"C:\Ioana\_uni\BTR_pipeline_code\results\xenium_laptop\xenium_OMP_pca10_k4_3x3_smoothing\mask.npy")
    mask = np.squeeze(np.asarray(mask))
    # print(mask.shape)
    # print(mask.ndim)

    original_shape = tuple(np.load(f"{run_folder}\\original_shape.npy"))

    
    reconstruct_and_plot_ion_images(matrix_scaled=matrix_scaled, 
                                    mask=mask, 
                                    original_shape=original_shape,
                                    mz_values=mz_values,
                                    feature = 534.22335607,
                                    rf_dict=rf_dict,
                                    run_folder=run_folder)
  

    consensus_df =  combine_anova_rf(
    anova_results_df=anova_results_df,
    rf_importances_df=rf_dict["importances_df"],
    top_n_rf=100,
    run_folder=run_folder,
    name_of_run=reduction_name,
    )

    # p_values = perform_anova_test(matrix=matrix_scaled, labels=labels)
   
#     reject, pvals_corrected = perform_fdr_correction(p_values)
#     save_path = f"{run_folder}\\anova_results.csv"
#     anova_results_df = pd.DataFrame({
#         "mz": pd.read_csv(r"C:\Ioana\_uni\BTR_pipeline_code\results\xenium_tic_omp\filtered_mz_values.csv")["mz"].values,
#         "p_value": p_values,
#         "adjusted_p_value": pvals_corrected,
#         "significant_after_fdr": reject
#     })
#     anova_results_df.to_csv(save_path, index=False)
#     print(f"ANOVA results saved to {save_path}")
    
#     mz_values = pd.read_csv(r"C:\Ioana\_uni\BTR_pipeline_code\results_segmentation_omp\filtered_mz_values.csv")["mz"].values
#     volcano_plot_plotly(matrix_scaled, labels, p_values, mz_values)

# we want to know how many features are significant after FDR correction, and what is the minimum adjusted p-value. This will give us an idea of how many features we can select for downstream analysis.

def run_feature_selection(params: dict, 
                          run_folder: str):
    matrix_scaled, embedding, labels = load_things(
        raw_matrix_file_path=f"{run_folder}\\matrix.npy",
        pca_file_path=f"{run_folder}\\pca_results.csv")
    p_values = perform_anova_test(matrix=matrix_scaled, labels=labels)
   
    reject, pvals_corrected = perform_fdr_correction(p_values)
    save_path = f"{run_folder}\\anova_results.csv"
    anova_results_df = pd.DataFrame({
        "mz": pd.read_csv(f"{run_folder}\\filtered_mz_values.csv")["mz"].values,
        "p_value": p_values,
        "adjusted_p_value": pvals_corrected,
        "significant_after_fdr": reject
    })
    anova_results_df.to_csv(save_path, index=False)
    print(f"ANOVA results saved to {save_path}")
    
    mz_values = pd.read_csv(f"{run_folder}\\filtered_mz_values.csv")["mz"].values
    volcano_plot_plotly(matrix_scaled, labels, p_values, mz_values)
