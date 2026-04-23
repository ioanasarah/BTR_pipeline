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
    _t = time.perf_counter()
    # load matrix
    matrix_scaled = np.load(raw_matrix_file_path)
    print(f"Matrix loaded in {time.perf_counter() - _t:.2f} seconds")

    # load pca 

    pca_df = pd.read_csv(file_path)
    labels = pca_df["cluster"]
    pca_transformed = pca_df.iloc[:, :-1].values
    print(f"PCA results loaded. Shape: {pca_transformed.shape}, Clusters: {labels.nunique()}")
    return matrix_scaled, pca_transformed, labels

def remove_matrix_peaks(peak_mz: np.ndarray,
                        matrix_peaks: np.ndarray,
                        # ratio_threshold: float,
                        tol: float = 0.1) -> tuple:
    """
    Remove peaks from peak_mz that match flagged matrix peaks above ratio_threshold.
    Returns filtered peak_mz and a list of removed peaks.
    """
    # flagged_mz = matrix_peaks["mz"].values
    
    keep = []
    removed = []
    for mz in peak_mz:
        if np.any(np.abs(matrix_peaks - mz) <= tol):
            removed.append(mz)
        else:
            keep.append(mz)

    print(f"[remove_matrix_peaks] Removed {len(removed)} matrix peaks, "
          f"kept {len(keep)} peaks (threshold={tol}x)")
    return np.array(keep), np.array(removed)


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
    p_values = np.where(np.isnan(p_values), 1.0, p_values)

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

    # fig.show()
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
        # max_samples = 0.5,   
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
        # xerr=top_df["importance_std"],
        align="center",
        color="steelblue",
        # ecolor="gray",
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
    
    run_folder = r"C:\Ioana\_uni\BTR_pipeline_code\results\hippocampus_laptop\OMP_pca10_hierarchical4_label_matrix_smoothing\hippocampus_OMP_pca10_hierarchical4_label_matrix_smoothing"
    reduction_name = "hippocampus_OMP_pca10_hierarchical4_label_matrix_smoothing"
    matrix = np.load(f"{run_folder}\\matrix.npy")
    labels = pd.read_csv(f"{run_folder}\\pca_results.csv").iloc[:, -1]  # assuming last column is 'cluster'
    spatial_map = np.load(f"{run_folder}\\spatial_map_matrix_{reduction_name}.npy")
    matrix_scaled = np.load(f"{run_folder}\\matrix_scaled.npy")
    mz_values = pd.read_csv(f"{run_folder}\\filtered_mz_values.csv")["mz"].values
    mask = np.load(f"{run_folder}\\mask.npy")
    original_shape = np.load(f"{run_folder}\\original_shape.npy")
    name_of_run = "hippocampus_OMP_pca10_hierarchical4_label_matrix_smoothing"


    h, w, n_peaks = matrix.shape
    matrix_flat = matrix.reshape(h * w, n_peaks)

 
    should_remove_matrix_peaks = True

    if "should_remove_matrix_peaks" in locals() and should_remove_matrix_peaks:
        all_mz_values = pd.read_csv(f"{run_folder}\\filtered_mz_values.csv")["mz"].values
        matrix_peaks = pd.read_csv(f"{run_folder}\\top_peaks_matrix_cluster_2.csv")["m/z"].values
        # remove patrix peaks from mz peaks
        
        print("Removing matrix peaks in feature selection.")
        mz_values, _ = remove_matrix_peaks(
            peak_mz=mz_values,
            matrix_peaks=matrix_peaks,
            # ratio_threshold=0.5, 
            tol=0.1)
        

        keep_indices = [i for i, mz in enumerate(all_mz_values)
                    if any(np.abs(mz_values - mz) < 1e-6)]


        # flatten 3D matrix and apply mask before column filtering
        matrix_flat = matrix_flat[mask]                    # remove zero pixels
        matrix_flat = matrix_flat[:, keep_indices]         # remove matrix peak columns
        #  so removing both matrix peaks and pixels !!!

        print(f"Matrix shape after removing matrix peaks: {matrix_flat.shape}")
        print(f"mz_values length: {len(mz_values)}")
        assert matrix_flat.shape[1] == len(mz_values)

        pd.DataFrame({"mz": mz_values}).to_csv(
            f"{run_folder}\\filtered_mz_values_no_matrix_peaks.csv", index=False)
    
    
    # perform anova on mz values (but without matrix peaks)
    p_values = perform_anova_test(
    matrix = matrix_scaled, 
    # f"{run_folder}\\{params['dimred']}_results.csv"
    labels = labels)
    
    reject, pvals_corrected = perform_fdr_correction(
        p_values=p_values
    )

    #   ANOVA + FDR
    anova_results_df = pd.DataFrame({
            "mz": pd.read_csv(f"{run_folder}\\filtered_mz_values.csv")["mz"].values,
            # "mz": mz_values,
            "p_value": p_values,
            "adjusted_p_value": pvals_corrected,
            "significant_after_fdr": reject
        })
    print(anova_results_df.head())
    # anova_results_df.to_csv(f"{run_folder}\\anova_results_{name_of_run}.csv", index=False)
    # volcano_plot_plotly(
    #     matrix=matrix_flat,
    #     labels=labels,
    #     p_values=pvals_corrected, 
    #     run_folder=run_folder,
    #     mz_values=mz_values,
    #     name_of_run="hippocampus_hierachica4_attempt",
    # )

    # RANDOM FOREST
    rf_dict = run_random_forest(
        matrix=matrix_flat,
        labels=labels,
        mz_values=mz_values,
        run_folder=run_folder,
        name_of_run="hippocampus_hierachica4_attempt",
    )

    # COMBO ANOVA/FDR + RFs
    consensus_df = combine_anova_rf(
        anova_results_df=anova_results_df,
        rf_importances_df=rf_dict["importances_df"],
        top_n_rf=100,
        run_folder=run_folder,
        name_of_run="hippocampus_hierachica4_attempt",
    )

    # plot first 3 ion images 
    for feature in consensus_df["mz"].head(3):
        reconstruct_and_plot_ion_images(
            matrix_scaled=matrix_scaled,
            mask=mask,
            original_shape=original_shape,
            mz_values=mz_values,
            feature=feature,
            rf_dict=rf_dict,
            run_folder=run_folder,
        )



def run_feature_selection(
        dimensionality_red_output: dict,
        params: dict, 
        run_folder: str):
    
    # load things
    matrix = dimensionality_red_output["matrix_scaled"]
    labels = dimensionality_red_output["labels"]
    mask = dimensionality_red_output["mask"]
    original_shape = dimensionality_red_output["original_shape"]
    name_of_run = params["run_id"]

    matrix_raw = np.load(f"{run_folder}\\matrix.npy")
    h, w, n_peaks = matrix_raw.shape
    matrix_flat = matrix_raw.reshape(h * w, n_peaks)

    if params["should_remove_matrix_peaks"]:
        all_mz_values = pd.read_csv(f"{run_folder}\\filtered_mz_values.csv")["mz"].values
        matrix_peaks = pd.read_csv(f"{run_folder}\\top_peaks_matrix_cluster_{dimensionality_red_output['matrix_cluster_id']}.csv")["m/z"].values
        # remove patrix peaks from mz peaks
        print("Removing matrix peaks in feature selection.")
        
        mz_values, removed = remove_matrix_peaks(
            peak_mz=all_mz_values,
            matrix_peaks=matrix_peaks,
            # ratio_threshold=0.5, 
            tol=0.1)
        
        keep_indices = [i for i, mz in enumerate(all_mz_values)
                    if any(np.abs(mz_values - mz) < 0.1)]
        print(f"all_mz_values: {len(all_mz_values)}")
        print(f"mz_values after remove_matrix_peaks: {len(mz_values)}")
        print(f"removed: {len(removed)}")
        print(f"keep_indices: {len(keep_indices)}")

        # flatten 3D matrix and apply mask before column filtering
        # matrix_flat = matrix_flat[mask]   # remove zero pixels
        # matrix_flat = matrix_flat[:, keep_indices]   # remove matrix peak columns
        
        mz_values_clean = all_mz_values[keep_indices]
        matrix_flat = matrix_flat[mask]
        matrix_flat = matrix_flat[:, keep_indices]

        
        #  so removing both matrix peaks and pixels !!!
        print(f"Matrix shape after removing matrix peaks: {matrix_flat.shape}")
        print(f"mz_values length: {len(mz_values)}")


        print(f"keep_indices: {len(keep_indices)}")
        print(f"mz_values_clean: {len(mz_values_clean)}")
        print(f"matrix_flat: {matrix_flat.shape}")

        assert matrix_flat.shape[1] == len(mz_values_clean) 
            # f"Mismatch: matrix has {matrix.shape[1]} columns but mz_values has {len(mz_values)} entries"

        save_path = f"{run_folder}\\filtered_mz_values_no_matrix_peaks.csv"
        pd.DataFrame({"mz": mz_values}).to_csv(save_path, index=False)
        print(f"Filtered m/z values saved to {save_path}")

    # perform anova on mz values (but without matrix peaks)
    p_values = perform_anova_test(
    matrix = dimensionality_red_output["matrix_scaled"], 
    # f"{run_folder}\\{params['dimred']}_results.csv"
    labels = dimensionality_red_output["labels"])
    
    reject, pvals_corrected = perform_fdr_correction(
        p_values=p_values
    )

    #   ANOVA + FDR
    anova_results_df = pd.DataFrame({
            "mz": pd.read_csv(f"{run_folder}\\filtered_mz_values.csv")["mz"].values,
            # "mz": ,
            "p_value": p_values,
            "adjusted_p_value": pvals_corrected,
            "significant_after_fdr": reject
        })
    anova_results_df.to_csv(f"{run_folder}\\anova_results_{name_of_run}.csv", index=False)

    # VOLCANO PLOT
    # volcano_plot_plotly(
    #     matrix=matrix,
    #     labels=labels,
    #     p_values=pvals_corrected, 
    #     run_folder=run_folder,
    #     mz_values=mz_values,
    #     name_of_run=name_of_run,
    # )

    # RANDOM FOREST
    rf_dict = run_random_forest(
        matrix=matrix_flat,
        labels=labels,
        mz_values=mz_values,
        run_folder=run_folder,
        name_of_run=name_of_run,
    )

    # COMBO ANOVA/FDR + RFs
    consensus_df = combine_anova_rf(
        anova_results_df=anova_results_df,
        rf_importances_df=rf_dict["importances_df"],
        top_n_rf=100,
        run_folder=run_folder,
        name_of_run=name_of_run,
    )

    
    # PLOT ION IMAGES
    if len(consensus_df) > 0:
        reconstruct_and_plot_ion_images(
            matrix_scaled=dimensionality_red_output["matrix_scaled"],
            mask=dimensionality_red_output["mask"],
            original_shape=dimensionality_red_output["original_shape"],
            mz_values=pd.read_csv(f"{run_folder}\\filtered_mz_values.csv")["mz"].values,
            feature=consensus_df["mz"].iloc[0],
            rf_dict=rf_dict,
            run_folder=run_folder,
        )
    else:
        print("No consensus features found -- skipping ion image reconstruction.")

    return {
        "anova_results_df": anova_results_df,
        "rf_dict": rf_dict,
        "consensus_df": consensus_df,
    }
