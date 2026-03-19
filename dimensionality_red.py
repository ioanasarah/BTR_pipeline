import time
from typing import Tuple, Optional
import numpy as np
# import scikit-learn as sklearn
from sklearn.preprocessing import StandardScaler
# import umap-learn 
# from Usia.dimensionality_reduction.umap import perform_umap
import pandas as pd
import os
import umap
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import sklearn
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, HDBSCAN
from scipy.ndimage import uniform_filter 
from scipy.stats import f_oneway
import statsmodels
from statsmodels.stats.multitest import multipletests
import threading


results_folder = r"C:\Users\i6338212\data\results" # change folder path as needed
preprocessing_run_name = "hippocampus_tic_omp"
reduction_name = "hdbscan_40min_5x5_smoothing"
run_folder = os.path.join(results_folder, preprocessing_run_name, reduction_name)
os.makedirs(run_folder, exist_ok=True)

start_time = time.perf_counter()

print("Loaded packages! Starting dimensionality reduction...")

# def run_timer(stop_event):
#     start = time.time()
#     while not stop_event.is_set():
#         elapsed = time.time() - start
#         mins, secs = divmod(int(elapsed), 60)
#         hrs, mins = divmod(mins, 60)
#         print(f"\r Elapsed: {hrs:02d}:{mins:02d}:{secs:02d}", end="", flush=True)
#         time.sleep(1)
#     print()

# stop_event = threading.Event()
# timer_thread = threading.Thread(target=run_timer, args=(stop_event,), daemon=True)
# timer_thread.start()


def load_and_preprocess_msi(
    file_path: str,
    remove_zero_pixels: bool = True,
    save_raw: Optional[str] = None,
    save_scaled: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]: 

    
    # Load matrix
    matrix = np.load(file_path, allow_pickle=True) # should be in npy format 
    original_shape = (matrix.shape[0], matrix.shape[1]) if matrix.ndim == 3 else None
    print(f"Loaded matrix shape: {matrix.shape}")
    
    # # apply spatial smoothing 
    # if matrix.ndim == 3:
    #     matrix = uniform_filter(matrix.astype(float), size=[5, 5, 1])
    #     # averages each pixel's spectrum with its neighbors
    #     # 3×3 pixel window spatially but no smoothing across the m/z dimension

    # apply spatial smoothing on 3D matrix before flattening
    # size=[5, 5, 1] smooths spatially (5x5 pixel window) without mixing across m/z channels
    if matrix.ndim == 3:
        matrix = uniform_filter(matrix.astype(float), size=[5, 5, 1])

    # reshape 3d matrix
    if matrix.ndim == 3:
        height, width, n_peaks = matrix.shape
        X = matrix.reshape(height * width, n_peaks)
        print(f"Reshaped to: {X.shape}")
    else:
        X = matrix

    # Remove zero pixels (important for MSI)
    if remove_zero_pixels:
        mask = np.sum(X, axis=1) > 0
        X = X[mask]
        print(f"Shape after removing zero pixels: {X.shape}")
    else:
        mask = None

    if save_raw:
        np.save(save_raw, X)
        print(f"Raw matrix saved to {save_raw}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled += np.random.normal(0, 1e-6, X_scaled.shape) # add small noise to avoid zero variance issues in UMAP
    if save_scaled:
        np.save(save_scaled, X_scaled)
        print(f"Scaled matrix saved to {save_scaled}")

    print(f"Scaling completed in {time.perf_counter() - start_time:.2f} seconds")
    
    return X_scaled, mask, original_shape


def subset_matrix(matrix: np.ndarray, subset_size: int = 50_000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed) # using seed to get the same random subset each time for reproducibility
    idx = rng.choice(matrix.shape[0], size=subset_size, replace=False)
    return matrix[idx], idx  



def perform_umap(X: np.ndarray, 
                y: Optional[pd.Series] = None,
                n_neighbors: int = 15, 
                min_dist: float = 0.1, 
                n_components: int = 2,
                metric: str = 'euclidean',
                random_state: int = 42,
                supervised: bool = False,
                target_weight: float = 0.5) -> np.ndarray:
    """
    Perform UMAP on the given data with flexible parameters.

    Args:
        X: Input feature matrix
        y: Target labels (optional, used for supervised UMAP)
        n_neighbors: Number of neighbors to consider
        min_dist: Minimum distance between points
        n_components: Number of dimensions to reduce to
        metric: Distance metric to use
        random_state: Random seed for reproducibility
        supervised: Whether to use supervised UMAP
        target_weight: Weight of the target in supervised mode (0-1)
        
    Returns:
        UMAP-transformed data
    """
    print("Performing UMAP dimensionality reduction...")
    if supervised and y is not None:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            n_jobs=-1,
            init='random',
            target_metric='categorical',
            target_weight=target_weight
        )
        umap_transformed = reducer.fit_transform(X, y)
    else:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            # random_state=random_state,
            n_jobs=-1, 
            init='random' # what does this do?
        )
        umap_transformed = reducer.fit_transform(X)
    
    print("done with umap! took {:.2f} seconds".format(time.perf_counter() - start_time))
    return umap_transformed


def save_umap_results(umap_transformed: np.ndarray,
                      labels: pd.Series,
                    save_path: str) -> None:
    umap_df = pd.DataFrame({
    "UMAP1": umap_transformed[:,0],
    "UMAP2": umap_transformed[:,1],
    "cluster": labels
    })

    umap_df.to_csv(save_path, index=False)

def save_preprocessed_matrix(matrix: np.ndarray, save_path: str) -> None:
    np.save(save_path, matrix)
    print(f"Preprocessed matrix saved to {save_path}. Took {time.perf_counter() - start_time:.2f} seconds")


def plot_umap_matplotlib(umap_transformed: np.ndarray, 
                        labels: pd.Series,
                        title: str = "UMAP 2D Visualization",
                        save_path: Optional[str] = None) -> None:
    """
    Plot UMAP results using Matplotlib.
    
    Args:
        umap_transformed: UMAP-transformed data
        labels: Cluster labels
        title: Plot title
        save_path: Path to save the figure (None to display only)
    """
    plt.figure(figsize=(12, 10))
    scatter = sns.scatterplot(
        x=umap_transformed[:, 0],
        y=umap_transformed[:, 1],
        hue=labels.astype(str),
        palette="viridis",
        alpha=0.7,
        s=15
    )
    
    # Improve legend - keep only most frequent clusters in legend
    handles, labels_legend = scatter.get_legend_handles_labels()
    
    # Count frequencies
    label_counts = labels.value_counts()
    top_n = 15  # Show top N clusters in legend
    top_clusters = label_counts.nlargest(top_n).index.astype(str).tolist()
    
    # Filter legend items
    filtered_handles = [h for h, l in zip(handles, labels_legend) if l in top_clusters]
    filtered_labels = [l for l in labels_legend if l in top_clusters]
    
    plt.legend(filtered_handles, filtered_labels, title="Top Clusters", 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(title, fontsize=14)
    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_umap_plotly(umap_transformed: np.ndarray, 
                    labels: pd.Series,
                    data: Optional[pd.DataFrame] = None,
                    title: str = "Interactive UMAP Visualization",
                    height: int = 800,
                    width: int = 1000,
                    point_size: int = 5,
                    opacity: float = 0.7,
                    color_discrete_sequence: Optional[list[str]] = None,
                    save_html: Optional[str] = None) -> go.Figure:
    """
    Create an interactive Plotly visualization of UMAP results.
    
    Args:
        umap_transformed: UMAP-transformed data
        labels: Cluster labels
        data: Original dataframe (optional, for hover data)
        title: Plot title
        height, width: Figure dimensions
        point_size: Size of scatter points
        opacity: Point opacity
        color_discrete_sequence: Custom color sequence
        save_html: Path to save HTML (None to skip saving)
        
    Returns:
        Plotly figure object
    """
    print("Creating interactive UMAP visualization with Plotly...")

    # Create a dataframe for Plotly
    df_plot = pd.DataFrame({
        'UMAP_1': umap_transformed[:, 0],
        'UMAP_2': umap_transformed[:, 1],
        'Cluster': labels.astype(str)
    })
    
    # Add additional hover information if original data provided
    hover_data = None
    if data is not None:
        # Reset index to ensure alignment
        data = data.reset_index(drop=True)
        df_plot = df_plot.reset_index(drop=True)
        
        # Add Cell-ID and coordinates if available
        if 'Cell-ID' in data.columns:
            df_plot['Cell_ID'] = data['Cell-ID']
        
        if 'X centroid' in data.columns and 'Y centroid' in data.columns:
            df_plot['X_pos'] = data['X centroid']
            df_plot['Y_pos'] = data['Y centroid']
            
        hover_data = ['Cell_ID', 'X_pos', 'Y_pos']
    
    # Create the plot
    fig = px.scatter(
        df_plot, 
        x='UMAP_1', 
        y='UMAP_2',
        color='Cluster',
        hover_data=hover_data,
        color_discrete_sequence=color_discrete_sequence,
        opacity=opacity,
        height=height,
        width=width,
        title=title
    )
    
    # Customize the plot
    fig.update_traces(marker=dict(size=point_size))
    
    # Improve layout
    fig.update_layout(
        legend_title_text='Cluster',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            itemsizing='constant'
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Save as HTML if requested
    if save_html:
        fig.write_html(save_html)
        print(f"Interactive plot saved to {save_html}. Took {time.perf_counter() - start_time:.2f} seconds")
    
    return fig


def kmeans_clustering(matrix: np.ndarray,
                      n_clusters: int,
                      random_state: Optional[int] = None,
                      n_init: int = 10,
                      init: str = 'k-means++') -> pd.Series:

    kmeans_labels = KMeans(n_clusters=n_clusters, 
                           random_state=random_state,
                           n_init=n_init,
                           init=init)
    labels = kmeans_labels.fit_predict(matrix) # groups pixels from umap into clusters based on their similarity 
    print(f"KMeans complete. Found {n_clusters} clusters.")
    print(f"Cluster sizes: {pd.Series(labels).value_counts().sort_index().to_dict()}")
    print(f"KMeans clustering took {time.perf_counter() - start_time:.2f} seconds")
    return pd.Series(labels)

def spectral_clustering(matrix: np.ndarray,
                        n_clusters: int,
                        random_state: Optional[int] = None) -> pd.Series:
    
        # (random_state = None, n_components = 20, n_init = 10, gamma = 1, affinity = 
# ‘rbf’, n_neighbors = 10, eigen_tol = 0.0, assign_labels = ‘kmeans’, degree = 3)
    idx = np.random.choice(len(matrix), size=10000, replace=False)
    sample = matrix[idx]

    sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans')
    sample_labels = sc.fit_predict(sample)  # fit on a random sample to find cluster centers
    sc.fit(sample)

    centroids = np.array([sample[sample_labels == k].mean(axis=0) for k in range(n_clusters)])
    # Use KMeans to assign labels to all remaining points based on spectral cluster centers
    # from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, init=centroids, n_init=1)
    km.fit(sample)  # fit KMeans to same sample
    labels_all = km.predict(matrix)  
    # spectral_labels = SpectralClustering(
    #     n_clusters=n_clusters, 
    #     n_components=20,
    #     n_init=10, 
    #     gamma=1, 
    #     random_state=random_state, 
    #     affinity='nearest_neighbors',  # instead of rbf because it would be too much ram (like 1 tb)
    #     n_neighbors=10, 
    #     assign_labels='kmeans'
    # )
    # labels = labels_all
    print(f"Spectral Clustering complete. Found {n_clusters} clusters.")
    print(f"Cluster sizes: {pd.Series(labels_all).value_counts().sort_index().to_dict()}")
    print(f"Spectral clustering took {time.perf_counter() - start_time:.2f} seconds")
    return pd.Series(labels_all)



def hdbscan_clustering(matrix: np.ndarray,
                       min_cluster_size: int = 10, # minimum number of samples in a group for that group to be considered a cluster
                        min_samples: Optional[int] = None,
                        cluster_selection_method: str = 'eom') -> pd.Series: 
    # eom = excess of mass, selects clusters based on stability
    print("Performing HDBSCAN clustering...")
    hdbscan_labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method
        # core_dist_n_jobs=-1
    )
    labels = hdbscan_labels.fit_predict(matrix)
    print(f"HDBSCAN clustering complete. Found {len(np.unique(labels[labels >= 0]))} clusters.")
    print(f"Cluster sizes: {pd.Series(labels).value_counts().sort_index().to_dict()}")
    print(f"HDBSCAN clustering took {time.perf_counter() - start_time:.2f} seconds")
    return pd.Series(labels)


def reconstruct_spatial_map(labels:pd.Series,
                            mask: np.ndarray,
                            original_shape: tuple) -> np.ndarray:
    print("Reconstructing spatial map from cluster labels...")
    height, width = original_shape
    spatial_map = np.full(height * width, -1)  # creates an empty array of the original size filled with -1 (background)
    # spatial_map[mask] = labels.values
    # labels from array to series to align with mask indexing
    spatial_map[mask] = labels

    # mask is boolean array which indicates which pixels are non-zero (from preprocessing)
    reconstructed_map = spatial_map.reshape(height, width) # reshape back to original image dimensions
    np.save(f"{run_folder}\\spatial_map_matrix_{reduction_name}.npy", reconstructed_map)
    print("Spatial map reconstruction complete. Took {:.2f} seconds".format(time.perf_counter() - start_time))
    # reshape converts 1d array into 2d grid which now has bg and actual image of sample
    return reconstructed_map

def plot_spatial_map(spatial_map: np.ndarray,
                     title: str = reduction_name):
    print( "Plotting spatial map of clusters...")
    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(spatial_map, cmap='tab10', interpolation='nearest')
    plt.colorbar(image, ax=ax, label="Cluster")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X (pixels)", fontsize=12)
    ax.set_ylabel("Y (pixels)", fontsize=12)

   
    plt.savefig(f"{run_folder}\\spatial_map.png", dpi=300, bbox_inches='tight')
    print(f"Spatial map figure saved to {run_folder}\\spatial_map.png")
    print("Spatial map plotting complete. Took {:.2f} seconds".format(time.perf_counter() - start_time))
    plt.show()


def plot_elbow_method(umap_transformed: np.ndarray, k_range: range) -> None:
    inertias = []
    # k_range = range(1, 10)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(umap_transformed)
        inertias.append(km.inertia_)

    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.savefig(f"{run_folder}\\{reduction_name}_elbow_method.png")
    plt.show()
    print(f"Elbow method plot saved to {run_folder}\\{reduction_name}_elbow_method.png. Took {time.perf_counter() - start_time:.2f} seconds")

# umap_transformed = perform_umap(
#     matrix_scaled, 
#     n_neighbors=15, 
#     min_dist=0.1, 
#     n_components=2, 
#     metric='euclidean', #can change to cosine to be faster
#     # random_state=42, 
#     supervised=False)

# kmeans_labels = kmeans_clustering(matrix=umap_transformed, n_clusters=2, random_state=42, n_init=10, init='k-means++')

# save_umap_results(
#     umap_transformed,
#     kmeans_labels,
#     f"{results_folder}\\umap_results.csv"
# )

# load umap from csv



# spectral_labels = spectral_clustering(matrix=umap_transformed, n_clusters=10, random_state=None)
# labels = spectral_labels
# labels = spectral_clustering(matrix=umap_transformed, n_clusters=10, random_state=None)

# fig = plot_umap_plotly(
#     umap_transformed, 
#     labels=kmeans_labels,
#     title="Interactive UMAP Visualization with KMeans Clusters", 
#     save_html=f"{results_folder}\\umap_msi_kmeans_k2_5x5_smoothing.html")

# spatial_map = reconstruct_spatial_map(kmeans_labels, mask, original_shape)
# plot_spatial_map(spatial_map, title="Mouse Brain MSI - KMeans Spatial Clusters", save_path=f"{results_folder}\\spatial_clusters_kmeans_k2_5x5_smoothing.png")

# print(f"Total time for dimensionality reduction and visualization: {time.perf_counter() - start_time:.2f} seconds")




# *import umap.plot*

# *reducer = umap.UMAP().fit(data)*
# *umap.plot.connectivity(reducer)*
# [image: image.png]
 
# new umap should be 23 feb 16:04 sau cv de genul called ceva spectral 10 smoothing 



# normalisation: TIC
# peak picking: baseline, smoothing, 


if __name__ == "__main__":
    # run_timer(stop_event)
    file_path = r"C:\Users\i6338212\data\msi_matrix_hippocampus_omp.npy"
    matrix_scaled, mask, original_shape = load_and_preprocess_msi(file_path=file_path, 
                                                              remove_zero_pixels=True,
                                                              save_raw=f"{run_folder}\\matrix_raw.npy",
                                                              save_scaled=f"{run_folder}\\matrix_scaled.npy")
        
    # read umap from csv if already done to save time
    # umap_file_path = f"{run_folder}\\umap_results.csv"
    # umap_transformed = pd.read_csv(umap_file_path).iloc[:, :2].values
    # labels = pd.read_csv(umap_file_path).iloc[:, 2].values
    
   

    umap_transformed = perform_umap(
        matrix_scaled, 
        n_neighbors=15, 
        min_dist=0.1, 
        n_components=2, 
        metric='euclidean', #can change to cosine to be faster
        # random_state=42, 
        supervised=False)
    
    # embedding_sub,idx = subset_matrix(umap_transformed, subset_size=, seed=42)
    # labels_sub = hdbscan_clustering(embedding_sub, min_cluster_size=20)

    # plot_elbow_method(umap_transformed, k_range=range(1, 8))


    # kmeans_labels = kmeans_clustering(matrix=umap_transformed, n_clusters=3, random_state=42, n_init=10, init='k-means++')
    # save_umap_results(
    #     umap_transformed,
    #     kmeans_labels,
    #     f"{run_folder}\\umap_results.csv"
    # )
    labels = hdbscan_clustering(matrix=umap_transformed, min_cluster_size=40, min_samples=None, cluster_selection_method='eom')
    save_umap_results(umap_transformed, labels, f"{run_folder}\\umap_results.csv")

    plot_umap_plotly(umap_transformed, 
        labels=labels,
        title=f"Interactive UMAP Visualization - {reduction_name}", 
        save_html=f"{run_folder}\\umap_msi_{reduction_name}.html")
    

    spatial_map = reconstruct_spatial_map(labels, mask, original_shape)
    # # # save spatial map as matrix w label for each pixel so we can do edge pixel analysis and stuff
    plot_spatial_map(spatial_map, title=f"Mouse Brain MSI - {reduction_name}")

    
    # umap_file_path=r"C:\Ioana\_uni\BTR_pipeline_code\results_segmentation_omp\umap_kmeans_k2_5x5_smoothing.csv"
    # umap_transformed = pd.read_csv(umap_file_path)

    print("Dimensionality reduction and clustering pipeline complete. Total time: {:.2f} seconds".format(time.perf_counter() - start_time))


    


