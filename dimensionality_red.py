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
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sklearn 
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, HDBSCAN
from scipy.ndimage import uniform_filter 
from scipy.stats import f_oneway
from scipy.spatial.distance import cdist
import statsmodels
from statsmodels.stats.multitest import multipletests
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh




results_folder = r"C:\Users\i6338212\data\results" # change folder path as needed



# preprocessing_run_name = "small_computer_xenium_omp"
# reduction_name = "OMP_pca10_k3_5x5_smoothing"
# run_folder = os.path.join(results_folder, preprocessing_run_name, reduction_name)
# os.makedirs(run_folder, exist_ok=True)

start_time = time.perf_counter()

print("Loaded packages! Starting dimensionality reduction...")


def load_and_preprocess_msi(
    file_path: str,
    run_folder: str,
    remove_zero_pixels: bool = True,
    save_raw: Optional[str] = None,
    save_scaled: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]: 

    
    # Load matrix
    matrix = np.load(file_path) 
    original_shape = (matrix.shape[0], matrix.shape[1]) if matrix.ndim == 3 else None
    print(f"Loaded matrix shape: {matrix.shape}")
    
    # # apply spatial smoothing 
    # if matrix.ndim == 3:
    #     matrix = uniform_filter(matrix.astype(float), size=[5, 5, 1])
    #     # averages each pixel's spectrum with its neighbors
    #     # 3×3 pixel window spatially but no smoothing across the m/z dimension

    # rehsape 3d matrix
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
    # save mask
    np.save(f"{run_folder}\\mask.npy", mask)
    print(f"Mask saved to {run_folder}\\mask.npy")
    # X = uniform_filter(X.astype(float), size=[5, 5])

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

def perform_pca(X: np.ndarray, n_components: int) -> np.ndarray:
    print("Performing PCA dimensionality reduction...")
    pca = PCA(n_components=n_components, 
              svd_solver='randomized', 
              random_state=42)
    pca_transformed = pca.fit_transform(X)
    loadings = pca.components_
    explained = pca.explained_variance_ratio_
    print("done with PCA! took {:.2f} seconds".format(time.perf_counter() - start_time))
    return pca_transformed, loadings, explained


def save_pca_results(pca_transformed: np.array,
                     labels: pd.Series,
                     save_path: str):
    pca_df = pd.DataFrame(
        pca_transformed,
        columns=[f"PC{i+1}" for i in range(pca_transformed.shape[1])]
    )
    pca_df["cluster"] = labels
    
    pca_df.to_csv(save_path, index=False)


def save_preprocessed_matrix(matrix: np.ndarray, save_path: str) -> None:
    np.save(save_path, matrix)
    print(f"Preprocessed matrix saved to {save_path}. Took {time.perf_counter() - start_time:.2f} seconds")


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
def plot_pca_3d(pca_transformed: np.ndarray,
                labels: pd.Series,
                explained_variance: Optional[np.ndarray] = None,
                data: Optional[pd.DataFrame] = None,
                title: str = "Interactive 3D PCA Visualization",
                height: int = 800,
                width: int = 1000,
                point_size: int = 3,
                opacity: float = 0.7,
                color_discrete_sequence: Optional[list[str]] = None,
                save_html: Optional[str] = None) -> go.Figure:

    if pca_transformed.shape[1] < 3:
        raise ValueError(f"Need at least 3 PCA components for 3D plot, got {pca_transformed.shape[1]}")

    print("Creating interactive 3D PCA visualization with Plotly...")

    if explained_variance is not None:
        axis_labels = [f"PC{i+1} ({explained_variance[i]*100:.1f}%)" for i in range(3)]
    else:
        axis_labels = ["PC1", "PC2", "PC3"]

    # Build plot dataframe
    df_plot = pd.DataFrame({
        'PC1': pca_transformed[:, 0],
        'PC2': pca_transformed[:, 1],
        'PC3': pca_transformed[:, 2],
        'Cluster': labels.astype(str)
    })

    # Add hover data 
    hover_data = None
    if data is not None:
        data = data.reset_index(drop=True)
        df_plot = df_plot.reset_index(drop=True)

        if 'Cell-ID' in data.columns:
            df_plot['Cell_ID'] = data['Cell-ID']
        if 'X centroid' in data.columns and 'Y centroid' in data.columns:
            df_plot['X_pos'] = data['X centroid']
            df_plot['Y_pos'] = data['Y centroid']

        hover_data = ['Cell_ID', 'X_pos', 'Y_pos']

    # Create 3D scatter
    fig = px.scatter_3d(
        df_plot,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Cluster',
        hover_data=hover_data,
        color_discrete_sequence=color_discrete_sequence,
        opacity=opacity,
        height=height,
        width=width,
        title=title,
        labels={
            'PC1': axis_labels[0],
            'PC2': axis_labels[1],
            'PC3': axis_labels[2]
        }
    )

    fig.update_traces(marker=dict(size=point_size))

    fig.update_layout(
        legend_title_text='Cluster',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            itemsizing='constant'
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2]
        )
    )

    if save_html:
        fig.write_html(save_html)
        print(f"3D PCA plot saved to {save_html}. Took {time.perf_counter() - start_time:.2f} seconds")

    return fig

def perform_spca(X: np.array, 
                 coords, 
                 n_components=10, 
                 bandwidth=1.0, 
                 alpha: float = 0.5):
    """
    X = matrix (n_pixels, n_features)
    coords = (n_pixels, 2) -- xy pixel coordinates
    bandwidth = how far neighbours influence each other
    alpha 0 = standard PCA, 1 = fully spatial
    """
    print("Performing SpatialPCA...")
    # normal pca
    pca = PCA(n_components=n_components, 
              svd_solver='randomized', 
              random_state=42)
    pca_transformed = pca.fit_transform(X)

    # gaussian spatial kernel
    distances = cdist(coords, coords, metric='euclidean')
    # distances[i,j] = distance between pixel i and j 
    # close together pixels = small value for this variable
    K = np.exp(-distances**2 / (2 * bandwidth**2))
    # turning distance into weights to convert distance to similarity
    # K[i, j] = "how much pixel j influences pixel i"

    # row normalise K
    K_normalised = K / K.sum(axis=1, keepdims = True)
    # each pixel takes a weighted average of neighbors

    # spatially smoothed embedding
    spatial_embedding = K_normalised @ pca_transformed
    # each pixel gets a new value of a weighted average of the neighbouring pca values


    combined_embedding = (1-alpha) * pca_transformed + alpha * spatial_embedding

    return combined_embedding

def get_pixel_coords(mask: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Get (x, y) coordinates for each non-zero pixel in the mask.
    Returns shape (n_pixels, 2).
    """
    height, width = original_shape
    all_coords = np.array([[i, j] for i in range(height) for j in range(width)])
    pixel_coords = all_coords[mask.flatten()]
    print(f"Extracted coordinates for {len(pixel_coords)} pixels")
    return pixel_coords

def build_pixel_grid_graph_sparse(coords: np.ndarray,
                                   connectivity: int = 4) -> csr_matrix:
    """
    Build a sparse spatial weight matrix for a pixel grid.
    Never computes all-to-all distances — looks up grid neighbours directly.
    
    connectivity=4:  up/down/left/right only
    connectivity=8:  includes diagonals
    
    Returns a sparse (n_pixels, n_pixels) adjacency matrix.
    """
    n = len(coords)
    coord_to_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(coords)}

    if connectivity == 4:
        offsets = [(-1,0),(1,0),(0,-1),(0,1)]
    else:
        offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    rows, cols, data = [], [], []
    for i, (r, c) in enumerate(coords):
        for dr, dc in offsets:
            j = coord_to_idx.get((int(r)+dr, int(c)+dc))
            if j is not None:
                rows.append(i)
                cols.append(j)
                data.append(1.0)

    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    print(f"Sparse graph: {W.nnz:,} edges, "
          f"{W.nnz/n:.1f} avg neighbours/pixel, "
          f"memory: {W.data.nbytes/1e6:.1f} MB")
    return W


def spatial_pca_sparse(X: np.ndarray,
                        coords: np.ndarray,
                        n_components: int = 10,
                        alpha: float = 0.5,
                        connectivity: int = 4,
                        chunk_size: int = 50_000,
                        run_folder: str = None,
                        start_time: float = None) -> np.ndarray:
    """
    Memory-safe spatial PCA using a sparse pixel grid graph.

    Args:
        X: (n_pixels, n_features) scaled intensity matrix
        coords: (n_pixels, 2) pixel (row, col) coordinates
        n_components: number of spatial PCs to compute
        alpha: blending weight — 0.0 = standard PCA, 1.0 = fully spatially smoothed
        connectivity: 4 (grid only) or 8 (include diagonals)
        chunk_size: rows to process at a time during smoothing — reduce if RAM is tight
        run_folder: where to save results
        start_time: perf_counter reference for timing
    
    Returns:
        embedding: (n_pixels, n_components) spatially aware embedding
    """
    print("Performing sparse SpatialPCA...")
    n_pixels = X.shape[0]
    t = start_time or time.perf_counter()

    # ── 1. Standard PCA on raw intensities ────────────────────────────────────
    print("Step 1/3: Running standard PCA...")
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    Z_pca = pca.fit_transform(X)   # (n_pixels, n_components)  float64
    explained = pca.explained_variance_ratio_
    print(f"  PCA done — top 3 components explain "
          f"{explained[:3].sum()*100:.1f}% variance. "
          f"Took {time.perf_counter()-t:.1f}s")

    # ── 2. Build sparse spatial graph ─────────────────────────────────────────
    print("Step 2/3: Building sparse spatial graph...")
    W = build_pixel_grid_graph_sparse(coords, connectivity=connectivity)

    # row-normalise: each pixel gets the mean of its neighbours' PCA scores
    # D^-1 W  (D = diagonal degree matrix)
    degree = np.array(W.sum(axis=1)).flatten()
    degree_inv = 1.0 / np.where(degree > 0, degree, 1.0)  # avoid /0 for isolated pixels
    D_inv = sp.diags(degree_inv)
    W_norm = D_inv @ W   # still sparse — no memory explosion

    print(f"  Graph normalised. Took {time.perf_counter()-t:.1f}s")

    # ── 3. Apply spatial smoothing in chunks (avoids loading full W*Z at once) ─
    print(f"Step 3/3: Applying spatial smoothing in chunks of {chunk_size:,}...")
    Z_smooth = np.empty_like(Z_pca)

    for start in range(0, n_pixels, chunk_size):
        end = min(start + chunk_size, n_pixels)
        # W_norm[start:end] is a sparse slice — the matmul stays sparse
        Z_smooth[start:end] = W_norm[start:end] @ Z_pca
        if start % 200_000 == 0:
            print(f"  Smoothed {end:,}/{n_pixels:,} pixels... "
                  f"({time.perf_counter()-t:.1f}s)")

    # ── 4. Blend standard PCA and spatially smoothed PCA ──────────────────────
    embedding = (1.0 - alpha) * Z_pca + alpha * Z_smooth

    print(f"Spatial PCA complete. Embedding shape: {embedding.shape}. "
          f"Total time: {time.perf_counter()-t:.1f}s")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    if run_folder:
        np.save(os.path.join(run_folder, "spatial_pca_embedding.npy"), embedding)

        # also save the explained variance from the underlying PCA
        pd.DataFrame({
            'component': [f'PC{i+1}' for i in range(n_components)],
            'explained_variance_ratio': explained
        }).to_csv(os.path.join(run_folder, "spatial_pca_explained_variance.csv"), index=False)

        print(f"Results saved to {run_folder}")

    return embedding, explained


def save_spatial_pca_results(embedding: np.ndarray,
                              labels: pd.Series,
                              save_path: str) -> None:
    df = pd.DataFrame(
        embedding,
        columns=[f"SPC{i+1}" for i in range(embedding.shape[1])]
    )
    df["cluster"] = labels.values
    df.to_csv(save_path, index=False)
    print(f"Spatial PCA results saved to {save_path}")

def kmeans_clustering(matrix: np.ndarray, 
                      n_clusters: int,
                      random_state: None,
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
                        random_state: None) -> pd.Series:
    
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

# run_folder = os.path.join(
#         results_folder,
#         folder_name,
#         params["run_id"]
#     )
# os.makedirs(run_folder, exist_ok=True)

def reconstruct_spatial_map(labels:pd.Series,
                            mask: np.ndarray,
                            original_shape: tuple,
                            run_folder: str,
                            run_name: str) -> np.ndarray:
    print("Reconstructing spatial map from cluster labels...")
    height, width = original_shape
    spatial_map = np.full(height * width, -1)  # creates an empty array of the original size filled with -1 (background)
    # spatial_map[mask] = labels.values
    # labels from array to series to align with mask indexing
    spatial_map[mask] = labels

    # mask is boolean array which indicates which pixels are non-zero (from preprocessing)
    reconstructed_map = spatial_map.reshape(height, width) # reshape back to original image dimensions
    np.save(f"{run_folder}\\spatial_map_matrix_{run_name}.npy", reconstructed_map)
    print("Spatial map reconstruction complete. Took {:.2f} seconds".format(time.perf_counter() - start_time))
    # reshape converts 1d array into 2d grid which now has bg and actual image of sample
    return reconstructed_map

def plot_spatial_map(spatial_map: np.ndarray,
                     title: str, 
                     run_folder: str):
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
    plt.savefig(f"{run_folder}\\ceva_elbow_method.png")
    plt.show()
    print(f"Elbow method plot saved to {run_folder}\\ceva_elbow_method.png. Took {time.perf_counter() - start_time:.2f} seconds")

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


# if __name__ == "__main__":
#     folder_name = "xenium_laptop"
#     run_folder = os.path.join(
#         results_folder,
#         folder_name,
#         "xenium_OMP_spca10_k5_smoothing"
#     )
# #     # run_timer(stop_event)
#     file_path = r"C:\Ioana\_uni\BTR_pipeline_code\msi_matrix_omp.npy"
#     matrix_scaled, mask, original_shape = load_and_preprocess_msi(file_path=file_path, 
#                                                             run_folder=run_folder,
#                                                               remove_zero_pixels=True,
#                                                               save_raw=f"{run_folder}\\matrix_raw.npy",
#                                                               save_scaled=f"{run_folder}\\matrix_scaled.npy")
# #     # matrix_scaled = np.load(f"{run_folder}\\matrix_scaled.npy")
# #     # mask = np.load(f"{run_folder}\\mask.npy")
# #     # original_shape = mask.shape
#     print(f"Loaded matrix with shape {original_shape}. Scaled matrix has shape {matrix_scaled.shape}")
# #     # pca_transformed = perform_pca(matrix_scaled, n_components=10)
# #     # save_preprocessed_matrix(pca_transformed, f"{run_folder}\\matrix_pca.npy")

#     coords = get_pixel_coords(mask, original_shape)
#     embedding, explained = spatial_pca_sparse(
#             X = matrix_scaled, 
#             coords = coords,
#             n_components= 10,
#             alpha = 0.5,
#             connectivity = 4, 
#             chunk_size= 50_000,
#             run_folder=run_folder,
#             start_time=start_time
#         )
    

# #     # read umap from csv if already done to save time
# #     # umap_file_path = f"{run_folder}\\umap_results.csv"
# #     # umap_transformed = pd.read_csv(umap_file_path).iloc[:, :2].values
# #     # labels = pd.read_csv(umap_file_path).iloc[:, 2].values
    
   
# #     # if os.path.exists(f"{run_folder}\\umap_results.csv"):
# #     #     print(f"Loading UMAP results from {f"{run_folder}\\umap_results.csv"}...")
# #     #     umap_df = pd.read_csv(f"{run_folder}\\umap_results.csv")
# #     #     umap_transformed = umap_df.iloc[:, :2].values
# #     #     labels = umap_df.iloc[:, 2].values
# #     #     print("UMAP results loaded successfully.")
# #     # else:
# #     #     umap_transformed = perform_umap(
# #     #         matrix_scaled, 
# #     #         n_neighbors=15, 
# #     #         min_dist=0.1, 
# #     #         n_components=2, 
# #     #         metric='euclidean', #can change to cosine to be faster
# #     #         # random_state=42, 
# #     #         supervised=False)
# #     #     labels = kmeans_clustering(matrix=umap_transformed, n_clusters=2, random_state=42, n_init=10, init='k-means++')
# #     #     save_umap_results(umap_transformed, labels, f"{run_folder}\\umap_results.csv")
   
# #     # if os.path.exists(f"{run_folder}\\pca_results.csv"):
# #     #     # print(f"Loading PCA results from {f"{run_folder}\\pca_results.csv"}...")
# #     #     pca_df = pd.read_csv(f"{run_folder}\\pca_results.csv")
# #     #     pca_transformed = pca_df.iloc[:, :-1].values
# #     #     labels = pca_df.iloc[:, -1].values
# #     #     print("PCA results loaded successfully.")
# #     # else:
# #     #     pca_transformed, loadings, explained = perform_pca(matrix_scaled, n_components=10)
# #     #     labels = kmeans_clustering(matrix=pca_transformed, n_clusters=3, random_state=42, n_init=10, init='k-means++')
# #     #     save_pca_results(pca_transformed, labels, f"{run_folder}\\pca_results.csv")

# #     pca_transformed, loadings, explained = perform_pca(matrix_scaled, n_components=10)
# #     labels = kmeans_clustering(matrix=pca_transformed, n_clusters=3, random_state=42, n_init=10, init='k-means++')
# #     save_pca_results(pca_transformed, labels, f"{run_folder}\\pca_results.csv")
# #     # embedding_sub,idx = subset_matrix(umap_transformed, subset_size=, seed=42)
# #     # labels_sub = hdbscan_clustering(embedding_sub, min_cluster_size=20)

#     plot_elbow_method(embedding, k_range=range(1,15))

# # redo all the first ones i did !!
#     # labels = kmeans_clustering(matrix=pca_transformed, n_clusters=5, random_state=42, n_init=10, init='k-means++')
#     # save_umap_results(pca_transformed, labels, f"{run_folder}\\pca_results.csv")
#     # save_umap_results(
#     #     umap_transformed, 
#     #     kmeans_labels,
#     #     f"{run_folder}\\umap_results.csv"
#     # )
#     # labels = hdbscan_clustering(matrix=umap_transformed, min_cluster_size=40, min_samples=None, cluster_selection_method='eom')
#     # save_umap_results(umap_transformed, labels, f"{run_folder}\\umap_results.csv")

#     plot_umap_plotly(pca_transformed, 
#         labels=labels,
#         title=f"Interactive PCA Visualization - {reduction_name}", 
#         save_html=f"{run_folder}\\pca_msi_{reduction_name}.html")
    
#     plot_pca_3d(pca_transformed, 
#                 labels,
#                 explained_variance=explained, 
#                 save_html=f"{run_folder}\\pca_3d_{reduction_name}.html")
#     spatial_map = reconstruct_spatial_map(labels, mask, original_shape)
#     # # # save spatial map as matrix w label for each pixel so we can do edge pixel analysis and stuff
#     plot_spatial_map(spatial_map, title=f"Mouse Brain MSI - {reduction_name}")

    
#     # umap_file_path=r"C:\Ioana\_uni\BTR_pipeline_code\results_segmentation_omp\umap_kmeans_k2_5x5_smoothing.csv"
#     # umap_transformed = pd.read_csv(umap_file_path)

#     print("Dimensionality reduction and clustering pipeline complete. Total time: {:.2f} seconds".format(time.perf_counter() - start_time))


def run_dimensionality_reduction(file_path: str, params: dict, run_folder: str):
    start_time = time.perf_counter()
    # folder_name = f"{params['dataset']}_{params['computer']}"
    # run_folder = os.path.join(
    #     results_folder,
    #     folder_name,
    #     params["run_id"]
    # )
    os.makedirs(run_folder,     exist_ok=True)

    # preprocessing
    matrix_scaled, mask, original_shape = load_and_preprocess_msi(
        file_path=file_path,
        run_folder=run_folder,
        remove_zero_pixels=params.get("remove_zero_pixels", True),
        save_raw=f"{run_folder}\\matrix_raw.npy",
        save_scaled=f"{run_folder}\\matrix_scaled.npy"
    )

    # dimensionality reduction
    if params["dimred"] == "pca":
        embedding, loadings, explained = perform_pca(
            matrix_scaled,
            n_components=params["n_components"]
        )

    elif params["dimred"] == "umap":
        embedding = perform_umap(
            matrix_scaled,
            n_neighbors=params.get("n_neighbors", 15),
            min_dist=params.get("min_dist", 0.1),
            n_components=params.get("n_components", 2)
        )
        explained = None
    elif params["dimred"] == "spca":
        coords = get_pixel_coords(mask, original_shape)
        embedding, explained = spatial_pca_sparse(
            X = matrix_scaled, 
            coords = coords,
            n_components= params.get("n_components", 10),
            alpha = params.get("spatial_alpha", 0.5),
            connectivity = params.get("spatial_connectivity", 4), 
            chunk_size= params.get("chunk_size", 50_000),
            run_folder=run_folder,
            start_time=start_time
        )
    else:
        raise ValueError(f"Unknown dimred method: {params['dimred']}")

    # clustering
    if params["clustering"] == "kmeans":
        labels = kmeans_clustering(
            embedding,
            n_clusters=params["n_clusters"],
            random_state=params.get("random_state", 42)
        )

    elif params["clustering"] == "hdbscan":
        labels = hdbscan_clustering(embedding)

    else:
        raise ValueError(f"Unknown clustering method")

    # save results
    if params["dimred"] == "pca":
        save_pca_results(embedding, labels, f"{run_folder}\\pca_results.csv")
    elif params["dimred"] == "spca":
        save_spatial_pca_results(embedding, labels, f"{run_folder}_spca_results.csv")
    else:
        save_umap_results(embedding, labels, f"{run_folder}\\umap_results.csv")

    # plotting
    plot_umap_plotly(
        embedding,
        labels,
        title=f"{params['dimred']}dimensionality reduction - {params['run_id']}",
        save_html=f"{run_folder}\\plot.html"
    )

    spatial_map = reconstruct_spatial_map(labels, mask, original_shape, run_folder, params['run_id'])
    plot_spatial_map(spatial_map, title=f"Spatial Map - {params['run_id']}", run_folder=run_folder)

    runtime = time.perf_counter() - start_time

    return {
        "embedding": embedding,
        "explained_variance": explained,
        # "spatial_map_file_path": ,
        "labels": labels,
        "run_name": params["run_id"],
        "runtime": runtime,
        "n_samples": len(labels),
        "n_clusters_found": len(set(labels)) - (1 if -1 in labels else 0)
    }

