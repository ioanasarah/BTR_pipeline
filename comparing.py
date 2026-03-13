import time
from typing import Tuple, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
# import umap-learn 
# from Usia.dimensionality_reduction.umap import perform_umap
import pandas as pd
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sklearn 
from sklearn.cluster import KMeans
# import umap.plot


def load_and_preprocess_msi(
    file_path: str,
    remove_zero_pixels: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess MSI matrix saved as .npy
    
    Args:
        file_path: Path to .npy file (flattened matrix or 3D matrix)
        remove_zero_pixels: Whether to remove pixels with zero intensity
        
    Returns:
        Tuple containing:
        - scaled feature matrix (X_scaled)
        - mask of kept pixels (for spatial reconstruction)
    """
    
    # Load matrix
    matrix = np.load(file_path) # should be in npy format 
    print(f"Loaded matrix shape: {matrix.shape}")
    
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
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled += np.random.normal(0, 1e-6, X_scaled.shape) # add small noise to avoid zero variance issues in UMAP
    print("Scaling complete.")
    
    return X_scaled, mask

file_path = r"C:\Ioana\_uni\btr\msi_matrix.npy"
matrix_scaled, mask = load_and_preprocess_msi(file_path=file_path, remove_zero_pixels=True)


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
    if supervised and y is not None:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state,
            n_jobs=-1,
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
            random_state=random_state,
            n_jobs=-1, 
            # init='random' # what does this do?
        )
        umap_transformed = reducer.fit_transform(X)
    
    print("done with umap!")
    return umap_transformed


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
        print(f"Interactive plot saved to {save_html}")
    
    return fig


def kmeans_clustering(matrix: np.ndarray, 
                      n_clusters: int = 10,
                      random_state: int = 42,
                      n_init: int = 10) -> np.ndarray:
    # (random_state = None, n_components = 20, n_init = 10, gamma = 1, affinity = 
# ‘rbf’, n_neighbors = 10, eigen_tol = 0.0, assign_labels = ‘kmeans’, degree = 3)
    kmeans_labels = KMeans(n_clusters=n_clusters, 
                           random_state=random_state,
                           n_init=n_init)
    labels = kmeans_labels.fit_predict(matrix)
    print(f"KMeans complete. Found {n_clusters} clusters.")
    print(f"Cluster sizes: {pd.Series(labels).value_counts().sort_index().to_dict()}")
    return pd.Series(labels)

umap_transformed = perform_umap(
    matrix_scaled, 
    n_neighbors=15, 
    min_dist=0.1, 
    n_components=2, 
    metric='euclidean', #can change to cosine to be faster
    random_state=42, 
    supervised=False)

# umap.plot.points(umap, labels=None, theme='viridis')

# figure = plot_umap_matplotlib(umap, labels=None, title="UMAP 2D Visualization of MSI Data", save_path="umap_msi.png")


# labels = pd.Series(["All"] * umap_tranformed.shape[0])

# fig = plot_umap_plotly(
#     umap_tranformed, 
#     labels=labels,
#     title="Interactive UMAP Visualization", 
#     save_html = "umap_msi_spectral.html")


# print(f"UMAP visualisation saved to umap_msi_spectral.html")

# elbow method to find optimal k for kmeans clustering on umap results
inertias = []
k_range = range(2, 20)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(umap_transformed)
    inertias.append(km.inertia_)

plt.plot(k_range, inertias, 'bo-')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

kmeans_labels = kmeans_clustering(matrix=umap_transformed, n_clusters=10)
labels = kmeans_labels

fig = plot_umap_plotly(
    umap_transformed, 
    labels=labels,
    title="Interactive UMAP Visualization with KMeans Clusters", 
    save_html = "umap_msi_kmeans_spectral.html")

print(f"UMAP visualisation with KMeans clusters saved to umap_msi_kmeans_spectral.html")
# print time it took to finish this
print(f"Total time for dimensionality reduction and visualization: {time.perf_counter():.2f} seconds")

# *import umap.plot*

# *reducer = umap.UMAP().fit(data)*
# *umap.plot.connectivity(reducer)*
# [image: image.png]