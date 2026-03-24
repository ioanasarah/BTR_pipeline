import sklearn.metrics 
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, rand_score
import os 
import time
import numpy as np
import pandas as pd

print("Loaded packages for calculating clustering metrics")

# results_folder = r"C:\Ioana\_uni\BTR_pipeline_code\results" # change folder path as needed
# preprocessing_run_name = "small_computer_xenium_omp"
# reduction_name = "pca_10_components_k3_5x5_smoothing"
# run_folder = os.path.join(results_folder, preprocessing_run_name, reduction_name)
# os.makedirs(run_folder, exist_ok=True)


start_time = time.perf_counter()

# data = pd.read_csv(f"{run_folder}/pca_results.csv")
# print("Data loaded in {:.2f} seconds".format(time.perf_counter() - start_time))

# labels = data.iloc[:, -1] # cluster labels are in the last column
# pca_transformed = data.iloc[:, :-1] # umap embeddings in the first two columns
# print("Extracted labels and embeddings in {:.2f} seconds".format(time.perf_counter() - start_time))


def percentage_abnormal_edge_pixels(spatial_map, start_time: float = None) -> float:
    # BUG FIX: start_time is now a parameter instead of reading the module-level
    # variable.  The module-level start_time is set at import time, so the old
    # "completed in X seconds" message reported time-since-import, not
    # time-in-function.  Callers pass their own local start time.
    if start_time is None:
        start_time = time.perf_counter()
    # spatial_reconstruction is a 2D array of the same shape as the original image, where each pixel's value is the cluster label assigned to that pixel
    # edge_window defines how many pixels from the ith pixel is used to determine if i is edge pixel or not 
    count_abnormal_edge_pixels = 0
    total_edge_pixels = 0
    # total_pixels = 0
    for i in range(spatial_map.shape[0]):
        if i % 100 == 0:
            print(f"Processed {i}/{spatial_map.shape[0]} pixels")
        for j in range(spatial_map.shape[1]):
            # check if the pixel is on the edge of the image
            label = spatial_map[i, j]
            if label == -1: # if the pixel is labeled as noise /bg then we skip
                continue
            total_edge_pixels += 1
            # colect neighbour pixels by looking at the 8 pixels around the current pixel (i,j) and checking their labels
            neighbour_labels = []
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    if x == 0 and y == 0:
                        continue # skip the current pixel
                    nx, ny = i+x, j+y # neighbour pixel coordinates
                    if 0 <= nx < spatial_map.shape[0] and 0 <= ny < spatial_map.shape[1]: # check if neighbour pixel is within bounds
                        neighbour_label = spatial_map[nx, ny]
                        if neighbour_label != -1: # if the neighbour pixel is not labeled as noise / bg then we consider it for edge detection
                            neighbour_labels.append(neighbour_label)
            different_labels = sum(1 for nl in neighbour_labels if nl != label)
            if different_labels > len(neighbour_labels) / 2:
                count_abnormal_edge_pixels += 1

    paep = (count_abnormal_edge_pixels/total_edge_pixels if total_edge_pixels > 0 else 0.0) * 100
    print(f"Percentage of abnormal edge pixels: {paep:.4f}%")
    print("Edge pixel analysis completed in {:.2f} seconds".format(time.perf_counter() - start_time))
    return paep


# spatial_map_file_path = f"{run_folder}\\spatial_map_matrix_{reduction_name}.npy"
# spatial_map = np.load(spatial_map_file_path)
# paep = percentage_abnormal_edge_pixels(spatial_map)
    


# # silhouette_avg = silhouette_score(umap_embeddings, labels)
# davies_bouldin_avg = davies_bouldin_score(pca_transformed, labels)
# calinski_harabasz_avg = calinski_harabasz_score(pca_transformed, labels)
# # rand_score = rand_score(labels, labels) # need ground truth labels for this
# # print(f"Silhouette Score: {silhouette_avg:.4f}")

# print(f"Davies-Bouldin Score: {davies_bouldin_avg:.4f}")
# print(f"Calinski-Harabasz Score: {calinski_harabasz_avg:.4f}")
# # print("Clustering metrics calculated in {:.2f} seconds".format(time.perf_counter() - start_time))

# # save metrics to a text file
# with open(f"{run_folder}/clustering_metrics_{reduction_name}.txt", "w") as f:
#     # f.write(f"Silhouette Score: {silhouette_avg:.4f}\n")
#     f.write(f"Davies-Bouldin Score: {davies_bouldin_avg:.4f}\n")
#     f.write(f"Calinski-Harabasz Score: {calinski_harabasz_avg:.4f}\n")
#     f.write(f"Percentage of abnormal edge pixels: {paep:.4f}\n%")

# print("Clustering metrics saved to file. Total time: {:.2f} seconds".format(time.perf_counter() - start_time))

def run_clustering_metrics(dimensionality_red_output, run_folder, params):
    spatial_map_file_path = f"{run_folder}\\spatial_map_matrix_{params['run_id']}.npy"
    spatial_map = np.load(spatial_map_file_path)
    X = dimensionality_red_output["embedding"]
    labels = dimensionality_red_output["labels"]

    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)

    # BUG FIX: pass a local start_time into percentage_abnormal_edge_pixels
    # so the reported duration is time spent in the function, not time since
    # module import.  The module-level start_time (line 17) is set at import,
    # so "completed in X seconds" would include all time since the module
    # was first loaded -- a meaningless and always-growing number.
    metrics_start = time.perf_counter()
    paep = percentage_abnormal_edge_pixels(spatial_map=spatial_map,
                                           start_time=metrics_start)

    return {
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "paep": paep
    }