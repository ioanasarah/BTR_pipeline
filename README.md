# Developing a reliable pipeline for spatial segmentation and feature selection of MSI data

(Ongoing Bachelor Thesis Research - Ioana Sarah Aizic, Maastricht University)

**<ins> Research question:</ins> What are the effects of different preprocessing, feature engineering, and clustering algorithms, and their order on spatial segmentation of MSI data?**

This Bachelor Thesis Research aims to develop a scalable and reliable pipeline for
spatial segmentation and feature selection of MSI data from liver tissue.

## Organization of files  
- [`logger.py`](./logger.py)
The main file which initialises the run, allowing the user to set the preffered dataset, peak picking, filtering dimensionality reduction, and clustering method for each run. Running this file runs [preprocessing.py](./preprocessing.py), [dimensionality_red.py](./dimensionality_red.py), [feature_selection.py](./feature_selection.py), and [clustering_metrics.py](./clustering_metrics.py). There is also an option to turn batch_runner on or off, which iterates through a set folder and runs the pipeline on all .zarr files in the folder. 

- [`preprcessing.py`](./preprocessing.py)
Preprocessing will be carried out according to the study performed by Guo et al., 2021. This involves performing spectrum alignment, peak detection, binning,
filtering, and pooling. Peak detection was done using Mean Absolute Deviation
estimation (MADestimation), but since this step is known to produce false positives, an orthogonal matching pursuit (OMP) algorithm will be compared to the
MADestimation. Binning and pooling involve grouping peaks which are close in m/z
values since they are likely to correspond to the same molecule. Filtering involves
discarding peaks appearing in less than 0.5% of pixels.
Using filtering or feature engineering algorithms has been shown to improve
results of machine learning algorithms applied. Median, guided, Gaussian, and
Savitzky-Golay filters will be implemented before peak picking. The algorithm leading to the most accurate and clear spatial segmentation will be analysed.

- [`dimensionality_red.py`](./dimensionality_red.py)
Dimensionality reduction was carried out through a Uniform Manifold Approximation and Projection (UMAP) approach in the study by Guo et al., 2021. The
results of the procedure will be compared to Principal Component Analysis (PCA),
spatial PCA (sPCA), and minimum noise fraction (MNF), as well as pair combinations of these techniques. This file also carries out clustering, which will be carried out using unsupervised machine learning methods, namely
k-means, spectral, or hierarchical clustering. After each pixel has been assigned to
a cluster, each cluster is colour-coded and the original image of the sample is reconstructed, showing each pixel as the colour of its cluster

- [`feature_selection.py`](./feature_selection.py)
The m/z peaks that differ between the clusters will be identified using an ANOVA
(F-test), or Kruskal–Wallis test, allowing for comparison between a parametric and
non-parametric approach at this step. The Benjamini–Hochberg False Discovery Rate (FDR) will be used to control for Type 1 error, before using a
random forest (RF) classifier to find the most important molecular peaks which
cause variation between the clusters. These can be matched to metabolites and
used to determine the molecular differences between the regions of the samples.
