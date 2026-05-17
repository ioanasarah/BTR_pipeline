# Developing a reliable pipeline for spatial segmentation and feature selection of MSI data

(Ongoing Bachelor Thesis Research - Ioana Sarah Aizic, Maastricht University)

**<ins> Research question:</ins> What are the effects of different preprocessing, feature engineering, and clustering algorithms, and their order on spatial segmentation of MSI data?**

This Bachelor Thesis Research aims to develop a scalable and reliable pipeline for
spatial segmentation and feature selection of MSI data from liver tissue.

## Organization of pipeline files  
- [`logger.py`](./logger.py)
The main file which initialises the run, allowing the user to set the preffered dataset, peak picking, filtering dimensionality reduction, and clustering method for each run. Running this file runs [preprocessing.py](./preprocessing.py), [dimensionality_red.py](./dimensionality_red.py), [feature_selection.py](./feature_selection.py), and [clustering_metrics.py](./clustering_metrics.py). There is also an option to turn batch_runner on or off, which iterates through a set folder and runs the pipeline on all .zarr files in the folder. 

 **To run full pipeline:**
```bash
  poetry run python logger.py
```

- [`preprcessing.py`](./preprocessing.py)
Preprocessing will be carried out according to the study performed by Guo et al., 2021. This involves performing spectrum alignment, peak detection, binning,
filtering, and pooling. Peak detection was done using Mean Absolute Deviation
estimation (MADestimation), but since this step is known to produce false positives, an orthogonal matching pursuit (OMP) algorithm will be compared to the
MADestimation. Binning and pooling involve grouping peaks which are close in m/z
values since they are likely to correspond to the same molecule. Filtering involves
discarding peaks appearing in less than 0.5% of pixels.
Using filtering or feature engineering algorithms has been shown to improve
results of machine learning algorithms applied. Median, guided, Gaussian, and
Savitzky-Golay filters are implemented before peak picking. The algorithm leading to the most accurate and clear spatial segmentation will be analysed.

- [`dimensionality_red.py`](./dimensionality_red.py)
Dimensionality reduction can be carried out through multiple methods: Uniform Manifold Approximation and Projection (UMAP), Principal Component Analysis (PCA),
spatial PCA (sPCA), Non-negative Matrix Factorization (NMF), and Minimum Noise Fraction (MNF). This file also carries out clustering, which will be carried out using unsupervised machine learning methods, namely
K-means, spectral, hierarchical, or HDBSCAN clustering. After each pixel has been assigned to
a cluster, each cluster is colour-coded and the original image of the sample is reconstructed, showing each pixel as the colour of its cluster, forming a segmentation map. 

- [`feature_selection.py`](./feature_selection.py)
The m/z peaks that differ between the clusters will be identified using an ANOVA
 and Benjamini–Hochberg False Discovery Rate (FDR) will be used to control for Type 1 error, before using a Random Forest (RF) classifier to find the most important molecular peaks which
cause variation between the clusters. These can be matched to metabolites and
used to determine the molecular differences between the regions of the samples.

- [`spectra_analysis.py`](./spectra_analysis.py)
Loads the outputs of a completed pipeline run and generates average mass spectra for each cluster, at the preprocessed peak level. This is only run if specified in the parameters in the [logger.py](./logger.py) For each cluster, a static PNG and an interactive HTML plot are produced, with the top peaks annotated. A combined overview figure showing all clusters is also saved.


## Standalone files
All of these scrips must be run after pipeline has been run, since they rely on the creation of files which are created during the analysis process. 

- [`tic_ion_image.py`](./tic_ion_image.py)
Creates a TIC image for any dataset given its .zarr path and results folder from its analysis. 

**To run this file:**
```bash
  poetry run python tic_ion_image.py
```

-   [`plot_ion_images.py`](./plot_ion_images.py)
Plots ion images for datasets containing one sample, given a feature m/z value, the dataset's .zarr file and the results folder from its analysis. These inputs should be defined at the end of the file, under "if __name__ == "__main__": ". 


**To run this file:**
```bash
  poetry run python plot_ion_images.py
```

-   [`ion_image_mosaic_feature.py`](./ion_image_mosaic_feature.py)
Plots ion images for the specified feature for any mosaic dataset. The dataset's .zarr file, results folder, and feature can be defined at the bottom of the file. 

**To run this file:**
```bash
  poetry run python ion_image_mosaic_feature.py
```



