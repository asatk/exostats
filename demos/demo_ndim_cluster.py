"""
demo_cluster.py

Demonstration of 3-dimensional clustering on NASA exoplanet data.

2025.11.05
Anthony Atkinson
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, MeanShift

# seed for random processes (e.g., initialization of clustering algorithms)
seed = 0x2025

# path to your data here
fname = "../db/nea/ps.csv"
df = pd.read_csv(fname)

# remove 0s in columns where you do not want zeros (useful when taking logs)
nz_cols = ["st_age"]
df[nz_cols] = df[nz_cols].replace([0.0], np.nan)

# placeholder to select all data
# subset = pd.notnull(df["pl_name"])

# log-transform data spread over many magnitudes to more managable numbers
logcols = ["pl_orbsmax", "pl_bmasse", "st_mass"]
df[logcols] = np.log10(df[logcols])

# columns of relevant parameters for clustering -- can have as many columns are there are in the data set
cols = ["pl_orbsmax", "pl_bmasse", "st_mass"]
# dataframe of planet masses and orbits
df_sub = df.loc[:, cols].dropna()

# unlabeled data to be cluster
X = df_sub.to_numpy()

# guess for number of clusters -- some methods must stick to that number
nc = 4

# clustering model specifications
model = SpectralClustering(
        n_clusters=nc,
        affinity="rbf",
        assign_labels="discretize",
        random_state=seed)

# cluster data
model = model
clustering = model.fit(X)

# identify each datum's label
labels = clustering.labels_
labels_u = np.unique(labels)


def plot_clusters_3d(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray,
                     title: str=None, label_1: str=None, label_2: str=None,
                     marker_min: int=5, marker_max: int=25, marker_alpha: float=0.1):
    """
    Plot the results of clustering in N dimensions using "three" dimensions.
    The first two dimensions are typical x and y; the "third" dimension is
    size of the plot marker. Typically useful when one of the variables 
    meausures a "size" of some sort.

    x1 : np.ndarray
        first dimension of data (x)
    x2 : np.ndarray
        second dimension of data (y)
    x3 : np.ndarray
        third dimension of data (size)

    title : str, default is None
        plot title
    label_1 : str, default is None
        axis label of first dimension (x)
    label_2 : str, default is None
        axis label of second dimension (y)

    marker_min : int, default is 5
        minimum marker size of scaled third dimension data
    marker_max : int, default is 25
        maximum marker size of scaled third dimension data
    marker_alpha : float, default is 0.1
        transparency of marker (from 0.0 to 1.0)
    """

    # plot data and clusters
    fig, ax = plt.subplots(figsize=(6,6))

    if title is not None:
        ax.set_title(title, fontsize=14)

    if label_1 is not None:
        ax.set_xlabel(label_1, fontsize=14)

    if label_2 is not None:
        ax.set_ylabel(label_2, fontsize=14)

    # scale 3rd dimension to reasonable marker sizes
    marker_size = (x3 - np.min(x2)) / (np.max(x2) - np.min(x2)) * (marker_max - marker_min) + marker_min

    # plot each data point colored according to its label
    for lbl in labels_u:
        ind = labels == lbl
        x1_i = x1[ind]
        x2_i = x2[ind]
        marker_size_i = marker_size[ind]
        ax.scatter(x1_i, x2_i, label=f"Cluster {lbl}", s=marker_size_i, alpha=0.5, lw=0)
        ax.legend(fontsize=8)

    fig.tight_layout()
    plt.show()

title = r"Spectral Clustering of Planets by orbit, M$_{pl}$, and M$_*$"
label_1 = r"$\log_{10}$ Semi-major Axis (au)"
label_2 = r"Planetary Mass (M$_\oplus$)"
label_3 = r"Stellar Mass (M$_\odot$)"

# orbit vs. planet mass
plot_clusters_3d(X[:,0], X[:,1], X[:,2], title=title, label_1=label_1, label_2=label_2)

# orbit vs. stellar mass
plot_clusters_3d(X[:,0], X[:,2], X[:,1], title=title, label_1=label_1, label_2=label_3)

# planet mass vs. stellar mass
plot_clusters_3d(X[:,1], X[:,2], X[:,0], title=title, label_1=label_2, label_2=label_3)
