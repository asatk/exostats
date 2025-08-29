"""
demo_cluster.py

Demonstration of clustering on a subset of NASA exoplanet data with different
clustering techniques.

2025.08.29
Anthony Atkinson
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, MeanShift

# seed for random processes (e.g., initialization of clustering algorithms)
seed = 0x2025

fname = "tables-merged/nasa_exo.csv"
df = pd.read_csv(fname)

# remove 0s in columns where you do not want zeros (useful when taking logs)
nz_cols = ["st_age"]
df[nz_cols] = df[nz_cols].replace([0.0], np.nan)

# subset of Jupiters orbiting host stars with measured rotation periods
subset = (0.25 <= df["pl_bmassj"]) & \
         (df["pl_bmassj"] <= 13) & \
         pd.notnull(df["st_rotp"])

# placeholder to select all data
# subset = pd.notnull(df["pl_name"])

# log-transform data spread over many magnitudes to more managable numbers
logcols = ["pl_orbsmax", "pl_bmassj", "st_age", "sy_dist"]
df[logcols] = np.log10(df[logcols])

# calculate (log10) flux = luminosity / (4 pi dist^2)
flux = df["st_lum"] - 2 * df["sy_dist"] - np.log10(4 * np.pi)
df["flux"] = flux

# columns of relevant parameters for clustering
cols = ["pl_orbsmax", "flux"]
# dataframe of planet masses and orbits
df_sub = df.loc[subset, cols].dropna()

# unlabeled data to be cluster
X = df_sub.to_numpy()

# guess for number of clusters -- some methods must stick to that number
nc = 3

models = {
        "kMeans": KMeans(
            n_clusters=nc,
            init="k-means++",
            random_state=seed),
        "Spectral Clustering": SpectralClustering(
            n_clusters=nc,
            affinity="rbf",
            assign_labels="discretize",
            random_state=seed),
        "DBSCAN": DBSCAN(),
        "Mean-Shift": MeanShift(
            bin_seeding=True,
            cluster_all=True)
}


# plot data and clusters
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
fig.suptitle("Clustering Jupiters by stellar flux and orbit")
fig.supxlabel(r"$\log_{10}$ Semi-major Axis (au)")
ylabel = r"$\log_{10} \mathrm{Flux}\; (\mathcal{F} = \frac{\mathcal{L}}{4 \pi " +\
    r"d^2})$ ($\mathcal{L}_\odot$ au$^{-2}$)"
fig.supylabel(ylabel)
ax = ax.ravel()

for i, m in enumerate(models.keys()):
    # cluster data
    model = models[m]
    clustering = model.fit(X)

    # identify each datum's label
    labels = clustering.labels_
    labels_u = np.unique(labels)

    axi = ax[i]
    
    # plot each data point colored according to its label
    for lbl in labels_u:
        ind = labels == lbl
        Xi = X[ind]
        axi.scatter(Xi[:,0], Xi[:,1], label=f"Cluster {lbl}", s=5)
        axi.set_title(m)
        axi.legend(fontsize=8)

fig.tight_layout()
plt.show()
