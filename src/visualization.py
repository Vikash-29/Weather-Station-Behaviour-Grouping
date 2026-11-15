"""All figures. Every function saves a PNG into outputs/figures and returns the figure."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from src import config
from src.feature_engineering import FEATURE_LABELS

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
PALETTE = "tab10"


def _label(c):
    return FEATURE_LABELS.get(c, c)


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / name, dpi=150, bbox_inches="tight")
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_missing(daily_missing, station_missing):
    """Fig 1: missing-value distribution (daily variables and station-level features)."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    daily_missing.sort_values().plot.barh(ax=ax[0], color="#4c72b0")
    ax[0].set_title("Missing values per daily variable (after cleaning)")
    ax[0].set_xlabel("Missing daily records (%)"); ax[0].set_ylabel("Daily variable")
    s = station_missing.sort_values()
    s.rename(index=_label).plot.barh(ax=ax[1], color="#dd8452")
    ax[1].set_title("Missing values per station-level feature")
    ax[1].set_xlabel("Stations with missing value (%)"); ax[1].set_ylabel("Station-level feature")
    ax[1].tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return _save(fig, "01_missing_values.png")


def plot_distributions(df, cols):
    """Fig 2: histograms of the station-level weather features."""
    n = len(cols); nc = 4; nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3 * nr))
    for ax, c in zip(axes.ravel(), cols):
        sns.histplot(df[c].dropna(), bins=20, kde=False, ax=ax, color="#4c72b0")
        ax.set_title(_label(c), fontsize=8); ax.set_xlabel("Value (see title for unit)", fontsize=7)
        ax.set_ylabel("Number of stations", fontsize=7); ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Distribution of station-level weather features", y=1.0, fontsize=13)
    fig.tight_layout()
    return _save(fig, "02_feature_distributions.png")


def plot_corr(df, cols):
    """Fig 3: Pearson correlation heatmap."""
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, annot=len(cols) <= 24, fmt=".2f",
                annot_kws={"size": 6}, ax=ax, cbar_kws={"label": "Pearson correlation (r)"})
    ax.set_title("Correlation between station-level features")
    ax.set_xlabel("Feature"); ax.set_ylabel("Feature")
    ax.tick_params(labelsize=7)
    return _save(fig, "03_correlation_heatmap.png")


def plot_boxplots(Xs, cols):
    """Fig 4: boxplots of standardised features (outlier screening)."""
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(cols) + 2))
    ax.boxplot(Xs, vert=False, tick_labels=cols, flierprops=dict(marker="o", markersize=3, markerfacecolor="#c44e52"))
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_title("Boxplots of standardised station-level features (red points = potential outliers)")
    ax.set_xlabel("Standardised value (z-score, unitless)"); ax.set_ylabel("Feature")
    ax.tick_params(labelsize=8)
    return _save(fig, "04_boxplots_outliers.png")


def plot_elbow(sweep, elbow, chosen):
    """Fig 5: WCSS vs K."""
    d = sweep[sweep.method == "kmeans"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(d.k, d.objective, "o-", color="#4c72b0", label="K-Means WCSS")
    ax.axvline(elbow, color="#dd8452", ls="--", label=f"Elbow (chord method): K={elbow}")
    ax.axvline(chosen, color="#55a868", ls=":", lw=2, label=f"Selected K={chosen}")
    ax.set_title("Elbow method: WCSS vs number of clusters")
    ax.set_xlabel("Number of clusters (K)"); ax.set_ylabel("WCSS (sum of squared distances, standardised units²)")
    ax.legend()
    return _save(fig, "05_elbow_wcss.png")


def plot_silhouette(sweep):
    """Fig 6: Silhouette vs K for each method / evaluation distance."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    km = sweep[sweep.method == "kmeans"]; kd = sweep[sweep.method == "kmedians"]
    ax.plot(km.k, km.silhouette_euclidean, "o-", label="K-Means, Euclidean silhouette")
    ax.plot(km.k, km.silhouette_manhattan, "s--", label="K-Means labels, Manhattan silhouette (evaluation only)")
    ax.plot(kd.k, kd.silhouette_manhattan, "^-", label="K-Medians, Manhattan silhouette")
    ax.set_title("Silhouette score vs number of clusters")
    ax.set_xlabel("Number of clusters (K)"); ax.set_ylabel("Mean silhouette score (higher is better)")
    ax.legend(fontsize=8)
    return _save(fig, "06_silhouette.png")


def plot_db(sweep):
    """Fig 7: Davies-Bouldin vs K."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m, lab, mk in (("kmeans", "K-Means", "o-"), ("kmedians", "K-Medians", "^--")):
        d = sweep[sweep.method == m]
        ax.plot(d.k, d.davies_bouldin, mk, label=lab)
    ax.set_title("Davies-Bouldin index vs number of clusters")
    ax.set_xlabel("Number of clusters (K)"); ax.set_ylabel("Davies-Bouldin index (lower is better)")
    ax.legend()
    return _save(fig, "07_davies_bouldin.png")


def plot_pca(X, labels, k, centers=None, labels_alt=None, alt_name="K-Medians"):
    """Fig 8: 2-D PCA projection coloured by cluster (optionally beside an alternative clustering)."""
    pca = PCA(n_components=2, random_state=config.SEED).fit(X)
    Z = pca.transform(X)
    panels = [(labels, f"K-Means (K={k})")] + ([(labels_alt, f"{alt_name} (K={k})")] if labels_alt is not None else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 6), squeeze=False)
    ev = pca.explained_variance_ratio_ * 100
    for ax, (lab, title) in zip(axes[0], panels):
        sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=lab, palette=PALETTE, s=60, edgecolor="k", ax=ax, legend="full")
        if centers is not None and title.startswith("K-Means"):
            C = pca.transform(centers)
            ax.scatter(C[:, 0], C[:, 1], marker="X", s=220, c="black", label="Centroid")
        ax.set_title(f"PCA projection, {title}")
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}% of variance)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}% of variance)")
        ax.legend(title="Cluster", fontsize=8)
    fig.tight_layout()
    return _save(fig, "08_pca_clusters.png"), pca


def plot_sizes(labels):
    """Fig 9: number of stations per cluster."""
    cnt = np.bincount(labels)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar([str(i) for i in range(len(cnt))], cnt, color=sns.color_palette(PALETTE, len(cnt)), edgecolor="k")
    ax.bar_label(bars)
    ax.set_title("Cluster sizes"); ax.set_xlabel("Cluster"); ax.set_ylabel("Number of stations")
    return _save(fig, "09_cluster_sizes.png")


def plot_profile_heatmap(z):
    """Fig 10: cluster-wise feature profile (z-score of cluster mean vs overall mean)."""
    fig, ax = plt.subplots(figsize=(1.0 * len(z) + 9, 0.45 * z.shape[1] + 2))
    sns.heatmap(z.T.rename(index=_label), cmap="RdBu_r", center=0, annot=True, fmt=".1f", annot_kws={"size": 7},
                ax=ax, cbar_kws={"label": "Cluster mean minus overall mean (in overall std. deviations)"})
    ax.set_title("Cluster profiles: feature means relative to all stations")
    ax.set_xlabel("Cluster"); ax.set_ylabel("Feature"); ax.tick_params(axis="y", labelsize=8)
    return _save(fig, "10_cluster_profile_heatmap.png")


def plot_cluster_comparison(imputed, labels, features):
    """Fig 11: boxplots of the most discriminating features by cluster (original units)."""
    n = len(features); nc = 3; nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3.8 * nr), squeeze=False)
    for ax, f in zip(axes.ravel(), features):
        sns.boxplot(x=labels, y=imputed[f].values, hue=labels, palette=PALETTE, legend=False, ax=ax)
        ax.set_title(_label(f), fontsize=9); ax.set_xlabel("Cluster"); ax.set_ylabel(_label(f).split("(")[-1].rstrip(")") if "(" in _label(f) else "Value")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Most discriminating features by cluster (original units)", y=1.0, fontsize=13)
    fig.tight_layout()
    return _save(fig, "11_cluster_feature_comparison.png")


def plot_stability(stab, variants, cross_k, chosen):
    """Fig 12: seed / bootstrap stability vs K, K-to-K agreement and preprocessing-variant ARI."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].errorbar(stab.k, stab.seed_ari_mean, yerr=stab.seed_ari_std, fmt="o-", capsize=3, label="Random initialisation (single-start runs)")
    ax[0].errorbar(stab.k, stab.boot_ari_mean, yerr=stab.boot_ari_std, fmt="s--", capsize=3, label="80% subsample refit")
    ax[0].axvline(chosen, color="green", ls=":", label=f"Selected K={chosen}")
    ax[0].set_title("Stability vs K"); ax[0].set_xlabel("Number of clusters (K)")
    ax[0].set_ylabel("Adjusted Rand Index (1 = identical)"); ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=8)
    ax[1].bar([f"{a}→{b}" for a, b in zip(cross_k.k_from, cross_k.k_to)], cross_k.ari, color="#8172b3", edgecolor="k")
    ax[1].set_title("Agreement between consecutive K"); ax[1].set_xlabel("K → K+1")
    ax[1].set_ylabel("Adjusted Rand Index"); ax[1].set_ylim(0, 1.05)
    v = variants.sort_values("ari_vs_baseline")
    ax[2].barh(v.variant, v.ari_vs_baseline, color="#55a868", edgecolor="k")
    ax[2].set_title(f"Preprocessing variants vs baseline (K={chosen})")
    ax[2].set_xlabel("Adjusted Rand Index vs baseline"); ax[2].set_ylabel("Variant"); ax[2].set_xlim(0, 1.05)
    fig.tight_layout()
    return _save(fig, "12_stability.png")


def plot_geo(station_df, labels):
    """Fig 13 (supplementary): station locations coloured by cluster (not used for clustering)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.scatterplot(x=station_df["longitude"], y=station_df["latitude"], hue=labels, palette=PALETTE,
                    s=60, edgecolor="k", ax=ax)
    ax.set_title("Station locations coloured by cluster (location not used in clustering)")
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)"); ax.legend(title="Cluster")
    return _save(fig, "13_station_map_by_cluster.png")
