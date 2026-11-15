"""Quantitative validation, stability analysis and feature-contribution analysis."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)

from src import clustering as cl
from src import config


def _sil(X, lab, metric):
    return float(silhouette_score(X, lab, metric=metric)) if 2 <= len(set(lab)) < len(X) else np.nan


def sweep_table(X, models):
    """One row per (method, K): objective, silhouette (Euclid./Manhattan), Davies-Bouldin, sizes."""
    rows = []
    for (method, k), m in sorted(models.items(), key=lambda t: (t[0][0], t[0][1])):
        lab = m.labels_
        sizes = np.bincount(lab, minlength=k)
        rows.append(dict(
            method=method, k=k, objective=float(m.inertia_),
            objective_name="WCSS (sum of squared Euclidean distances)" if method == "kmeans"
            else "Within-cluster L1 cost (sum of Manhattan distances)",
            silhouette_euclidean=_sil(X, lab, "euclidean"),
            silhouette_manhattan=_sil(X, lab, "manhattan"),
            davies_bouldin=float(davies_bouldin_score(X, lab)),
            min_size=int(sizes.min()), max_size=int(sizes.max()), min_size_frac=float(sizes.min() / len(X))))
    return pd.DataFrame(rows)


def elbow_k(ks, wcss):
    """Elbow = point with the largest distance below the chord joining first and last point."""
    ks, w = np.asarray(ks, float), np.asarray(wcss, float)
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (w - w[-1]) / (w[0] - w[-1])
    return int(ks[np.argmax((1 - x - y) / np.sqrt(2))])


def seed_stability(X, k, method, seeds, n_init=1):
    """Mean pairwise ARI between runs that differ only in the random initialisation."""
    labs = [cl.fit_model(method, X, k, s, n_init).labels_ for s in seeds]
    aris = [adjusted_rand_score(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]
    return float(np.mean(aris)), float(np.std(aris))


def bootstrap_stability(X, k, method, seed, n_boot=30, frac=0.8, n_init=10):
    """Fit on random 80 % subsamples, assign all stations, compare with the full-data clustering."""
    ref = cl.fit_model(method, X, k, seed, 20).labels_
    rng = np.random.default_rng(seed)
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        m = cl.fit_model(method, X[idx], k, int(rng.integers(1_000_000)), n_init)
        aris.append(adjusted_rand_score(ref, m.predict(X)))
    return float(np.mean(aris)), float(np.std(aris))


def stability_table(X, k_range, seed, n_seeds=10, n_boot=30):
    rows = []
    for k in k_range:
        s_m, s_sd = seed_stability(X, k, "kmeans", list(range(seed, seed + n_seeds)))
        b_m, b_sd = bootstrap_stability(X, k, "kmeans", seed, n_boot)
        rows.append(dict(k=k, seed_ari_mean=s_m, seed_ari_std=s_sd, boot_ari_mean=b_m, boot_ari_std=b_sd))
    return pd.DataFrame(rows)


def cross_k_agreement(models):
    rows = []
    ks = sorted(k for (m, k) in models if m == "kmeans")
    for a, b in zip(ks[:-1], ks[1:]):
        la, lb = models[("kmeans", a)].labels_, models[("kmeans", b)].labels_
        rows.append(dict(k_from=a, k_to=b, ari=adjusted_rand_score(la, lb), nmi=normalized_mutual_info_score(la, lb)))
    return pd.DataFrame(rows)


def kmeans_vs_kmedians(models):
    rows = []
    for k in sorted(k for (m, k) in models if m == "kmeans"):
        if ("kmedians", k) in models:
            la, lb = models[("kmeans", k)].labels_, models[("kmedians", k)].labels_
            rows.append(dict(k=k, ari=adjusted_rand_score(la, lb), nmi=normalized_mutual_info_score(la, lb)))
    return pd.DataFrame(rows)


def select_final_k(sweep, stab, n_stations, k_min=3, min_frac=config.MIN_CLUSTER_FRAC):
    """Composite of silhouette, Davies-Bouldin, seed stability and bootstrap stability.

    Eligibility: K >= k_min (K = 2 is usually a trivial split) and every cluster holds at least
    max(3, min_frac * n) stations. Each criterion is min-max scaled over eligible K and averaged.
    The elbow K is reported separately and used only as a tie-breaker.
    """
    km = sweep[sweep.method == "kmeans"].merge(stab, on="k").reset_index(drop=True)
    km["min_ok"] = km["min_size"] >= max(3, int(np.ceil(min_frac * n_stations)))
    km["eligible"] = km["min_ok"] & (km["k"] >= k_min)
    el = km[km.eligible] if km.eligible.any() else km.copy()

    def mm(s, higher=True):
        r = s.max() - s.min()
        z = (s - s.min()) / r if r > 0 else s * 0 + 0.5
        return z if higher else 1 - z

    km["score"] = np.nan
    km.loc[el.index, "score"] = (mm(el.silhouette_euclidean) + mm(el.davies_bouldin, False)
                                 + mm(el.seed_ari_mean) + mm(el.boot_ari_mean)) / 4
    e = elbow_k(km.k, km.objective)
    km["is_elbow_k"] = km.k == e
    ranked = km.dropna(subset=["score"]).assign(tb=lambda d: -(d.is_elbow_k.astype(int)))
    best = ranked.sort_values(["score", "tb"], ascending=[False, True]).iloc[0]
    return int(best.k), e, km[["k", "silhouette_euclidean", "davies_bouldin", "seed_ari_mean", "boot_ari_mean",
                               "min_size", "min_ok", "eligible", "is_elbow_k", "score"]]


def feature_contribution(df, labels):
    """Between-cluster separation per feature: ANOVA F, eta-squared, Kruskal-Wallis H."""
    rows = []
    for c in df.columns:
        v = df[c].values
        groups = [v[labels == g] for g in np.unique(labels)]
        grand = v.mean()
        ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        sst = ((v - grand) ** 2).sum()
        try:
            F, p = stats.f_oneway(*groups)
            H, pk = stats.kruskal(*groups)
        except ValueError:
            F = p = H = pk = np.nan
        rows.append(dict(feature=c, eta_squared=ssb / sst if sst > 0 else np.nan,
                         anova_F=F, anova_p=p, kruskal_H=H, kruskal_p=pk))
    return pd.DataFrame(rows).sort_values("eta_squared", ascending=False)


def leave_one_feature_out(X, cols, k, seed, base_labels):
    """Re-cluster without each feature; ARI vs baseline shows how much structure depends on it."""
    rows = []
    for j, c in enumerate(cols):
        Xj = np.delete(X, j, axis=1)
        lab = cl.fit_model("kmeans", Xj, k, seed).labels_
        rows.append(dict(feature_removed=c, ari_vs_baseline=adjusted_rand_score(base_labels, lab),
                         silhouette=_sil(Xj, lab, "euclidean")))
    return pd.DataFrame(rows).sort_values("ari_vs_baseline")


def cluster_profiles(imputed_df, labels):
    """Mean / median / std per cluster, plus z-scores of cluster means (overall-std units)."""
    d = imputed_df.copy()
    d["cluster"] = labels
    mean = d.groupby("cluster").mean()
    median = d.groupby("cluster").median()
    std = d.groupby("cluster").std()
    z = (mean - imputed_df.mean()) / imputed_df.std(ddof=0).replace(0, np.nan)
    sizes = d["cluster"].value_counts().sort_index().rename("n_stations").to_frame()
    sizes["share"] = sizes["n_stations"] / len(d)
    return mean, median, std, z, sizes
