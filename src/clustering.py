"""K-Means (Euclidean) and K-Medians (Manhattan) implementations."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


class KMedians:
    """K-Medians: assignment by Manhattan (L1) distance, centres = coordinate-wise medians.

    This is the correct Manhattan-based counterpart of K-Means (the median minimises the
    sum of absolute deviations, the mean minimises the sum of squared deviations).
    Attributes after fit: cluster_centers_, labels_, inertia_ (= total within-cluster L1 cost).
    """

    def __init__(self, n_clusters, n_init=20, max_iter=300, random_state=0):
        self.n_clusters, self.n_init = n_clusters, n_init
        self.max_iter, self.random_state = max_iter, random_state

    def _init_centres(self, X, rng):
        idx = [rng.integers(len(X))]
        for _ in range(1, self.n_clusters):          # k-means++ style seeding with L1 distance
            d = cdist(X, X[idx], "cityblock").min(axis=1)
            p = d / d.sum() if d.sum() > 0 else None
            idx.append(rng.choice(len(X), p=p))
        return X[idx].copy()

    def fit(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            C = self._init_centres(X, rng)
            for _ in range(self.max_iter):
                D = cdist(X, C, "cityblock")
                lab = D.argmin(axis=1)
                newC = C.copy()
                for j in range(self.n_clusters):
                    m = lab == j
                    newC[j] = np.median(X[m], axis=0) if m.any() else X[D.min(axis=1).argmax()]
                if np.allclose(newC, C):
                    break
                C = newC
            D = cdist(X, C, "cityblock")
            cost = D.min(axis=1).sum()
            if best is None or cost < best[0]:
                best = (cost, C, D.argmin(axis=1))
        self.inertia_, self.cluster_centers_, self.labels_ = best
        return self

    def predict(self, X):
        return cdist(np.asarray(X, float), self.cluster_centers_, "cityblock").argmin(axis=1)


def make_model(method, k, seed, n_init=20):
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    if method == "kmedians":
        return KMedians(n_clusters=k, n_init=n_init, random_state=seed)
    raise ValueError(method)


def fit_model(method, X, k, seed, n_init=20):
    return make_model(method, k, seed, n_init).fit(X)


def run_models(X, k_range, seed, n_init=20, methods=("kmeans", "kmedians")):
    """Fit every method for every K. Returns {(method, k): fitted model}."""
    return {(m, k): fit_model(m, X, k, seed, n_init) for m in methods for k in k_range}


def relabel_by_size(labels):
    """Renumber clusters so that cluster 0 is the largest (labels are arbitrary otherwise)."""
    order = np.argsort(-np.bincount(labels))
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[l] for l in labels])
