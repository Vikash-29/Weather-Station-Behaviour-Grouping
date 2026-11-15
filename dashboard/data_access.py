"""Reads the REAL pipeline outputs for the dashboard. No Streamlit/Plotly imports here (testable).

Nothing in this module invents data: every value comes from files written by src/pipeline.py.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA

from src import config
from src import report_generator as rg
from src.feature_engineering import FEATURE_LABELS

METRIC_FILES = ["k_sweep_metrics.csv", "stability_by_k.csv", "k_selection_scores.csv", "preprocessing_variants.csv",
                "feature_contribution.csv", "leave_one_feature_out.csv", "cross_k_agreement.csv",
                "kmeans_vs_kmedians.csv", "cleaning_log.csv", "feature_selection_log.csv", "excluded_stations.csv",
                "completeness_threshold_sensitivity.csv", "run_summary.json"]
PROFILE_FILES = ["cluster_profile_mean.csv", "cluster_profile_median.csv", "cluster_profile_std.csv",
                 "cluster_profile_zscores.csv", "cluster_sizes.csv"]
PROCESSED_FILES = ["station_features_with_clusters.csv", "station_features_scaled.csv"]


def required_paths():
    return ([config.METRIC_DIR / f for f in METRIC_FILES] + [config.PROFILE_DIR / f for f in PROFILE_FILES]
            + [config.PROC_DIR / f for f in PROCESSED_FILES] + [config.ROOT / "final_report.md"])


def marker_path():
    return config.METRIC_DIR / "run_complete.json"


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def analysis_state():
    """state: not_run | incomplete | unverified | complete  (+ list of missing files).

    complete   = every required output exists AND the pipeline wrote its end-of-run marker
    unverified = all outputs exist but no marker (e.g. produced by an older version)
    incomplete = some outputs exist but not all
    not_run    = no outputs
    """
    req = required_paths()
    missing = [str(p.relative_to(config.ROOT)) for p in req if not p.exists()]
    marker = read_json(marker_path()) if marker_path().exists() else None
    existing = len(req) - len(missing)
    if existing == 0 and marker is None:
        return dict(state="not_run", missing=missing, marker=None)
    if missing:
        return dict(state="incomplete", missing=missing, marker=marker)
    if marker is None:
        return dict(state="unverified", missing=[], marker=None)
    return dict(state="complete", missing=[], marker=marker)


def signature():
    """Changes whenever any output file changes (used as cache key)."""
    return tuple((p.name, p.stat().st_mtime_ns) for p in required_paths() if p.exists())


def load_bundle():
    """Load every output table. Raises FileNotFoundError naming the file if something is missing."""
    M, P, D = config.METRIC_DIR, config.PROFILE_DIR, config.PROC_DIR
    b = {"summary": json.loads((M / "run_summary.json").read_text()),
         "marker": read_json(marker_path())}
    for key, f in dict(sweep="k_sweep_metrics.csv", stab="stability_by_k.csv", scores="k_selection_scores.csv",
                       variants="preprocessing_variants.csv", contrib="feature_contribution.csv",
                       loo="leave_one_feature_out.csv", cross_k="cross_k_agreement.csv",
                       kk="kmeans_vs_kmedians.csv", clean_log="cleaning_log.csv",
                       feat_log="feature_selection_log.csv", excluded="excluded_stations.csv",
                       sens="completeness_threshold_sensitivity.csv").items():
        b[key] = pd.read_csv(M / f)
    for key, f in dict(mean="cluster_profile_mean.csv", median="cluster_profile_median.csv",
                       std="cluster_profile_std.csv", z="cluster_profile_zscores.csv", sizes="cluster_sizes.csv").items():
        b[key] = pd.read_csv(P / f, index_col=0)
    b["stations"] = pd.read_csv(D / "station_features_with_clusters.csv", dtype={"STATION": str}, index_col=0)
    b["scaled"] = pd.read_csv(D / "station_features_scaled.csv", dtype={"STATION": str}, index_col=0)
    return b


def label(c):
    return FEATURE_LABELS.get(c, c)


def best_config(b):
    s, k = b["summary"], b["summary"]["selected_k"]
    row = b["sweep"][(b["sweep"].method == "kmeans") & (b["sweep"].k == k)].iloc[0]
    st = b["stab"][b["stab"].k == k].iloc[0]
    return dict(k=k, elbow_k=s["elbow_k"], silhouette=float(row.silhouette_euclidean),
                silhouette_manhattan=float(row.silhouette_manhattan), davies_bouldin=float(row.davies_bouldin),
                wcss=float(row.objective), seed_ari=float(st.seed_ari_mean), boot_ari=float(st.boot_ari_mean),
                min_size=int(row.min_size), max_size=int(row.max_size))


def completeness_stats(b):
    s, st = b["summary"], b["stations"]
    n_total = s["n_stations"] + s["n_stations_dropped"]
    out = dict(n_total=n_total, n_retained=s["n_stations"], n_excluded=s["n_stations_dropped"],
               pct_retained=100 * s["n_stations"] / n_total if n_total else float("nan"),
               min_valid_days=s["min_valid_temp_days"], n_prcp_unavailable=s["n_prcp_unavailable"],
               prcp_min_coverage=s["prcp_min_coverage"])
    out["median_valid_temp_days"] = float(st["n_valid_temp_days"].median())
    if "prcp_coverage" in st:
        out["median_prcp_coverage"] = float(st["prcp_coverage"].median())
    return out


def findings(b):
    """Plain-language findings, every number computed from the output files."""
    c, comp = best_config(b), completeness_stats(b)
    sizes = b["sizes"]
    out = [
        f"{comp['n_retained']} of {comp['n_total']} stations were analysed; {comp['n_excluded']} were excluded for having fewer than "
        f"{comp['min_valid_days']} valid temperature days. Results therefore describe well-reporting stations only (selection bias).",
        f"Selected configuration: K-Means with K = {c['k']} (Euclidean silhouette {c['silhouette']:.3f}, Davies-Bouldin "
        f"{c['davies_bouldin']:.3f}, WCSS {c['wcss']:.1f}).",
        rg.metric_agreement_text().replace("**", ""),
        f"Stability at K = {c['k']}: seed-to-seed ARI {c['seed_ari']:.2f}, 80% subsample refit ARI {c['boot_ari']:.2f}.",
        f"Cluster sizes range from {int(sizes.n_stations.min())} to {int(sizes.n_stations.max())} stations.",
    ]
    if comp["n_prcp_unavailable"]:
        out.append(f"Precipitation features are unavailable (coverage below {comp['prcp_min_coverage']:.0%}) for "
                   f"{comp['n_prcp_unavailable']} retained stations; those values were median-imputed.")
    out.append("Clusters are exploratory groupings of one year of observations, not permanent station categories or "
               "judgements of station reliability.")
    return out


def pca_frame(b):
    """2-D PCA of the saved standardised matrix, coloured by the saved cluster labels."""
    X = b["scaled"]
    st = b["stations"].reindex(X.index)
    pca = PCA(n_components=2, random_state=config.SEED).fit(X.values)
    Z = pca.transform(X.values)
    df = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "cluster": st["cluster"].astype(str).values,
                       "station": X.index, "name": st["name"].values}, index=X.index)
    return df, pca.explained_variance_ratio_


def station_table(b):
    st = b["stations"].reset_index()
    cols = [c for c in ["STATION", "name", "cluster", "latitude", "longitude", "elevation_m", "n_days",
                        "n_valid_temp_days"] if c in st.columns]
    return st[cols].sort_values(["cluster", "STATION"]).reset_index(drop=True)
