"""End-to-end pipeline. Each step_* function is also called one by one from the notebook."""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src import clustering as cl
from src import config
from src import data_preprocessing as dp
from src import evaluation as ev
from src import feature_engineering as fe
from src import visualization as viz

# Preprocessing / feature-selection variants used for the sensitivity analysis
VARIANTS = {
    "robust scaler": dict(scaler="robust"),
    "KNN imputation": dict(imputer="knn"),
    "log1p on precipitation": dict(log1p=["precip_observed_total_mm", "precip_intensity_mm"]),
    "no pressure features": dict(drop=["slp_mean", "slp_std", "stp_mean", "stp_std"]),
    "no extreme-value features": dict(drop=["temp_min", "temp_max", "wind_peak"]),
    "no event-report features": dict(drop=["fog_reported_frac", "thunder_reported_frac"]),
    "no precipitation features": dict(drop=["precip_observed_total_mm", "rainy_day_frac", "precip_intensity_mm"]),
    "plus data-completeness features": dict(add=fe.DQ_FEATURES),
}


def step_load(year=config.YEAR, skip_download=False):
    config.ensure_dirs()
    if not skip_download:
        stations = dp.select_candidate_stations(year)
        print(f"Candidate stations in ISD history: {len(stations)}")
        print("Download summary:", dp.download_gsod(year, stations["station_id"].tolist()))
    return dp.load_raw_observations(year)


def step_clean(raw):
    return dp.clean_observations(raw)


def step_features(clean, year=config.YEAR):
    daily = fe.add_daily_derived(clean)
    daily_missing = fe.daily_missing_pct(daily)
    station_df, excluded, sens = fe.aggregate_to_stations(daily, year)
    cols, sel_log = fe.select_features(station_df)
    return station_df, cols, sel_log, daily_missing, excluded, sens


def step_matrix(station_df, cols, scaler="standard", imputer="median", log1p_cols=()):
    return dp.build_matrix(station_df, cols, scaler, imputer, log1p_cols)


def step_sweep(X, seed=config.SEED, k_range=config.K_RANGE):
    models = cl.run_models(X, k_range, seed)
    return models, ev.sweep_table(X, models)


def step_stability(X, k_range=config.K_RANGE, seed=config.SEED):
    return ev.stability_table(X, k_range, seed)


def step_variants(station_df, cols, k, seed, base_labels):
    rows = []
    for name, opt in VARIANTS.items():
        c = [x for x in cols if x not in opt.get("drop", [])] + [x for x in opt.get("add", []) if x in station_df]
        Xv, _ = dp.build_matrix(station_df, c, opt.get("scaler", "standard"), opt.get("imputer", "median"),
                                opt.get("log1p", ()))
        lab = cl.fit_model("kmeans", Xv, k, seed).labels_
        rows.append(dict(variant=name, n_features=len(c), ari_vs_baseline=adjusted_rand_score(base_labels, lab),
                         silhouette=ev._sil(Xv, lab, "euclidean")))
    return pd.DataFrame(rows)


N_STAGES = 7


def _stage(i, text):
    """Progress marker parsed by the dashboard (dashboard/run_worker.py); harmless on the console."""
    print(f"[STAGE {i}/{N_STAGES}] {text}", flush=True)


def run_all(year=config.YEAR, skip_download=False):
    config.ensure_dirs()
    started = time.time()
    # A run is only valid once run_complete.json is written at the very end (see bottom).
    (config.METRIC_DIR / "run_complete.json").unlink(missing_ok=True)
    seed = config.SEED
    _stage(1, "Loading NOAA GSOD station files" + (" (local files only)" if skip_download else " (downloading missing files)"))
    raw = step_load(year, skip_download)
    _stage(2, "Cleaning daily observations")
    clean, clean_log = step_clean(raw)
    _stage(3, "Aggregating to station-level features")
    station_df, cols, sel_log, daily_missing, excluded, sens = step_features(clean, year)
    n_dropped = len(excluded)
    print(f"{len(station_df)} stations kept ({n_dropped} excluded for < {config.MIN_VALID_TEMP_DAYS} valid temperature days); "
          f"{len(cols)} clustering features")
    _stage(4, "Imputing and standardising features")
    X, Xi = step_matrix(station_df, cols)

    _stage(5, "Fitting K-Means and K-Medians for every K")
    models, sweep = step_sweep(X, seed)
    _stage(6, "Stability analysis and selection of K (can take several minutes)")
    stab = step_stability(X)
    k, elbow, scores = ev.select_final_k(sweep, stab, len(X))
    print(f"Selected K = {k} (elbow K = {elbow})")

    final = models[("kmeans", k)]
    labels = cl.relabel_by_size(final.labels_)
    order = np.argsort(-np.bincount(final.labels_))           # keep centroid order consistent with labels
    centers = final.cluster_centers_[order]
    alt_labels = models[("kmedians", k)].labels_

    _stage(7, "Profiling clusters, saving tables, figures and report")
    mean, median, std, z, sizes = ev.cluster_profiles(Xi, labels)
    contrib = ev.feature_contribution(Xi, labels)
    loo = ev.leave_one_feature_out(X, cols, k, seed, final.labels_)
    variants = step_variants(station_df, cols, k, seed, final.labels_)
    cross_k, km_vs_kmed = ev.cross_k_agreement(models), ev.kmeans_vs_kmedians(models)

    # ------------------------------------------------------------- save tables
    M, P = config.METRIC_DIR, config.PROFILE_DIR
    clean_log.to_csv(M / "cleaning_log.csv", index=False)
    sel_log.to_csv(M / "feature_selection_log.csv", index=False)
    sweep.to_csv(M / "k_sweep_metrics.csv", index=False)
    stab.to_csv(M / "stability_by_k.csv", index=False)
    scores.to_csv(M / "k_selection_scores.csv", index=False)
    variants.to_csv(M / "preprocessing_variants.csv", index=False)
    contrib.to_csv(M / "feature_contribution.csv", index=False)
    loo.to_csv(M / "leave_one_feature_out.csv", index=False)
    cross_k.to_csv(M / "cross_k_agreement.csv", index=False)
    excluded.to_csv(M / "excluded_stations.csv", index=False)
    sens.to_csv(M / "completeness_threshold_sensitivity.csv", index=False)
    km_vs_kmed.to_csv(M / "kmeans_vs_kmedians.csv", index=False)
    mean.to_csv(P / "cluster_profile_mean.csv"); median.to_csv(P / "cluster_profile_median.csv")
    std.to_csv(P / "cluster_profile_std.csv"); z.to_csv(P / "cluster_profile_zscores.csv")
    sizes.to_csv(P / "cluster_sizes.csv")
    out = station_df.copy()
    out["cluster"] = labels
    out.to_csv(config.PROC_DIR / "station_features_with_clusters.csv")
    pd.DataFrame(X, index=station_df.index, columns=cols).to_csv(config.PROC_DIR / "station_features_scaled.csv")
    summary = dict(year=year, n_daily_records=int(len(clean)), n_stations=int(len(station_df)),
                   n_stations_dropped=int(n_dropped), min_valid_temp_days=config.MIN_VALID_TEMP_DAYS,
                   prcp_min_coverage=config.PRCP_MIN_COVERAGE,
                   n_prcp_unavailable=int(station_df["precip_observed_total_mm"].isna().sum()), n_features=len(cols), features=cols, selected_k=int(k),
                   elbow_k=int(elbow), seed=seed, daily_missing_pct=daily_missing.round(2).to_dict())
    (M / "run_summary.json").write_text(json.dumps(summary, indent=2))

    # ------------------------------------------------------------- figures
    st_missing = station_df[fe.WEATHER_FEATURES].isna().mean() * 100
    viz.plot_missing(daily_missing, st_missing)
    viz.plot_distributions(station_df, cols)
    viz.plot_corr(station_df, [c for c in fe.WEATHER_FEATURES if c in station_df.columns])
    viz.plot_boxplots(X, cols)
    viz.plot_elbow(sweep, elbow, k)
    viz.plot_silhouette(sweep)
    viz.plot_db(sweep)
    viz.plot_pca(X, labels, k, centers, alt_labels)
    viz.plot_sizes(labels)
    viz.plot_profile_heatmap(z)
    viz.plot_cluster_comparison(Xi, labels, list(contrib.feature.head(6)))
    viz.plot_stability(stab, variants, cross_k, k)
    viz.plot_geo(station_df, labels)

    from src import report_generator as rg
    rg.generate_report()
    (config.METRIC_DIR / "run_complete.json").write_text(json.dumps(
        dict(run_id=os.environ.get("WSBG_RUN_ID", "manual"), year=year, selected_k=int(k),
             n_stations=int(len(station_df)), completed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
             duration_seconds=round(time.time() - started, 1)), indent=2))
    print("Done. See outputs/ and final_report.md", flush=True)
    return summary


if __name__ == "__main__":
    run_all()
