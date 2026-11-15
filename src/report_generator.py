"""Builds final_report.md and short narratives from the saved result files (no hard-coded numbers)."""
import json

import pandas as pd

from src import config
from src.feature_engineering import FEATURE_LABELS

M, P = config.METRIC_DIR, config.PROFILE_DIR


def _lab(c):
    return FEATURE_LABELS.get(c, c)


def _md(df, floatfmt=3):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.values)


def metric_agreement_text():
    s = json.loads((M / "run_summary.json").read_text())
    sw = pd.read_csv(M / "k_sweep_metrics.csv")
    km = sw[sw.method == "kmeans"]
    k_sil = int(km.loc[km.silhouette_euclidean.idxmax(), "k"])
    k_db = int(km.loc[km.davies_bouldin.idxmin(), "k"])
    txt = (f"Best K by Euclidean silhouette: **{k_sil}**; best K by Davies-Bouldin: **{k_db}**; "
           f"elbow K (chord method): **{s['elbow_k']}**; selected K (composite incl. stability): **{s['selected_k']}**. ")
    if len({k_sil, k_db, s["elbow_k"]}) == 1:
        txt += "The three indices point to the same K."
    else:
        txt += ("The indices do not agree. This is expected: WCSS always decreases with K and its elbow is a "
                "visual/heuristic criterion, silhouette rewards compact and well separated groups, and "
                "Davies-Bouldin penalises overlap between the most similar pair of clusters. The final K was "
                "therefore chosen with the composite score (silhouette, Davies-Bouldin, two stability measures, "
                "minimum cluster size) rather than any single index.")
    return txt


def cluster_narrative(top_n=4):
    z = pd.read_csv(P / "cluster_profile_zscores.csv", index_col=0)
    mean = pd.read_csv(P / "cluster_profile_mean.csv", index_col=0)
    sizes = pd.read_csv(P / "cluster_sizes.csv", index_col=0)
    st = pd.read_csv(config.PROC_DIR / "station_features_with_clusters.csv", index_col=0)
    overall = st[mean.columns].mean()
    out = []
    for c in z.index:
        top = z.loc[c].abs().sort_values(ascending=False).head(top_n).index
        bits = [f"{_lab(f)}: {mean.loc[c, f]:.2f} vs {overall[f]:.2f} overall (z = {z.loc[c, f]:+.2f})" for f in top]
        sub = st[st.cluster == c]
        names = ", ".join(str(n).title() for n in sub["name"].dropna().head(5))
        out.append(f"**Cluster {c}** — {int(sizes.loc[c, 'n_stations'])} stations ({sizes.loc[c, 'share']:.0%}). "
                   f"Most distinguishing features: " + "; ".join(bits) + ". "
                   f"Mean latitude {sub.latitude.mean():.1f}°N, mean elevation {sub.elevation_m.mean():.0f} m "
                   f"(descriptive only, not used for clustering). Example stations: {names}.")
    return "\n\n".join(out)


def exclusion_text():
    s = json.loads((M / "run_summary.json").read_text())
    ex = pd.read_csv(M / "excluded_stations.csv")
    sens = pd.read_csv(M / "completeness_threshold_sensitivity.csv")
    st = pd.read_csv(config.PROC_DIR / "station_features_with_clusters.csv", index_col=0)
    n_total = s["n_stations"] + s["n_stations_dropped"]
    t = (f"Of {n_total} stations with daily records, **{s['n_stations_dropped']}** were excluded because they reported a valid "
         f"mean temperature on fewer than {s['min_valid_temp_days']} days ({s['n_stations']} retained).\n\n")
    if len(ex):
        t += (f"Excluded stations: median valid temperature days = {ex.n_valid_temp_days.median():.0f} "
              f"(range {ex.n_valid_temp_days.min():.0f}–{ex.n_valid_temp_days.max():.0f}); mean latitude {ex.latitude.mean():.1f}°N "
              f"vs {st.latitude.mean():.1f}°N retained; mean elevation {ex.elevation_m.mean():.0f} m vs {st.elevation_m.mean():.0f} m retained "
              f"(full list: `outputs/metrics/excluded_stations.csv`).\n\n")
    t += "Sensitivity of the rule to the threshold:\n\n" + _md(sens, 0) + "\n\n"
    t += ("**Selection bias and implications.** The rule keeps only stations that report temperature almost every day. "
          "Clusters therefore describe *well-reporting* stations only; stations with the poorest completeness — the ones an "
          "automated data-quality screen would most want to flag — are removed before clustering and cannot be assigned to any "
          "group. Completeness correlates with network type, region, instrumentation and station status, so the retained set is not a "
          "random sample and the clusters should not be generalised to the whole network. Completeness itself is not part of the baseline "
          "features; the variant that adds data-completeness features shows how much the grouping changes when it is included. A data-quality "
          "application would need either a separate completeness screen before clustering or a design that retains incomplete stations.")
    return t


def generate_report(path=None):
    path = path or config.ROOT / "final_report.md"
    s = json.loads((M / "run_summary.json").read_text())
    sw = pd.read_csv(M / "k_sweep_metrics.csv")
    stab = pd.read_csv(M / "stability_by_k.csv")
    sel = pd.read_csv(M / "k_selection_scores.csv")
    var = pd.read_csv(M / "preprocessing_variants.csv")
    con = pd.read_csv(M / "feature_contribution.csv")
    loo = pd.read_csv(M / "leave_one_feature_out.csv")
    ck = pd.read_csv(M / "cross_k_agreement.csv")
    kk = pd.read_csv(M / "kmeans_vs_kmedians.csv")
    clog = pd.read_csv(M / "cleaning_log.csv")
    flog = pd.read_csv(M / "feature_selection_log.csv")
    k = s["selected_k"]
    sk = sw[(sw.method == "kmeans") & (sw.k == k)].iloc[0]
    sizes = pd.read_csv(P / "cluster_sizes.csv", index_col=0)
    best_stab = stab[stab.k == k].iloc[0]
    fig = lambda f, cap: f"![{cap}](outputs/figures/{f})\n\n*{cap}*\n"
    sweep_view = sw[["method", "k", "objective", "silhouette_euclidean", "silhouette_manhattan", "davies_bouldin", "min_size"]]

    r = f"""# Weather-Station Behaviour Grouping for Automated Data Quality Assessment

*Auto-generated from the executed analysis (`python run_pipeline.py`). All numbers below are read from files in `outputs/`.*

## Abstract
Daily observations from {s['n_stations']} Indian weather stations (NOAA Global Surface Summary of the Day, calendar year {s['year']}; {s['n_daily_records']:,} cleaned station-days) were aggregated to station-level statistical features ({s['n_features']} clustering features) and grouped with K-Means. K = {s['selected_k']} was selected from the tested range K = {config.K_RANGE[0]}–{config.K_RANGE[-1]} using silhouette, Davies-Bouldin, the elbow of WCSS, cluster size and two stability measures. At the selected K the Euclidean silhouette is {sk.silhouette_euclidean:.3f} and the Davies-Bouldin index is {sk.davies_bouldin:.3f}. The clusters are exploratory groupings of annual observed behaviour and are not permanent categories or judgements of station reliability.

## 1. Introduction
Weather-station networks produce heterogeneous records: stations differ in climate, altitude, instrumentation and reporting completeness. Grouping stations by observed behaviour can support screening and comparison when assessing data quality, without labelled reference categories.

## 2. Problem statement
Given daily records of many stations, can stations be grouped by similarity of their observed weather behaviour in a reproducible, validated way, and are the groups stable and interpretable?

## 3. Objectives
1. Prepare and preprocess an open NOAA station dataset for clustering.
2. Build a K-Means workflow over multiple K with Euclidean/Manhattan comparison, WCSS, silhouette, Davies-Bouldin, elbow and PCA.
3. Interpret and validate the clusters (profiles, stability, feature contributions, limitations).
4. Deliver a reproducible end-to-end analysis.

## 4. Dataset
NOAA NCEI **Global Surface Summary of the Day (GSOD)**, derived from the Integrated Surface Hourly dataset. Landing page: https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00516 ; files: {config.GSOD_BASE}/{s['year']}/<station>.csv ; station metadata: {config.ISD_HISTORY_URL} ; documentation: https://www.ncei.noaa.gov/data/global-summary-of-the-day/doc/readme.txt . Stations of India (FIPS `{config.COUNTRY_FIPS}`) whose ISD record spans the whole year were downloaded by `src/data_preprocessing.py`. Each row is one station-day; variables include mean/max/min temperature, dew point, sea-level and station pressure, visibility, wind speed, peak wind, precipitation and a fog/rain/snow/hail/thunder/tornado flag string. The **clustering unit is the station**, so daily records are aggregated.

## 5. Methodology
Pipeline: download → clean daily records → derive variables → aggregate per station → select features → impute → scale → K-Means/K-Medians for K = {config.K_RANGE[0]}–{config.K_RANGE[-1]} → validation → selection → profiling → stability → feature contribution. Random seed: {s['seed']}.

## 6. Data preprocessing
Every cleaning step and its count:

{_md(clog, 0)}

Decisions: GSOD missing-value sentinels were converted to NaN; units converted to SI; physically implausible values and internally inconsistent records (max < min, dew point > temperature) set to missing. Precipitation attribute **I** (no precipitation report) is treated as missing. Attribute **H** (station reported 0 for the day, but hourly observations reported precipitation) is handled separately: per NOAA it is incomplete data, so it is *not* treated as a confirmed dry day and is excluded from precipitation calculations; its frequency per station is kept as the data-quality feature `dq_prcp_flag_h_frac`. Outliers that are physically plausible were **kept** (genuine extremes carry weather information) and screened with boxplots (Fig. 4). Daily missing percentages: Fig. 1.

{fig('01_missing_values.png', 'Figure 1. Missing values')}

## 7. Feature engineering and aggregation
Derived daily: relative humidity (Magnus formula from temperature and dew point), diurnal temperature range, dew-point depression, fog/thunder/rain-day indicators. Per station: mean, standard deviation and extremes of the daily series, observed precipitation total, rainy-day fraction and intensity, and reported fog/thunder frequency.

**Precipitation.** A partial precipitation record is *not* scaled up to 365 days, because nothing shows that missing days are representative (missing days may be unreported dry days or wet days with accumulated reports). For each station, precipitation coverage = valid PRCP days / calendar days. The precipitation features (`precip_observed_total_mm`, `rainy_day_frac`, `precip_intensity_mm`) are computed from valid days only and are set to **unavailable (NaN)** when coverage is below {config.PRCP_MIN_COVERAGE:.0%}; {s['n_prcp_unavailable']} of {s['n_stations']} retained stations are unavailable. Even above the threshold the total is an *observed total on reported days*, a lower bound rather than an annual estimate. Features unavailable for more than {config.MAX_FEATURE_MISSING:.0%} of stations are dropped; otherwise remaining gaps are median-imputed, which must be kept in mind when interpreting precipitation differences.

**Event indicators.** In GSOD the FRSHTT digits are 1 = event reported and 0 = *no event or not reported*; zero is not confirmed absence. The features are named `fog_reported_frac` and `thunder_reported_frac` and are interpreted as lower bounds that also reflect station reporting practice. The variant analysis includes a run without them.

**Station completeness rule.** Stations with fewer than {config.MIN_VALID_TEMP_DAYS} valid temperature days were removed ({s['n_stations_dropped']} stations); see Section 7.1. Latitude, longitude, elevation and station name were **excluded from clustering** (identifiers/context) and used only for interpretation. No categorical variables remain after aggregation, so no encoding was needed. Missing station-level values were median-imputed and features were standardised (zero mean, unit variance) because K-Means is distance-based.

### 7.1 Effect of the completeness exclusion
{exclusion_text()}

Feature selection log (features missing for > {config.MAX_FEATURE_MISSING:.0%} of stations or with |r| > {config.CORR_PRUNE_THRESHOLD} to an earlier feature are dropped):

{_md(flog)}

{fig('02_feature_distributions.png', 'Figure 2. Feature distributions')}
{fig('03_correlation_heatmap.png', 'Figure 3. Correlation heatmap')}
{fig('04_boxplots_outliers.png', 'Figure 4. Boxplots of standardised features')}

## 8. Clustering methodology and distance metrics
Standard K-Means (scikit-learn, `n_init=20`) minimises the within-cluster sum of squares (WCSS) with **Euclidean** distance. It does **not** support Manhattan distance. Manhattan distance was therefore used in two clearly separate ways: (a) **evaluation only** — Manhattan silhouette computed on the K-Means labels (the clustering itself is unchanged); (b) **a different algorithm** — K-Medians (own implementation in `src/clustering.py`), which assigns by Manhattan distance and updates centres with medians. Davies-Bouldin is computed with scikit-learn (Euclidean, cluster means) for both algorithms. The K-Medians objective is an L1 cost and is not comparable to WCSS.

## 9. Evaluation across K
{_md(sweep_view)}

{fig('05_elbow_wcss.png', 'Figure 5. Elbow curve')}
{fig('06_silhouette.png', 'Figure 6. Silhouette vs K')}
{fig('07_davies_bouldin.png', 'Figure 7. Davies-Bouldin vs K')}

{metric_agreement_text()}

K-Means vs K-Medians label agreement (ARI):

{_md(kk)}

### Composite selection table (K-Means)
{_md(sel)}

K = 2 is excluded as a final choice because a two-way split is rarely informative; clusters smaller than max(3, {config.MIN_CLUSTER_FRAC:.0%} of stations) make a K ineligible.

## 10. PCA visualisation
{fig('08_pca_clusters.png', 'Figure 8. PCA projection of the selected clustering (left) and K-Medians (right)')}
PCA is used only for visual inspection; clustering was done in the full standardised feature space. Overlap in two dimensions does not imply overlap in the full space.

## 11. Results and cluster interpretation (K = {k})
Cluster sizes:

{_md(sizes.reset_index(), 3)}

{fig('09_cluster_sizes.png', 'Figure 9. Cluster sizes')}
{fig('10_cluster_profile_heatmap.png', 'Figure 10. Cluster profile heatmap')}
{fig('11_cluster_feature_comparison.png', 'Figure 11. Cluster comparison')}
{fig('13_station_map_by_cluster.png', 'Figure 13. Station locations by cluster (supplementary)')}

{cluster_narrative()}

*These descriptions are generated automatically from the calculated profiles (`outputs/cluster_profiles/`). They describe how groups differ in observed statistics; they do not rank weather quality or station reliability.*

### Feature contributions
Between-cluster separation (eta², ANOVA F, Kruskal-Wallis H; p-values are descriptive because clusters were derived from the same data):

{_md(con)}

Leave-one-feature-out (ARI of the clustering without the feature vs the baseline; lower = more influential):

{_md(loo)}

## 12. Stability analysis
At the selected K: seed-to-seed ARI = {best_stab.seed_ari_mean:.3f} ± {best_stab.seed_ari_std:.3f}; subsample-refit ARI = {best_stab.boot_ari_mean:.3f} ± {best_stab.boot_ari_std:.3f}.

{_md(stab)}

Agreement between consecutive K:

{_md(ck)}

Preprocessing/feature-set variants (ARI vs baseline at K = {k}):

{_md(var)}

{fig('12_stability.png', 'Figure 12. Stability analysis')}

## 13. Limitations and potential bias
- One year ({s['year']}) of data; inter-annual variability is not captured and clusters may change in another year.
- Only stations present in NOAA's ISD/GSOD with sufficient reporting are included; the network is not a random sample of Indian stations and is denser in some regions.
- Median imputation and standardisation affect distances; the variant analysis shows the sensitivity at K = {k}.
- K-Means favours compact, similarly sized, roughly spherical groups; real climate regimes can be continuous.
- Features derived from GSOD (e.g. relative humidity from dew point) inherit upstream measurement and rounding errors; pressure features are strongly tied to elevation.
- ANOVA/Kruskal p-values are not valid hypothesis tests after clustering.

## 14. Responsible use
Clusters are exploratory groupings of observed behaviour in one year, not permanent station categories, and must not be read as measures of weather quality, instrument accuracy or station reliability. Any operational data-quality use requires expert review, metadata (instrument type, siting), and validation against independent references.

## 15. Conclusion
The workflow produced a validated K = {k} grouping of {s['n_stations']} stations; the evidence (metrics, stability, interpretability) is summarised in Sections 9–12. See `README.md` to reproduce.

## 16. References
- NOAA NCEI, Global Surface Summary of the Day (GSOD). https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00516
- NOAA NCEI, Integrated Surface Database station history. {config.ISD_HISTORY_URL}
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825–2830.
- Rousseeuw, P. J. (1987). Silhouettes. J. Comput. Appl. Math. 20, 53–65.
- Davies, D. L. & Bouldin, D. W. (1979). A cluster separation measure. IEEE TPAMI 1(2), 224–227.
- Hubert, L. & Arabie, P. (1985). Comparing partitions. J. Classification 2, 193–218.
- Bradley, P. S., Mangasarian, O. L. & Street, W. N. (1997). Clustering via concave minimization (K-Medians).
"""
    path.write_text(r, encoding="utf-8")
    return path
