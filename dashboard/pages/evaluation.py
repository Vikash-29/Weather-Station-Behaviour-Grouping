import streamlit as st

from dashboard import charts, data_access as da, ui

LIMITS = [
    "One calendar year of data: clusters may differ in other years.",
    "Only stations present in NOAA GSOD/ISD with enough valid temperature days are included; this is not a random sample of the network.",
    "Stations with fewer than the minimum valid temperature days are excluded, so poorly reporting stations are never clustered (selection bias).",
    "Precipitation is never scaled to a full year; stations with low coverage have it marked unavailable and median-imputed.",
    "Fog/thunder indicators are lower bounds: in GSOD a zero means no event OR not reported.",
    "K-Means favours compact, similar-sized groups; climate regimes can be continuous. ANOVA/Kruskal p-values are descriptive only.",
    "Clusters are exploratory groupings, not permanent station categories and not statements about station reliability.",
]


def render():
    ui.title("Evaluation", "Validation metrics, stability and sensitivity to preprocessing")
    b = ui.require_results()
    if b is None:
        return
    c = da.best_config(b)
    cols = st.columns(4)
    ui.kpi(cols[0], "Silhouette (Euclid.)", f"{c['silhouette']:.3f}", f"Manhattan: {c['silhouette_manhattan']:.3f}")
    ui.kpi(cols[1], "Davies-Bouldin", f"{c['davies_bouldin']:.3f}", "lower is better")
    ui.kpi(cols[2], "WCSS", f"{c['wcss']:.1f}", "K-Means objective")
    ui.kpi(cols[3], "Selected K", c["k"], f"elbow K = {c['elbow_k']}")
    st.write("")
    st.markdown("#### Metrics for every K")
    m = st.radio("Algorithm", ["kmeans", "kmedians"], horizontal=True,
                 format_func=lambda x: "K-Means (Euclidean)" if x == "kmeans" else "K-Medians (Manhattan)")
    ui.show_df(b["sweep"][b["sweep"].method == m].drop(columns=["method"]).round(4))
    st.markdown("#### Selection of K (composite score)")
    st.caption("K = 2 and K values with a very small cluster are ineligible; the score averages min-max scaled silhouette, "
               "Davies-Bouldin, seed stability and subsample stability.")
    ui.show_df(b["scores"].round(4))
    st.info(da.findings(b)[2])
    st.markdown("#### Stability analysis")
    l, r = st.columns(2)
    with l:
        ui.show_plotly(charts.stability(b["stab"], c["k"]))
    with r:
        ui.show_plotly(charts.cross_k(b["cross_k"]))
    ui.show_df(b["stab"].round(4))
    st.markdown("#### Preprocessing and feature-set comparisons")
    ui.show_plotly(charts.variants(b["variants"], c["k"]))
    ui.show_df(b["variants"].round(4))
    t1, t2, t3 = st.tabs(["K-Means vs K-Medians", "Leave-one-feature-out", "Feature contribution"])
    with t1:
        ui.show_df(b["kk"].round(4))
    with t2:
        ui.show_df(b["loo"].round(4))
    with t3:
        ui.show_df(b["contrib"].round(4))
    st.markdown("#### Data completeness and station exclusion")
    l, r = st.columns(2)
    with l:
        st.caption("Sensitivity of the exclusion rule")
        ui.show_df(b["sens"])
    with r:
        st.caption(f"Excluded stations ({len(b['excluded'])})")
        ui.show_df(b["excluded"])
    st.markdown("#### Model limitations")
    st.markdown("".join(f'<div class="finding">{x}</div>' for x in LIMITS), unsafe_allow_html=True)
