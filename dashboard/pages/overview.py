import streamlit as st

from dashboard import charts, data_access as da, ui


def render():
    ui.title("Overview", "Weather-Station Behaviour Grouping for Automated Data Quality Assessment")
    with st.container():
        st.markdown("#### Analysis control")
        ui.run_panel("overview")
    b = ui.require_results()
    if b is None:
        return
    c, comp = da.best_config(b), da.completeness_stats(b)
    cols = st.columns(4)
    ui.kpi(cols[0], "Stations analysed", comp["n_retained"], f"of {comp['n_total']} with data", accent=True)
    ui.kpi(cols[1], "Clusters (selected K)", c["k"], f"elbow suggests K = {c['elbow_k']}", accent=True)
    ui.kpi(cols[2], "Silhouette (Euclidean)", f"{c['silhouette']:.3f}", "K-Means, selected K", accent=True)
    ui.kpi(cols[3], "Davies-Bouldin", f"{c['davies_bouldin']:.3f}", "lower is better", accent=True)
    st.write("")
    cols = st.columns(4)
    ui.kpi(cols[0], "Stations retained", f"{comp['pct_retained']:.0f}%", f"{comp['n_excluded']} excluded (< {comp['min_valid_days']} valid temp. days)")
    ui.kpi(cols[1], "Median valid temp. days", f"{comp['median_valid_temp_days']:.0f}", "retained stations")
    cov = comp.get("median_prcp_coverage")
    ui.kpi(cols[2], "Median precip. coverage", "n/a" if cov is None else f"{cov:.0%}", f"{comp['n_prcp_unavailable']} stations below {comp['prcp_min_coverage']:.0%} (unavailable)")
    ui.kpi(cols[3], "Stability (ARI)", f"{c['seed_ari']:.2f} / {c['boot_ari']:.2f}", "seeds / 80% subsamples")
    st.write("")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Best clustering configuration")
        st.markdown(f"**K-Means, K = {c['k']}** (standard scikit-learn, Euclidean). Chosen with a composite of silhouette, "
                    f"Davies-Bouldin, two stability measures and a minimum cluster size, not by silhouette alone. "
                    f"Cluster sizes: {c['min_size']}–{c['max_size']} stations. WCSS = {c['wcss']:.1f}.")
        st.markdown("#### Important findings")
        st.markdown("".join(f'<div class="finding">{f}</div>' for f in da.findings(b)), unsafe_allow_html=True)
    with right:
        df, evr = da.pca_frame(b)
        ui.show_plotly(charts.pca_scatter(df, evr))
    run = b.get("marker")
    if run:
        st.caption(f"Analysis completed {run.get('completed_at')} (year {run.get('year')}, run id {run.get('run_id')}).")
