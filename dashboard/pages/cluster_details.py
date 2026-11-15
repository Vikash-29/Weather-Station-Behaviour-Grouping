import streamlit as st

from dashboard import charts, data_access as da, ui
from src import report_generator as rg


def render():
    ui.title("Cluster Details", "Station assignments, profiles and interpretation")
    b = ui.require_results()
    if b is None:
        return
    st.markdown("#### Station assignments")
    tbl = da.station_table(b)
    f1, f2 = st.columns([2, 3])
    clusters = sorted(tbl["cluster"].unique())
    pick = f1.multiselect("Cluster", clusters, default=clusters)
    q = f2.text_input("Search station ID or name")
    v = tbl[tbl["cluster"].isin(pick)]
    if q:
        v = v[v["STATION"].str.contains(q, case=False, na=False) | v["name"].astype(str).str.contains(q, case=False, na=False)]
    ui.show_df(v)
    st.download_button("Download station assignments (CSV)", tbl.to_csv(index=False).encode("utf-8"),
                       "station_cluster_assignments.csv", "text/csv")
    st.markdown("#### Cluster profiles")
    ui.show_plotly(charts.profile_heatmap(b["z"]))
    t1, t2, t3 = st.tabs(["Means", "Medians", "Sizes"])
    with t1:
        ui.show_df(b["mean"].T.rename(index=da.label).round(3).reset_index().rename(columns={"index": "feature"}))
    with t2:
        ui.show_df(b["median"].T.rename(index=da.label).round(3).reset_index().rename(columns={"index": "feature"}))
    with t3:
        ui.show_df(b["sizes"].reset_index())
    st.markdown("#### Feature-wise comparison")
    feats = list(b["mean"].columns)
    f = st.selectbox("Feature", feats, format_func=da.label)
    l, r = st.columns([3, 2])
    with l:
        ui.show_plotly(charts.feature_by_cluster(b["stations"], f))
    with r:
        ui.show_plotly(charts.eta_bar(b["contrib"]))
    st.markdown("#### Interpretation of each cluster")
    st.caption("Generated from the computed profiles. Descriptive only: not a ranking of weather quality or station reliability.")
    for para in rg.cluster_narrative().split("\n\n"):
        st.markdown(f'<div class="panel">{para.replace("**", "")}</div>', unsafe_allow_html=True)
