from dashboard import charts, data_access as da, ui
import streamlit as st


def render():
    ui.title("Visualizations", "Interactive charts built from the saved analysis outputs")
    b = ui.require_results()
    if b is None:
        return
    c = da.best_config(b)
    df, evr = da.pca_frame(b)
    l, r = st.columns([3, 2])
    with l:
        ui.show_plotly(charts.pca_scatter(df, evr))
        st.caption("PCA is for visual inspection only; clustering is done in the full standardised feature space.")
    with r:
        ui.show_plotly(charts.cluster_distribution(b["sizes"]))
    l, r = st.columns(2)
    with l:
        ui.show_plotly(charts.elbow(b["sweep"], c["elbow_k"], c["k"]))
    with r:
        ui.show_plotly(charts.silhouette(b["sweep"]))
    l, r = st.columns(2)
    with l:
        ui.show_plotly(charts.davies_bouldin(b["sweep"]))
    with r:
        ui.show_plotly(charts.euclid_vs_manhattan(b["sweep"]))
    st.markdown('<div class="panel"><b>Distance metrics.</b> Standard K-Means optimises squared Euclidean distance and does not support '
                'Manhattan distance. "Manhattan silhouette of K-Means labels" only changes how the <i>same</i> clustering is evaluated; '
                '<b>K-Medians</b> is a different algorithm (Manhattan assignment, median centres) with an L1 objective that is not '
                'comparable to WCSS.</div>', unsafe_allow_html=True)
