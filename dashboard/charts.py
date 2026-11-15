"""Plotly figures built from the real output tables."""
import plotly.express as px
import plotly.graph_objects as go

from dashboard.data_access import label

PAL = px.colors.qualitative.Safe
BASE = dict(template="plotly_white", margin=dict(l=40, r=20, t=60, b=40), font=dict(size=13))


def _fin(fig, title, x=None, y=None, h=420):
    fig.update_layout(title=title, height=h, **BASE)
    if x: fig.update_xaxes(title=x)
    if y: fig.update_yaxes(title=y)
    return fig


def pca_scatter(df, evr):
    fig = px.scatter(df.sort_values("cluster"), x="PC1", y="PC2", color="cluster", hover_data=["station", "name"],
                     color_discrete_sequence=PAL)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="#1e293b")))
    return _fin(fig, "PCA projection of stations (K-Means clusters)", f"PC1 ({evr[0]*100:.1f}% of variance)",
                f"PC2 ({evr[1]*100:.1f}% of variance)", 520)


def cluster_distribution(sizes):
    d = sizes.reset_index().rename(columns={sizes.index.name or "index": "cluster"})
    d["cluster"] = d["cluster"].astype(str)
    fig = px.bar(d, x="cluster", y="n_stations", color="cluster", text="n_stations", color_discrete_sequence=PAL)
    fig.update_layout(showlegend=False)
    return _fin(fig, "Cluster sizes", "Cluster", "Number of stations")


def elbow(sweep, elbow_k, k):
    d = sweep[sweep.method == "kmeans"]
    fig = go.Figure(go.Scatter(x=d.k, y=d.objective, mode="lines+markers", name="WCSS"))
    fig.add_vline(x=elbow_k, line_dash="dash", line_color="#dd8452", annotation_text=f"elbow K={elbow_k}")
    fig.add_vline(x=k, line_dash="dot", line_color="#15803d", annotation_text=f"selected K={k}", annotation_position="bottom right")
    return _fin(fig, "Elbow curve: WCSS vs K", "Number of clusters (K)", "WCSS (standardised units²)")


def _lines(sweep, cols, title, ytitle):
    fig = go.Figure()
    for method, col, name in cols:
        d = sweep[sweep.method == method]
        fig.add_trace(go.Scatter(x=d.k, y=d[col], mode="lines+markers", name=name))
    return _fin(fig, title, "Number of clusters (K)", ytitle)


def silhouette(sweep):
    return _lines(sweep, [("kmeans", "silhouette_euclidean", "K-Means, Euclidean"),
                          ("kmeans", "silhouette_manhattan", "K-Means labels, Manhattan (evaluation only)"),
                          ("kmedians", "silhouette_manhattan", "K-Medians, Manhattan")],
                  "Silhouette score vs K (higher is better)", "Mean silhouette")


def davies_bouldin(sweep):
    return _lines(sweep, [("kmeans", "davies_bouldin", "K-Means"), ("kmedians", "davies_bouldin", "K-Medians")],
                  "Davies-Bouldin index vs K (lower is better)", "Davies-Bouldin index")


def euclid_vs_manhattan(sweep):
    fig = go.Figure()
    km, kd = sweep[sweep.method == "kmeans"], sweep[sweep.method == "kmedians"]
    fig.add_bar(x=km.k, y=km.silhouette_euclidean, name="K-Means · Euclidean silhouette")
    fig.add_bar(x=km.k, y=km.silhouette_manhattan, name="K-Means · Manhattan silhouette (same labels)")
    fig.add_bar(x=kd.k, y=kd.silhouette_manhattan, name="K-Medians · Manhattan silhouette")
    fig.update_layout(barmode="group")
    return _fin(fig, "Euclidean vs Manhattan comparison", "Number of clusters (K)", "Mean silhouette")


def profile_heatmap(z):
    zz = z.T.rename(index=label)
    fig = px.imshow(zz.values, x=[f"Cluster {c}" for c in zz.columns], y=list(zz.index), color_continuous_scale="RdBu_r",
                    zmin=-2.5, zmax=2.5, aspect="auto", text_auto=".1f")
    fig.update_layout(coloraxis_colorbar_title="z (vs all stations)")
    return _fin(fig, "Cluster profiles: cluster mean relative to all stations (std. deviations)", None, None,
                max(420, 28 * len(zz)))


def feature_by_cluster(stations, feature):
    d = stations[["cluster", feature]].dropna().copy()
    d["cluster"] = d["cluster"].astype(str)
    fig = px.box(d.sort_values("cluster"), x="cluster", y=feature, color="cluster", points="all", color_discrete_sequence=PAL)
    fig.update_layout(showlegend=False)
    return _fin(fig, f"{label(feature)} by cluster", "Cluster", label(feature))


def eta_bar(contrib):
    d = contrib.sort_values("eta_squared")
    fig = go.Figure(go.Bar(x=d.eta_squared, y=[label(f) for f in d.feature], orientation="h"))
    return _fin(fig, "Between-cluster separation per feature (eta²)", "eta² (share of variance explained by cluster)", None,
                max(380, 24 * len(d)))


def stability(stab, k):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stab.k, y=stab.seed_ari_mean, mode="lines+markers", name="Random initialisation",
                             error_y=dict(type="data", array=stab.seed_ari_std)))
    fig.add_trace(go.Scatter(x=stab.k, y=stab.boot_ari_mean, mode="lines+markers", name="80% subsample refit",
                             error_y=dict(type="data", array=stab.boot_ari_std)))
    fig.add_vline(x=k, line_dash="dot", line_color="#15803d", annotation_text=f"selected K={k}")
    fig.update_yaxes(range=[0, 1.05])
    return _fin(fig, "Stability vs K (Adjusted Rand Index)", "Number of clusters (K)", "ARI (1 = identical)")


def cross_k(ck):
    fig = go.Figure(go.Bar(x=[f"{a}→{b}" for a, b in zip(ck.k_from, ck.k_to)], y=ck.ari))
    fig.update_yaxes(range=[0, 1.05])
    return _fin(fig, "Agreement between consecutive K", "K → K+1", "Adjusted Rand Index")


def variants(var, k):
    d = var.sort_values("ari_vs_baseline")
    fig = go.Figure(go.Bar(x=d.ari_vs_baseline, y=d.variant, orientation="h", text=d.ari_vs_baseline.round(2)))
    fig.update_xaxes(range=[0, 1.05])
    return _fin(fig, f"Preprocessing variants vs baseline (K={k})", "Adjusted Rand Index vs baseline", None)
