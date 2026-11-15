import io
import re
import zipfile

import streamlit as st

from dashboard import data_access as da, ui
from src import config

METHOD = """
**Data.** NOAA NCEI Global Surface Summary of the Day (GSOD), Indian stations (FIPS `IN`), one calendar year. Daily observations are aggregated to **station-level** features, and stations are the clustering unit.

**Preprocessing.** GSOD sentinels become missing; units converted to SI; implausible or inconsistent records set to missing (plausible extremes kept). Precipitation flag **I** is missing; flag **H** (0 reported but hourly precipitation seen) is ambiguous and excluded from precipitation calculations. Precipitation features are computed only on valid days and only when coverage is high enough (never scaled to 365 days). FRSHTT zeros are not treated as confirmed absence. Stations with too few valid temperature days are excluded and counted.

**Clustering.** Median imputation, z-score standardisation, K-Means (Euclidean) for K = 2-10, K-Medians for the Manhattan comparison, WCSS / silhouette / Davies-Bouldin, elbow, PCA, stability (seeds, subsamples, neighbouring K, preprocessing variants) and feature contribution analysis.
"""


def _strip_images(md):
    return re.sub(r"!\[[^\]]*\]\([^)]*\)\n*", "", md)


def render():
    ui.title("Report", "Methodology, findings and downloads")
    st.markdown("#### Project methodology")
    st.markdown(METHOD)
    b = ui.require_results()
    if b is None:
        return
    st.markdown("#### Actual results and findings")
    st.markdown("".join(f'<div class="finding">{f}</div>' for f in da.findings(b)), unsafe_allow_html=True)
    st.markdown("#### Downloads")
    files = [p for p in da.required_paths() if p.suffix == ".csv"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.name)
    c1, c2 = st.columns(2)
    c1.download_button("Download all CSV files (ZIP)", buf.getvalue(), "weather_station_results_csv.zip", "application/zip")
    rep = config.ROOT / "final_report.md"
    c2.download_button("Download final report (Markdown)", rep.read_bytes(), "final_report.md", "text/markdown")
    with st.expander("Individual CSV files"):
        for p in files:
            st.download_button(p.name, p.read_bytes(), p.name, "text/csv", key=f"dl_{p.name}")
    with st.expander("Preview final report"):
        st.markdown(_strip_images(rep.read_text(encoding="utf-8")))
