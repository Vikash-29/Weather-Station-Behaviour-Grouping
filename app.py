"""Weather-Station Behaviour Grouping dashboard.  Start with:  streamlit run app.py  (or start_app.bat)"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Weather Station Behaviour Grouping", page_icon="🌦️", layout="wide")

from dashboard import runner, ui  # noqa: E402
from dashboard.pages import cluster_details, evaluation, overview, report, visualizations  # noqa: E402

PAGES = {"Overview": overview, "Visualizations": visualizations, "Cluster Details": cluster_details,
         "Evaluation": evaluation, "Report": report}

ui.inject_css()
with st.sidebar:
    st.markdown('<div class="brand">🌦️ Station Grouping</div><div class="brand-sub">Weather-station behaviour clustering · NOAA GSOD</div>',
                unsafe_allow_html=True)
    page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    s = runner.read_status()
    st.markdown("---")
    st.caption({"running": "Analysis running…", "success": "Last run: success", "failed": "Last run: FAILED",
                "stalled": "Last run: stalled", "idle": "No run started from the dashboard"}.get(s["state"], ""))

try:
    PAGES[page].render()
except Exception as exc:          # never crash silently: show the error in the page
    st.error(f"This page could not be displayed: {exc}")
    st.exception(exc)

if runner.read_status()["state"] == "running":      # live progress: refresh while the analysis runs
    time.sleep(3)
    st.rerun()
