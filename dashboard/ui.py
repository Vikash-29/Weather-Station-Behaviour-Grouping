"""Shared Streamlit helpers: CSS, cards, banners, cached loading, run panel."""
import html
import time
from pathlib import Path

import streamlit as st

from dashboard import data_access as da
from dashboard import runner

CSS = Path(__file__).with_name("style.css")


def inject_css():
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def title(text, sub=""):
    st.markdown(f'<div class="page-title">{html.escape(text)}</div><div class="page-sub">{html.escape(sub)}</div>',
                unsafe_allow_html=True)


def banner(text, kind="info"):
    st.markdown(f'<div class="banner {kind}">{html.escape(text)}</div>', unsafe_allow_html=True)


def kpi(col, label, value, sub="", accent=False):
    with col:
        st.markdown(f'<div class="kpi{" accent" if accent else ""}"><div class="lab">{html.escape(label)}</div>'
                    f'<div class="val">{html.escape(str(value))}</div><div class="sub">{html.escape(sub)}</div></div>',
                    unsafe_allow_html=True)


def show_plotly(fig):
    """Works across Streamlit versions (width='stretch' replaced use_container_width)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df, **kw):
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True, **kw)


@st.cache_data(show_spinner=False)
def _cached_bundle(sig):
    return da.load_bundle()


def get_bundle():
    return _cached_bundle(da.signature())


def run_panel(key="run"):
    """Run-analysis controls + live progress. Returns True while a run is in progress."""
    s = runner.read_status()
    n_local = len(runner.local_station_files())
    if s["state"] == "running":
        stage, n = int(s.get("stage", 0)), int(s.get("n_stages", 7))
        st.progress(min(stage / n, 1.0), text=f"Stage {stage}/{n}: {s.get('message', '')}")
        elapsed = time.time() - s.get("started_ts", time.time())
        st.caption(f"Running for {elapsed/60:.1f} min. This page refreshes automatically. "
                   + ("Using local NOAA files." if s.get("use_local_data") else "Downloading missing NOAA files."))
        with st.expander("Live log", expanded=False):
            st.code(runner.log_tail() or "(no output yet)")
        return True
    if s["state"] == "stalled":
        banner("The last analysis stopped reporting progress (the worker may have been closed). You can start a new run.", "warn")
    if s["state"] == "failed":
        banner("The last analysis FAILED. Results shown elsewhere (if any) may be incomplete. Details below.", "bad")
        st.code(s.get("error", "No error output captured."))
        st.caption("Full log: outputs/run_log.txt")
    if s["state"] == "success":
        banner("The last analysis completed successfully.", "ok")
    refresh = False
    if n_local:
        st.caption(f"{n_local} NOAA station files found in data/raw — they will be reused, nothing is downloaded again.")
        refresh = st.checkbox("Also check NOAA for missing stations (needs internet)", value=False, key=f"{key}_refresh")
    else:
        st.caption("No local NOAA files yet: the analysis will download them from NOAA (internet required; may take several minutes).")
    if st.button("Run Analysis", type="primary", key=f"{key}_btn"):
        ok, msg = runner.start_run(refresh_download=refresh)
        (st.success if ok else st.error)(msg)
        if ok:
            time.sleep(1)
            st.rerun()
    return False


def require_results(allow_unverified=True):
    """Return the data bundle, or None after showing a clear status message (never fake results)."""
    s = runner.read_status()
    if s["state"] == "running":
        banner("An analysis is running. Results will appear when it finishes.", "info")
        return None
    a = da.analysis_state()
    if a["state"] == "not_run":
        banner("The analysis has not been run yet. There are no results to display. Use 'Run Analysis' on the Overview page.", "warn")
        return None
    if a["state"] == "incomplete":
        banner("Analysis outputs are incomplete, so nothing is shown. Missing: " + ", ".join(a["missing"][:6])
               + (" ..." if len(a["missing"]) > 6 else "") + ". Re-run the analysis.", "bad")
        return None
    if s["state"] in ("failed", "stalled"):
        banner("The most recent analysis run did not finish successfully; the results below may come from an earlier run.", "warn")
    if a["state"] == "unverified":
        banner("These outputs have no completion marker (produced outside the dashboard or by an older version). "
               "They look complete but are unverified; re-run the analysis to confirm.", "warn")
    try:
        return get_bundle()
    except Exception as exc:
        banner(f"Could not read the analysis outputs: {exc}", "bad")
        return None
