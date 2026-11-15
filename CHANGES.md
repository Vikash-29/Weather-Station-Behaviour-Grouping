# Changes made for the dashboard upgrade

## Modified files
- `src/pipeline.py` — added `[STAGE i/7]` progress prints; `run_all` deletes `run_complete.json` at start and writes it (with run id, K, station count, time) only after everything finished. No algorithm, preprocessing, metric or report logic was changed.
- `requirements.txt` — added `streamlit>=1.37`, `plotly>=5.18`.
- `README.md` — dashboard installation, launch, run, troubleshooting.

## New files
- `app.py` — Streamlit entry point (sidebar navigation, auto-refresh while a run is in progress).
- `start_app.bat` — one-click Windows launcher (CRLF).
- `.streamlit/config.toml` — theme; disables usage-statistics prompt.
- `dashboard/style.css` — custom CSS.
- `dashboard/data_access.py` — reads real outputs, state detection (not_run / incomplete / unverified / complete), KPIs, findings, PCA frame (no Streamlit dependency).
- `dashboard/runner.py`, `dashboard/run_worker.py` — safe background execution of `run_pipeline.py` with status, heartbeat and log; reuses local NOAA files.
- `dashboard/open_browser.py` — opens the browser when the server is ready.
- `dashboard/charts.py` (Plotly), `dashboard/ui.py` (helpers), `dashboard/pages/{overview,visualizations,cluster_details,evaluation,report}.py`.
- `tests/test_dashboard.py`, `tests/stubs.py`.

Nothing was deleted: raw data, existing outputs, tests, notebook and research code are kept.

## Test results (what was and was not executed)
Executed and passing (22/22): all earlier methodology tests; completion-marker behaviour; dashboard KPIs/tables equal the saved pipeline outputs and contain only real input station IDs; a real detached background run (local files reused, status reaches success); not-run, incomplete, unverified, stale/corrupt status scenarios; failed run (with a stale marker present) is never reported as success; launcher static checks; browser-opener port wait.

NOT executed (be aware):
- **Streamlit and Plotly were not installed** in the build environment (no internet), so the dashboard was never started and charts were never rendered. Page code was run against recording stubs, which catches logic and data-connection errors but not real Streamlit/Plotly API or layout problems. Expect possible small visual/API fixes on first launch.
- **`start_app.bat` was not run** (no Windows); only static checks were done.
- **No real NOAA data** could be downloaded, so no genuine analysis results exist yet; all test runs used test-only synthetic files in temporary folders.
- The reference image mentioned in the request was not received (only text), so the visual design is my own clean dashboard style, not a copy of that image.
