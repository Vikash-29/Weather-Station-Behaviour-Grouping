# Weather-Station Behaviour Grouping for Automated Data Quality Assessment

K-Means clustering of Indian weather stations using real NOAA **Global Surface Summary of the Day (GSOD)** data.
The unit of clustering is the **station** (daily records are aggregated to station-level statistics).

## IMPORTANT: status of this package
- **No real-data results have been generated yet.** The environment used to build and revise this project has no internet access (NOAA's host could not be resolved; the real-data run was attempted and failed with a connection error). `outputs/`, `data/processed/` and `final_report.md` are therefore empty/placeholder until you run the pipeline once with internet. Every number, chart and report sentence is then produced from the real data.
- What *was* verified: 7 automated tests (`python -m unittest discover -s tests -v`) pass, and the full pipeline and notebook code cells run without errors on small **test-only synthetic files in temporary folders**. Those files are never part of the deliverable and never used for results.
- The NOAA download code and the GSOD column handling have not been exercised against the live files; if the first real run fails, it will most likely be there.

## Dashboard (Streamlit) — quick start on Windows + VS Code
1. Install Python 3.10+ from https://www.python.org/downloads/ (tick **"Add python.exe to PATH"**).
2. In VS Code: *File → Open Folder…* and choose this project folder.
3. **Double-click `start_app.bat`** (in VS Code's Explorer: right-click → *Reveal in File Explorer*, then double-click).
   First run: it creates `.venv`, installs the packages (a few minutes), starts the dashboard and opens your browser at **http://localhost:8501**. Later runs start in seconds. Keep the black window open while you use the dashboard; close it to stop.
4. On the **Overview** page click **Run Analysis**. Progress and errors are shown live. Reuse: if NOAA files already exist in `data/raw/gsod_<year>/` they are reused and **nothing is downloaded again** (tick the optional box to check NOAA for missing stations).
5. Browse **Overview, Visualizations, Cluster Details, Evaluation, Report**. The Report page has CSV (ZIP) and `final_report.md` downloads.

Manual alternative (inside the activated venv): `streamlit run app.py`.

### How the dashboard stays honest
- Every number, table and chart is read from files the pipeline wrote (`outputs/`, `data/processed/`). Nothing is generated for display.
- Before any analysis exists, pages say so instead of showing results.
- A run counts as **successful only if** the pipeline exits with code 0 **and** writes `outputs/metrics/run_complete.json` carrying that run's id. Failed or interrupted runs show the error (full log: `outputs/run_log.txt`); incomplete outputs are refused; outputs without the marker are shown with an "unverified" warning.
- The analysis runs in a separate background process, so closing a browser tab does not kill it; a run with no heartbeat for 3 minutes is shown as "stalled".

### Troubleshooting
| Problem | What to do |
|---|---|
| "Python 3 was not found" | Install Python (tick *Add to PATH*), double-click `start_app.bat` again. |
| Package installation fails | Check internet; run `start_app.bat` again. Or in a terminal: `.venv\Scripts\activate` then `pip install -r requirements.txt`. |
| Broken `.venv` | Delete the `.venv` folder and double-click `start_app.bat` again. |
| Browser does not open | Open http://localhost:8501 manually. |
| Port 8501 busy / "address already in use" | Close other dashboard windows (or change `PORT` at the top of `start_app.bat`). |
| Run Analysis fails with a connection error | NOAA is unreachable. Use the manual download steps below, then click Run Analysis again. |
| Stage 6 seems slow | Stability analysis can take several minutes with many stations; the progress bar keeps updating. |
| Pages say results are incomplete | Click Run Analysis again; the missing files are listed on screen. |
| Windows SmartScreen warning on the `.bat` | Choose *More info → Run anyway* (the file is plain text; open it to inspect). |

## Data source (genuine, public)
- Dataset: NOAA NCEI Global Surface Summary of the Day — https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00516
- Files: `https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/<YEAR>/<USAF><WBAN>.csv`
- Station list: https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv (stations with country code `IN`)
- Documentation: https://www.ncei.noaa.gov/data/global-summary-of-the-day/doc/readme.txt
- Default year: 2023 (`src/config.py`). Nothing is fabricated; unavailable stations are simply skipped.

## Quick start (Windows, PowerShell or Command Prompt)
```
cd Weather_Station_Behaviour_Grouping
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py            # downloads data, runs everything, writes outputs + final_report.md
```
Or open `notebooks/Weather_Station_Clustering.ipynb` (`jupyter notebook`) and use *Restart & Run All*.
The first run downloads roughly one small CSV per Indian station (a few minutes; downloads are resumable and cached in `data/raw/`).

## If automatic download is blocked (manual instructions)
1. Save `isd-history.csv` (link above) into `data/raw/`.
2. In it, find rows with `CTRY` = `IN` whose `BEGIN` <= 20230101 and `END` >= 20231231; station id = `USAF` + `WBAN`.
3. Download `.../access/2023/<station id>.csv` for each and save into `data/raw/gsod_2023/`.
4. Run `python run_pipeline.py --skip-download` (or set `SKIP_DOWNLOAD = True` in the notebook).

## Project structure
```
data/raw/            downloaded NOAA files (not redistributed here)
data/processed/      station_features_with_clusters.csv, station_features_scaled.csv  (created by the run)
notebooks/           Weather_Station_Clustering.ipynb   (full narrated workflow)
src/                 config, data_preprocessing, feature_engineering, clustering, evaluation,
                     visualization, pipeline, report_generator
outputs/figures/     13 PNG figures      outputs/metrics/   evaluation CSVs + run_summary.json
outputs/cluster_profiles/  profile CSVs  final_report.md    generated from the results
tests/               unit + integration tests (test-only synthetic fixtures in temp dirs)
app.py, dashboard/   Streamlit dashboard (pages, charts, runner, style.css)
start_app.bat        one-click Windows launcher
run_pipeline.py      one-command end-to-end run
```

## Methodological notes
- **Preprocessing:** GSOD sentinels -> NaN; unit conversion to SI; implausible/inconsistent records -> NaN (plausible extremes are kept); duplicates removed; every step counted in `outputs/metrics/cleaning_log.csv`.
- **Precipitation flags (corrected):** attribute **I** (no precipitation report) -> missing. Attribute **H** (station reported 0, but hourly observations reported precipitation; NOAA: "incomplete data for the day") -> handled separately: *not* a confirmed dry day, excluded from precipitation calculations, counted in the log and kept as station feature `dq_prcp_flag_h_frac`.
- **Precipitation features (corrected):** a partial record is **not** multiplied up to 365 days (missing days are not shown to be representative). Per-station coverage = valid PRCP days / calendar days. `precip_observed_total_mm`, `rainy_day_frac` and `precip_intensity_mm` are computed on valid days only and are set to **unavailable (NaN)** below 90 % coverage (`PRCP_MIN_COVERAGE`). Even above it the total is an observed lower bound, not an annual estimate. Unavailable values are median-imputed (features missing for > 40 % of stations are dropped); variants without precipitation features are in the sensitivity analysis.
- **Event indicators (corrected):** in GSOD FRSHTT, 1 = reported, 0 = *no event or not reported*. Features are renamed `fog_reported_frac` / `thunder_reported_frac` and described as lower bounds that also reflect reporting practice; a variant without them is tested.
- **Completeness exclusion (documented):** stations with < 250 valid temperature days are excluded. The number excluded, their list (`excluded_stations.csv`), their latitude/elevation compared with retained stations, and a threshold-sensitivity table (`completeness_threshold_sensitivity.csv`) are generated and reported. **Selection bias:** clusters describe well-reporting stations only; the poorest-reporting stations, which a data-quality screen most needs to flag, are removed before clustering, so results must not be generalised to the whole network.
- **Other features:** station mean/std/extremes of temperature, derived relative humidity, pressure, wind, visibility. Latitude/longitude/elevation/name are *not* clustering features. Features >40 % missing or |r| > 0.97 with an earlier feature are dropped; median imputation; z-score standardisation.
- **Distance metrics:** scikit-learn K-Means is Euclidean only. Manhattan is used (a) *for evaluation only* (silhouette of the same K-Means labels) and (b) via **K-Medians** (own implementation, `src/clustering.py`), a different algorithm with an L1 objective. Davies-Bouldin uses scikit-learn's Euclidean definition for both.
- **Selection of K** (K = 2–10): composite of Euclidean silhouette, Davies-Bouldin, seed stability and subsample stability among K >= 3 with a minimum cluster size; the elbow K is only a tie-breaker. Disagreement between metrics is reported, not hidden.
- **Stability:** random initialisations, 80 % subsample refits, consecutive-K agreement, six preprocessing/feature-set variants (ARI).
- **Reproducibility:** fixed seed (42) everywhere, relative paths, pinned minimum versions.
- **Interpretation:** the report's cluster descriptions are auto-generated from computed profiles. They are exploratory groupings of one year of observations, not permanent station categories and not statements about station reliability or weather quality. Please read and refine them yourself before submission.

## Tests
`python -m unittest discover -s tests -v` (22 tests: methodology, pipeline, dashboard data connection, failure scenarios, launcher static checks).
See `CHANGES.md` for the list of changed files and the test limitations.

## Dependencies
`pip install -r requirements.txt` (numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, requests, jupyter, nbformat, ipykernel).
