"""Dataset download, loading, cleaning and matrix preparation (NOAA GSOD, India)."""
import time
import numpy as np
import pandas as pd
import requests
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

from src import config

MANUAL_HELP = f"""
Automatic download failed or no raw files were found.
Manual download instructions:
 1. Open  {config.ISD_HISTORY_URL}  and save it as  data/raw/isd-history.csv
 2. Station files are at {config.GSOD_BASE}/<YEAR>/<USAF><WBAN>.csv
    (e.g. .../{config.YEAR}/42182099999.csv). Pick stations whose CTRY column is 'IN'.
 3. Save each CSV into  data/raw/gsod_<YEAR>/  (keep the file name) and re-run with skip_download=True.
Dataset page: https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00516
"""

# GSOD missing-value sentinels (documented in the GSOD readme)
SENTINELS = {"TEMP": 9999.9, "DEWP": 9999.9, "SLP": 9999.9, "STP": 9999.9, "VISIB": 999.9,
             "WDSP": 999.9, "MXSPD": 999.9, "GUST": 999.9, "MAX": 9999.9, "MIN": 9999.9,
             "PRCP": 99.99, "SNDP": 999.9}

# Physically plausible ranges (SI-converted columns)
RANGES = {"temp_c": (-60, 60), "dewp_c": (-80, 40), "tmax_c": (-60, 65), "tmin_c": (-70, 60),
          "slp_hpa": (870, 1090), "stp_hpa": (300, 1100), "wind_ms": (0, 75), "mxspd_ms": (0, 100),
          "gust_ms": (0, 120), "visib_km": (0, 200), "prcp_mm": (0, 1500)}


# ----------------------------------------------------------------------------- download
def _get(url, retries=3, timeout=60):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            last = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(2 * (i + 1))
    raise ConnectionError(f"Could not fetch {url}: {last}")


def download_station_list():
    """Download NOAA's ISD station history (metadata for every station)."""
    config.ensure_dirs()
    path = config.RAW_DIR / "isd-history.csv"
    if not path.exists():
        r = _get(config.ISD_HISTORY_URL)
        if r is None:
            raise ConnectionError("isd-history.csv not found. " + MANUAL_HELP)
        path.write_bytes(r.content)
    return path


def select_candidate_stations(year=config.YEAR, country=config.COUNTRY_FIPS):
    """Stations of the country whose ISD record spans the whole analysis year."""
    hist = pd.read_csv(download_station_list(), dtype=str)
    hist = hist[(hist["CTRY"] == country) & (hist["USAF"] != "999999")]
    hist = hist[(hist["BEGIN"] <= f"{year}0101") & (hist["END"] >= f"{year}1231")].copy()
    hist["station_id"] = hist["USAF"] + hist["WBAN"]
    return hist.drop_duplicates("station_id").reset_index(drop=True)


def download_gsod(year, station_ids, pause=0.2):
    """Download one GSOD CSV per station. Existing files are skipped (resumable)."""
    out = config.RAW_DIR / f"gsod_{year}"
    out.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "cached": 0, "not_available": 0, "failed": 0}
    for sid in station_ids:
        f = out / f"{sid}.csv"
        if f.exists():
            counts["cached"] += 1
            continue
        try:
            r = _get(f"{config.GSOD_BASE}/{year}/{sid}.csv")
        except ConnectionError:
            counts["failed"] += 1
            continue
        if r is None:
            counts["not_available"] += 1
            continue
        f.write_bytes(r.content)
        counts["downloaded"] += 1
        time.sleep(pause)
    return counts


def load_raw_observations(year=config.YEAR):
    """Read all station CSVs for one year into one daily observation table."""
    files = sorted((config.RAW_DIR / f"gsod_{year}").glob("*.csv"))
    if not files:
        raise FileNotFoundError(MANUAL_HELP)
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f, dtype={"STATION": str, "FRSHTT": str}, low_memory=False)
            if len(d):
                frames.append(d)
        except Exception as exc:  # unreadable file: report and continue
            print(f"WARNING: skipped {f.name}: {exc}")
    if not frames:
        raise FileNotFoundError(MANUAL_HELP)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- cleaning
def clean_observations(raw):
    """Clean daily records. Returns (clean_df, log_df); every step is logged."""
    log = []
    df = raw.copy()
    log.append(("raw daily records", len(df), "rows read from station files"))

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    n = int(df["DATE"].isna().sum())
    df = df.dropna(subset=["DATE"])
    log.append(("unparseable dates removed", n, "rows"))

    n = int(df.duplicated(["STATION", "DATE"]).sum())
    df = df.drop_duplicates(["STATION", "DATE"], keep="first")
    log.append(("duplicate station-days removed", n, "rows (same STATION and DATE)"))

    for col in list(SENTINELS) + ["LATITUDE", "LONGITUDE", "ELEVATION"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "NAME" not in df.columns:
        df["NAME"] = ""
    if "FRSHTT" not in df.columns:
        df["FRSHTT"] = "000000"

    for col, s in SENTINELS.items():
        m = df[col] >= s
        log.append((f"{col}: missing-value sentinel -> NaN", int(m.sum()), f"values >= {s}"))
        df.loc[m, col] = np.nan

    # PRCP attribute flags (GSOD readme):
    #   I = no precipitation data and no hourly precipitation occurrence reported -> missing
    #   H = station reported 0 for the day BUT hourly observations report precipitation occurred;
    #       NOAA: "should be considered as incomplete data for the day" -> ambiguous, NOT a confirmed dry day
    # Both are set to NaN for precipitation calculations but are counted separately.
    df["prcp_flag_h"] = 0.0
    if "PRCP_ATTRIBUTES" in df.columns:
        attr = df["PRCP_ATTRIBUTES"].astype(str).str.strip()
        mi = attr.eq("I") & df["PRCP"].notna()
        mh = attr.eq("H") & df["PRCP"].notna()
        log.append(("PRCP flag 'I' (no precipitation report) -> NaN", int(mi.sum()), "treated as missing"))
        log.append(("PRCP flag 'H' (0 reported, but precipitation seen in hourly obs) -> NaN", int(mh.sum()),
                    "ambiguous/incomplete day: not a confirmed dry day; counted separately as dq_prcp_flag_h_frac"))
        df.loc[mh, "prcp_flag_h"] = 1.0
        df.loc[mi | mh, "PRCP"] = np.nan
    else:
        log.append(("PRCP_ATTRIBUTES column absent: flags H/I not handled", 0, "check raw files"))

    # Unit conversion to SI
    df["temp_c"] = (df["TEMP"] - 32) * 5 / 9
    df["dewp_c"] = (df["DEWP"] - 32) * 5 / 9
    df["tmax_c"] = (df["MAX"] - 32) * 5 / 9
    df["tmin_c"] = (df["MIN"] - 32) * 5 / 9
    df["slp_hpa"] = df["SLP"]
    df["stp_hpa"] = df["STP"]
    df["wind_ms"] = df["WDSP"] * 0.514444
    df["mxspd_ms"] = df["MXSPD"] * 0.514444
    df["gust_ms"] = df["GUST"] * 0.514444
    df["visib_km"] = df["VISIB"] * 1.609344
    df["prcp_mm"] = df["PRCP"] * 25.4

    # Plausibility ranges
    for col, (lo, hi) in RANGES.items():
        m = (df[col] < lo) | (df[col] > hi)
        if m.sum():
            log.append((f"{col}: implausible value -> NaN", int(m.sum()), f"outside [{lo}, {hi}]"))
            df.loc[m, col] = np.nan

    # Internal consistency
    m = df["tmax_c"] < df["tmin_c"]
    log.append(("daily max < daily min -> both NaN", int(m.sum()), "inconsistent record"))
    df.loc[m, ["tmax_c", "tmin_c"]] = np.nan
    m = df["dewp_c"] > df["temp_c"] + 0.5
    log.append(("dew point > temperature + 0.5 C -> dew point NaN", int(m.sum()), "physically inconsistent"))
    df.loc[m, "dewp_c"] = np.nan

    log.append(("clean daily records kept", len(df), "rows (outliers are not removed unless implausible)"))
    return df, pd.DataFrame(log, columns=["step", "count", "note"])


# ----------------------------------------------------------------------------- matrix
def build_matrix(df, feature_cols, scaler="standard", imputer="median", log1p_cols=()):
    """Impute -> (optional log1p) -> scale. Returns (X_scaled ndarray, X_imputed DataFrame)."""
    X = df[list(feature_cols)].astype(float).copy()
    if imputer == "knn":
        imp = KNNImputer(n_neighbors=5)
    else:
        imp = SimpleImputer(strategy="median")
    arr = imp.fit_transform(X)
    Xi = pd.DataFrame(arr, index=X.index, columns=X.columns)
    for c in log1p_cols:
        if c in Xi.columns:
            Xi[c] = np.log1p(Xi[c].clip(lower=0))
    sc = RobustScaler() if scaler == "robust" else StandardScaler()
    return sc.fit_transform(Xi.values), Xi
