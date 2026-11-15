"""Central configuration for the Weather-Station Behaviour Grouping project."""
import os
from pathlib import Path

# Project root (override with the WSBG_ROOT environment variable if needed)
ROOT = Path(os.environ.get("WSBG_ROOT", Path(__file__).resolve().parents[1])).resolve()
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
PROFILE_DIR = ROOT / "outputs" / "cluster_profiles"
METRIC_DIR = ROOT / "outputs" / "metrics"

SEED = 42                      # global random seed
YEAR = 2023                    # GSOD calendar year analysed
COUNTRY_FIPS = "IN"            # FIPS country code used by NOAA ISD (IN = India)
K_RANGE = list(range(2, 11))   # cluster counts evaluated
MIN_VALID_TEMP_DAYS = 250      # station must report mean temperature on >= this many days
MAX_FEATURE_MISSING = 0.40     # drop station-level features missing for > 40 % of stations
CORR_PRUNE_THRESHOLD = 0.97    # drop one of two features with |r| above this value
PRCP_MIN_COVERAGE = 0.90      # precipitation features need valid PRCP on >= 90 % of calendar days
COMPLETENESS_THRESHOLDS = [150, 200, 250, 300, 330]  # sensitivity of station exclusion rule
MIN_CLUSTER_FRAC = 0.05        # smallest acceptable cluster (share of stations) for the final K

GSOD_BASE = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access"
ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"


def ensure_dirs():
    for d in (RAW_DIR, PROC_DIR, FIG_DIR, PROFILE_DIR, METRIC_DIR):
        d.mkdir(parents=True, exist_ok=True)
