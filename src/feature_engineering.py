"""Daily derived variables, station-level aggregation and feature selection."""
import numpy as np
import pandas as pd

from src import config

FEATURE_LABELS = {
    "temp_mean": "Mean temperature (°C)", "temp_std": "Std. of daily mean temperature (°C)",
    "temp_min": "Lowest daily minimum temperature (°C)", "temp_max": "Highest daily maximum temperature (°C)",
    "dtr_mean": "Mean diurnal temperature range (°C)", "rh_mean": "Mean relative humidity (%)",
    "rh_std": "Std. of relative humidity (%)", "dewp_dep_mean": "Mean dew-point depression (°C)",
    "wind_mean": "Mean wind speed (m/s)", "wind_std": "Std. of wind speed (m/s)",
    "wind_peak": "Peak maximum sustained wind (m/s)", "slp_mean": "Mean sea-level pressure (hPa)",
    "slp_std": "Std. of sea-level pressure (hPa)", "stp_mean": "Mean station pressure (hPa)",
    "stp_std": "Std. of station pressure (hPa)", "visib_mean": "Mean visibility (km)",
    "precip_observed_total_mm": "Observed precipitation total on reported days (mm; ≥90% coverage only)",
    "rainy_day_frac": "Fraction of reported days with ≥0.1 mm (≥90% coverage only)",
    "precip_intensity_mm": "Mean precipitation on rainy days (mm; ≥90% coverage only)",
    "fog_reported_frac": "Fraction of days with fog reported (lower bound)",
    "thunder_reported_frac": "Fraction of days with thunder reported (lower bound)",
}
WEATHER_FEATURES = list(FEATURE_LABELS)
DQ_FEATURES = ["dq_missing_temp", "dq_missing_rh", "dq_missing_slp", "dq_missing_stp",
               "dq_missing_wind", "dq_missing_prcp", "dq_missing_visib", "dq_prcp_flag_h_frac"]


def add_daily_derived(df):
    """Relative humidity (Magnus formula), diurnal range, dew-point depression, event flags."""
    df = df.copy()
    es = lambda t: 6.112 * np.exp(17.62 * t / (243.12 + t))
    df["rh"] = (100 * es(df["dewp_c"]) / es(df["temp_c"])).clip(0, 100)
    df["dtr_c"] = df["tmax_c"] - df["tmin_c"]
    df["dewp_dep_c"] = df["temp_c"] - df["dewp_c"]
    fr = df["FRSHTT"].fillna("000000").astype(str).str.zfill(6)
    # GSOD: 1 = event reported, 0 = no event OR not reported. Zero is NOT confirmed absence,
    # so these flags only give a lower bound on event occurrence.
    df["fog_reported"] = fr.str[0].eq("1").astype(float)
    df["thunder_reported"] = fr.str[4].eq("1").astype(float)
    df["rain_day"] = np.where(df["prcp_mm"].isna(), np.nan, (df["prcp_mm"] >= 0.1).astype(float))
    return df


def daily_missing_pct(df):
    cols = ["temp_c", "dewp_c", "rh", "tmax_c", "tmin_c", "slp_hpa", "stp_hpa", "wind_ms",
            "mxspd_ms", "gust_ms", "visib_km", "prcp_mm"]
    return (df[cols].isna().mean() * 100).sort_values(ascending=False)


def aggregate_to_stations(df, year=config.YEAR, min_valid_temp_days=config.MIN_VALID_TEMP_DAYS):
    """One row per station: statistics of the daily series (+ data-completeness features)."""
    g = df.groupby("STATION")
    days_in_year = 366 if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366 else 365
    out = g.agg(
        n_days=("DATE", "size"), n_valid_temp_days=("temp_c", "count"),
        temp_mean=("temp_c", "mean"), temp_std=("temp_c", "std"),
        temp_min=("tmin_c", "min"), temp_max=("tmax_c", "max"),
        dtr_mean=("dtr_c", "mean"), rh_mean=("rh", "mean"), rh_std=("rh", "std"),
        dewp_dep_mean=("dewp_dep_c", "mean"), wind_mean=("wind_ms", "mean"),
        wind_std=("wind_ms", "std"), wind_peak=("mxspd_ms", "max"),
        slp_mean=("slp_hpa", "mean"), slp_std=("slp_hpa", "std"),
        stp_mean=("stp_hpa", "mean"), stp_std=("stp_hpa", "std"),
        visib_mean=("visib_km", "mean"),
        rainy_day_frac=("rain_day", "mean"), fog_reported_frac=("fog_reported", "mean"),
        thunder_reported_frac=("thunder_reported", "mean"), prcp_valid_days=("prcp_mm", "count"),
        prcp_total_reported=("prcp_mm", lambda s: s.sum(min_count=1)),
        latitude=("LATITUDE", "first"), longitude=("LONGITUDE", "first"),
        elevation_m=("ELEVATION", "first"), name=("NAME", "first"),
    )
    # Precipitation: NO scaling of a partial record to 365 days (missing days are not shown to be
    # representative). Use the observed total only where PRCP is valid on >= PRCP_MIN_COVERAGE of days;
    # otherwise the precipitation features are set to NaN ("unavailable").
    out["prcp_coverage"] = out["prcp_valid_days"] / days_in_year
    prcp_ok = out["prcp_coverage"] >= config.PRCP_MIN_COVERAGE
    out["precip_observed_total_mm"] = out["prcp_total_reported"].where(prcp_ok)
    out["rainy_day_frac"] = out["rainy_day_frac"].where(prcp_ok)
    rainy = df[df["rain_day"] == 1].groupby("STATION")["prcp_mm"].mean()
    out["precip_intensity_mm"] = rainy.reindex(out.index).where(prcp_ok)
    no_rain = prcp_ok & out["precip_intensity_mm"].isna() & out["rainy_day_frac"].eq(0)
    out.loc[no_rain, "precip_intensity_mm"] = 0.0   # adequate record with no rainy day
    # Data-completeness (quality) features: share of calendar days without a valid value
    for name, col in {"temp": "temp_c", "rh": "rh", "slp": "slp_hpa", "stp": "stp_hpa",
                      "wind": "wind_ms", "prcp": "prcp_mm", "visib": "visib_km"}.items():
        out[f"dq_missing_{name}"] = 1 - g[col].count() / days_in_year
    out["dq_prcp_flag_h_frac"] = g["prcp_flag_h"].sum() / days_in_year
    out = out.drop(columns=["prcp_valid_days", "prcp_total_reported"])
    # Completeness rule: effect of the exclusion is documented, not hidden
    keep = out["n_valid_temp_days"] >= min_valid_temp_days
    excluded = out[~keep].reset_index()[["STATION", "name", "latitude", "longitude", "elevation_m",
                                         "n_days", "n_valid_temp_days"]]
    sens = pd.DataFrame([dict(min_valid_temp_days=t, stations_kept=int((out["n_valid_temp_days"] >= t).sum()),
                              stations_excluded=int((out["n_valid_temp_days"] < t).sum()))
                         for t in sorted(set(config.COMPLETENESS_THRESHOLDS + [min_valid_temp_days]))])
    return out[keep], excluded, sens


def select_features(station_df, candidates=WEATHER_FEATURES,
                    max_missing=config.MAX_FEATURE_MISSING, corr_thr=config.CORR_PRUNE_THRESHOLD):
    """Drop features that are mostly missing, constant, or near-duplicates. Returns (cols, log_df)."""
    log, keep = [], []
    for c in candidates:
        if c not in station_df.columns:
            continue
        miss = station_df[c].isna().mean()
        if miss > max_missing:
            log.append((c, "dropped", f"{miss:.0%} of stations missing (> {max_missing:.0%})"))
        elif station_df[c].nunique(dropna=True) <= 1:
            log.append((c, "dropped", "constant across stations"))
        else:
            keep.append(c)
    corr = station_df[keep].corr().abs()
    final = []
    for c in keep:
        partner = next((k for k in final if corr.loc[c, k] > corr_thr), None)
        if partner:
            log.append((c, "dropped", f"|r| = {corr.loc[c, partner]:.3f} with {partner} (> {corr_thr})"))
        else:
            final.append(c)
            log.append((c, "kept", f"{station_df[c].isna().mean():.0%} missing -> median-imputed"))
    return final, pd.DataFrame(log, columns=["feature", "decision", "reason"])
