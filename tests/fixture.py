"""TEST-ONLY synthetic GSOD-format files. Used exclusively inside temporary directories by the unit
tests to check code paths. Never used for any reported result."""
import numpy as np
import pandas as pd


def make_station(i, n_days=365, prcp_cover=1.0, h_frac=0.0, i_frac=0.0, seed=0):
    rng = np.random.default_rng(seed + i)
    days = pd.date_range("2023-01-01", "2023-12-31")[:n_days]
    reg = i % 3
    t = (24 + 5 * reg) + 8 * np.sin(np.arange(n_days) / 58) + rng.normal(0, 2, n_days)
    tf = t * 9 / 5 + 32
    prcp = np.where(rng.random(n_days) < 0.25, rng.uniform(0.01, 1.0, n_days), 0.0)
    attr = np.array(["G"] * n_days, dtype=object)
    miss = rng.random(n_days) > prcp_cover
    prcp = np.where(miss, 99.99, prcp)
    attr[miss] = ""
    attr[(rng.random(n_days) < h_frac) & ~miss] = "H"
    attr[(rng.random(n_days) < i_frac) & ~miss & (attr != "H")] = "I"
    fr = np.where(rng.random(n_days) < 0.1, "010000", "000000")
    return pd.DataFrame({
        "STATION": f"4200{i:05d}99999", "DATE": days.strftime("%Y-%m-%d"), "LATITUDE": 10 + i * 0.4,
        "LONGITUDE": 70 + i * 0.3, "ELEVATION": 50 + reg * 400.0, "NAME": f"STN {i}, IN", "TEMP": tf,
        "DEWP": tf - rng.uniform(1, 20, n_days), "SLP": 1010 + rng.normal(0, 3, n_days),
        "STP": 950 - reg * 40 + rng.normal(0, 3, n_days), "VISIB": rng.uniform(2, 6, n_days),
        "WDSP": rng.uniform(1, 8 + 4 * reg, n_days), "MXSPD": rng.uniform(5, 20, n_days), "GUST": 999.9,
        "MAX": tf + 8, "MIN": tf - 8, "PRCP": prcp, "PRCP_ATTRIBUTES": attr, "SNDP": 999.9, "FRSHTT": fr})
