"""Run with:  python -m unittest discover -s tests -v"""
import os, subprocess, sys, tempfile, unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import clustering as cl, config, data_preprocessing as dp, evaluation as ev  # noqa: E402
from src import feature_engineering as fe  # noqa: E402
from tests.fixture import make_station  # noqa: E402


def cleaned(frames):
    return dp.clean_observations(pd.concat(frames, ignore_index=True))


class TestPrecipFlags(unittest.TestCase):
    def test_flags_H_and_I_are_missing_and_logged_separately(self):
        s = make_station(1, h_frac=0.2, i_frac=0.2)
        raw_h = int((s.PRCP_ATTRIBUTES == "H").sum()); raw_i = int((s.PRCP_ATTRIBUTES == "I").sum())
        self.assertGreater(raw_h, 0); self.assertGreater(raw_i, 0)
        df, log = cleaned([s])
        self.assertTrue(df.loc[df.PRCP_ATTRIBUTES == "H", "prcp_mm"].isna().all())
        self.assertTrue(df.loc[df.PRCP_ATTRIBUTES == "I", "prcp_mm"].isna().all())
        self.assertEqual(int(df.prcp_flag_h.sum()), raw_h)
        rows = log[log.step.str.contains("flag 'H'")]
        self.assertEqual(int(rows["count"].iloc[0]), raw_h)
        self.assertEqual(int(log[log.step.str.contains("flag 'I'")]["count"].iloc[0]), raw_i)
        self.assertTrue((df.loc[df.PRCP_ATTRIBUTES == "G", "prcp_mm"].notna()).all())


class TestPrecipCoverage(unittest.TestCase):
    def test_low_coverage_unavailable_and_no_scaling(self):
        full, part = make_station(1, prcp_cover=1.0), make_station(2, prcp_cover=0.5)
        df, _ = cleaned([full, part])
        st, excl, _ = fe.aggregate_to_stations(fe.add_daily_derived(df), 2023)
        a, b = st.loc[full.STATION.iloc[0]], st.loc[part.STATION.iloc[0]]
        self.assertAlmostEqual(a.precip_observed_total_mm, df[df.STATION == a.name].prcp_mm.sum(), places=6)
        for c in ("precip_observed_total_mm", "rainy_day_frac", "precip_intensity_mm"):
            self.assertTrue(np.isnan(b[c]), c)
        self.assertLess(b.prcp_coverage, config.PRCP_MIN_COVERAGE)
        self.assertNotIn("precip_annual_mm", st.columns)


class TestEventIndicators(unittest.TestCase):
    def test_renamed_reported_features(self):
        self.assertIn("fog_reported_frac", fe.WEATHER_FEATURES)
        self.assertNotIn("fog_frac", fe.WEATHER_FEATURES)
        self.assertIn("lower bound", fe.FEATURE_LABELS["thunder_reported_frac"])


class TestCompletenessExclusion(unittest.TestCase):
    def test_excluded_table_and_sensitivity(self):
        df, _ = cleaned([make_station(1), make_station(2, n_days=100), make_station(3, n_days=249)])
        st, excl, sens = fe.aggregate_to_stations(fe.add_daily_derived(df), 2023, 250)
        self.assertEqual(len(st), 1); self.assertEqual(len(excl), 2)
        self.assertTrue((excl.n_valid_temp_days < 250).all())
        row = sens[sens.min_valid_temp_days == 250].iloc[0]
        self.assertEqual((row.stations_kept, row.stations_excluded), (1, 2))


class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = np.vstack([rng.normal(0, .3, (30, 3)), rng.normal(5, .3, (30, 3))])

    def test_kmedians_and_kmeans_recover_blobs(self):
        for m in ("kmeans", "kmedians"):
            lab = cl.fit_model(m, self.X, 2, 0).labels_
            self.assertEqual(len(set(lab[:30])), 1); self.assertNotEqual(lab[0], lab[-1])

    def test_evaluation_and_selection(self):
        models = cl.run_models(self.X, [2, 3, 4, 5], 0, n_init=5)
        sw = ev.sweep_table(self.X, models)
        self.assertEqual(len(sw), 8)
        stab = ev.stability_table(self.X, [2, 3, 4, 5], 0, n_seeds=3, n_boot=3)
        k, el, sc = ev.select_final_k(sw, stab, len(self.X))
        self.assertIn(k, (3, 4, 5))
        self.assertAlmostEqual(ev.elbow_k([2, 3, 4], [10, 2, 1.5]), 3)


class TestEndToEnd(unittest.TestCase):
    def test_pipeline_runs_on_temp_synthetic_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for d in ("src", "run_pipeline.py"):
                if d == "src":
                    import shutil; shutil.copytree(ROOT / "src", tmp / "src", ignore=shutil.ignore_patterns("__pycache__"))
                else:
                    (tmp / d).write_text((ROOT / d).read_text())
            raw = tmp / "data" / "raw" / "gsod_2023"; raw.mkdir(parents=True)
            for i in range(45):
                make_station(i, prcp_cover=1.0 if i % 4 else 0.6, h_frac=0.05, i_frac=0.05,
                             n_days=365 if i != 7 else 120).to_csv(raw / f"s{i}.csv", index=False)
            env = {**os.environ, "WSBG_ROOT": str(tmp), "MPLBACKEND": "Agg"}
            r = subprocess.run([sys.executable, "run_pipeline.py", "--skip-download"], cwd=tmp, env=env,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertEqual(len(list((tmp / "outputs/figures").glob("*.png"))), 13)
            for f in ("excluded_stations.csv", "completeness_threshold_sensitivity.csv", "k_sweep_metrics.csv",
                      "stability_by_k.csv", "preprocessing_variants.csv"):
                self.assertTrue((tmp / "outputs/metrics" / f).exists(), f)
            rep = (tmp / "final_report.md").read_text()
            for s in ("7.1 Effect of the completeness exclusion", "flag 'H'", "precip_observed_total_mm", "fog_reported_frac"):
                self.assertIn(s, rep)


if __name__ == "__main__":
    unittest.main()
