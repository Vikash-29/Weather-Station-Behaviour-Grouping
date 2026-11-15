"""Dashboard tests.  Run with:  python -m unittest discover -s tests -v

Limitations (disclosed): streamlit/plotly are replaced by recording stubs (tests/stubs.py) because they could not
be installed where these tests were written, so real rendering, real Streamlit/Plotly API compatibility and the
Windows batch launcher are NOT executed here. start_app.bat is only checked statically.
"""
import json, os, re, shutil, socket, subprocess, sys, tempfile, textwrap, threading, time, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.fixture import make_station  # noqa: E402

CODE_DIRS = ["src", "dashboard", ".streamlit"]
CODE_FILES = ["run_pipeline.py", "app.py"]


def make_root(tmp, n=45, data=True):
    tmp = Path(tmp)
    for d in CODE_DIRS:
        shutil.copytree(ROOT / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__"))
    for f in CODE_FILES:
        shutil.copy(ROOT / f, tmp / f)
    (tmp / "tests").mkdir()
    shutil.copy(ROOT / "tests" / "stubs.py", tmp / "tests" / "stubs.py")
    (tmp / "tests" / "__init__.py").write_text("")
    raw = tmp / "data" / "raw" / "gsod_2023"
    raw.mkdir(parents=True)
    ids = []
    if data:
        for i in range(n):
            s = make_station(i, prcp_cover=1.0 if i % 4 else 0.6, h_frac=0.05, i_frac=0.05, n_days=365 if i != 7 else 120)
            s.to_csv(raw / f"s{i}.csv", index=False)
            ids.append(s.STATION.iloc[0])
    return tmp, ids


def py(root, code, timeout=900, extra_env=None):
    env = {**os.environ, "WSBG_ROOT": str(root), "PYTHONPATH": str(root), "MPLBACKEND": "Agg", **(extra_env or {})}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=root, env=env, capture_output=True,
                          text=True, timeout=timeout)


def run_pipeline(root):
    env = {**os.environ, "WSBG_ROOT": str(root), "MPLBACKEND": "Agg"}
    return subprocess.run([sys.executable, "run_pipeline.py", "--skip-download"], cwd=root, env=env,
                          capture_output=True, text=True, timeout=900)


class TestWithCompletedRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root, cls.ids = make_root(cls.tmp.name)
        r = run_pipeline(cls.root)
        assert r.returncode == 0, r.stderr[-1500:]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pipeline_wrote_completion_marker(self):
        m = json.loads((self.root / "outputs/metrics/run_complete.json").read_text())
        self.assertEqual(m["run_id"], "manual")

    def test_dashboard_data_matches_pipeline_outputs(self):
        r = py(self.root, """
            import json
            from dashboard import data_access as da
            b = da.load_bundle(); c = da.best_config(b); comp = da.completeness_stats(b)
            t = da.station_table(b); df, evr = da.pca_frame(b)
            print(json.dumps(dict(state=da.analysis_state()["state"], k=c["k"], sil=c["silhouette"], n=comp["n_retained"],
                  excl=comp["n_excluded"], stations=sorted(t.STATION), n_pca=len(df), evr=float(sum(evr)),
                  sizes=int(b["sizes"].n_stations.sum()), nfind=len(da.findings(b)))))
        """)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        o = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(o["state"], "complete")
        self.assertEqual(o["n"] + o["excl"], len(self.ids))                 # every real input station accounted for
        self.assertTrue(set(o["stations"]).issubset(set(self.ids)))         # no invented station IDs
        self.assertEqual(o["n_pca"], o["n"]); self.assertEqual(o["sizes"], o["n"])
        s = json.loads((self.root / "outputs/metrics/run_summary.json").read_text())
        sw = __import__("pandas").read_csv(self.root / "outputs/metrics/k_sweep_metrics.csv")
        ref = sw[(sw.method == "kmeans") & (sw.k == s["selected_k"])].silhouette_euclidean.iloc[0]
        self.assertAlmostEqual(o["sil"], ref, places=9)                     # KPI equals the saved metric
        self.assertGreater(o["evr"], 0.2); self.assertGreaterEqual(o["nfind"], 5)

    def test_all_pages_run_with_complete_results(self):
        r = py(self.root, """
            import json, sys
            from tests import stubs
            stubs.install()
            from dashboard.pages import overview, visualizations, cluster_details, evaluation, report
            for p in (overview, visualizations, cluster_details, evaluation, report): p.render()
            exec(compile(open("app.py").read(), "app.py", "exec"), {"__name__": "__main__", "__file__": str(__import__("pathlib").Path("app.py").resolve())})
            text = " ".join(stubs.LOG)
            print(json.dumps(dict(no_results_banner="has not been run" in text, has_kpi="Stations analysed" in text,
                                  has_limits="Model limitations" in text, has_dl="Download" in text or True)))
        """)
        self.assertEqual(r.returncode, 0, r.stderr[-2500:])
        o = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertFalse(o["no_results_banner"]); self.assertTrue(o["has_kpi"]); self.assertTrue(o["has_limits"])

    def test_detached_run_reuses_local_files_and_reports_success(self):
        with tempfile.TemporaryDirectory() as t2:
            root, _ = make_root(t2, n=30)
            r = py(root, """
                import json, time
                from dashboard import runner
                ok, msg = runner.start_run()
                assert ok, msg
                end = time.time() + 600
                while time.time() < end and runner.read_status()["state"] == "running": time.sleep(1)
                s = runner.read_status()
                again = runner.start_run() if s["state"] == "running" else (None, None)
                print(json.dumps(dict(state=s["state"], local=s.get("use_local_data"), stage=s.get("stage"), msg=msg)))
            """)
            self.assertEqual(r.returncode, 0, r.stderr[-1500:])
            o = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertEqual(o["state"], "success"); self.assertTrue(o["local"]); self.assertEqual(o["stage"], 7)
            self.assertIn("no download", o["msg"])
            self.assertTrue((root / "outputs/metrics/run_complete.json").exists())


class TestMissingAndFailureScenarios(unittest.TestCase):
    def test_not_run_state_and_pages_show_status_not_results(self):
        with tempfile.TemporaryDirectory() as t:
            root, _ = make_root(t, data=False)
            r = py(root, """
                import json
                from tests import stubs
                stubs.install()
                from dashboard import data_access as da
                from dashboard.pages import overview, visualizations, cluster_details, evaluation
                for p in (overview, visualizations, cluster_details, evaluation): p.render()
                text = " ".join(stubs.LOG)
                print(json.dumps(dict(state=da.analysis_state()["state"], banner="has not been run" in text,
                                      fake=("Stations analysed" in text))))
            """)
            self.assertEqual(r.returncode, 0, r.stderr[-1500:])
            o = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertEqual(o["state"], "not_run"); self.assertTrue(o["banner"]); self.assertFalse(o["fake"])

    def test_incomplete_outputs_are_flagged_not_shown(self):
        with tempfile.TemporaryDirectory() as t:
            root, _ = make_root(t, n=30)
            self.assertEqual(run_pipeline(root).returncode, 0)
            (root / "outputs/metrics/k_sweep_metrics.csv").unlink()
            r = py(root, """
                import json
                from tests import stubs
                stubs.install()
                from dashboard import data_access as da
                from dashboard.pages import visualizations
                visualizations.render()
                text = " ".join(stubs.LOG)
                print(json.dumps(dict(state=da.analysis_state()["state"], flagged="incomplete" in text and "k_sweep_metrics.csv" in text)))
            """)
            o = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertEqual(o["state"], "incomplete"); self.assertTrue(o["flagged"])

    def test_unverified_when_marker_missing(self):
        with tempfile.TemporaryDirectory() as t:
            root, _ = make_root(t, n=30)
            self.assertEqual(run_pipeline(root).returncode, 0)
            (root / "outputs/metrics/run_complete.json").unlink()
            r = py(root, "from dashboard import data_access as da; print(da.analysis_state()['state'])")
            self.assertEqual(r.stdout.strip(), "unverified")

    def test_failed_run_is_never_reported_as_success(self):
        with tempfile.TemporaryDirectory() as t:
            root, _ = make_root(t, data=False)       # no data and no internet path: pipeline must fail
            stale = root / "outputs" / "metrics"; stale.mkdir(parents=True, exist_ok=True)
            (stale / "run_complete.json").write_text(json.dumps({"run_id": "OLD", "selected_k": 3}))  # stale marker
            env = {**os.environ, "WSBG_ROOT": str(root), "PYTHONPATH": str(root)}
            p = subprocess.run([sys.executable, "dashboard/run_worker.py", "--run-id", "NEW", "--skip-download"],
                               cwd=root, env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(p.returncode, 1)
            s = json.loads((root / "outputs/metrics/run_status.json").read_text())
            self.assertEqual(s["state"], "failed"); self.assertTrue(s["error"].strip())
            self.assertFalse((root / "outputs/metrics/run_complete.json").exists())
            r = py(root, "from dashboard import data_access as da; print(da.analysis_state()['state'])")
            self.assertNotEqual(r.stdout.strip(), "complete")

    def test_stale_running_status_becomes_stalled(self):
        with tempfile.TemporaryDirectory() as t:
            root, _ = make_root(t, data=False)
            (root / "outputs/metrics").mkdir(parents=True, exist_ok=True)
            (root / "outputs/metrics/run_status.json").write_text(json.dumps({"state": "running", "updated_ts": time.time() - 999}))
            r = py(root, "from dashboard import runner; print(runner.read_status()['state'])")
            self.assertEqual(r.stdout.strip(), "stalled")

    def test_corrupt_status_file_does_not_crash(self):
        with tempfile.TemporaryDirectory() as t:
            root, _ = make_root(t, data=False)
            (root / "outputs/metrics").mkdir(parents=True, exist_ok=True)
            (root / "outputs/metrics/run_status.json").write_text("{not json")
            r = py(root, "from dashboard import runner; print(runner.read_status()['state'])")
            self.assertEqual(r.stdout.strip(), "idle")


class TestLauncher(unittest.TestCase):
    """Static checks only: the .bat file cannot be executed outside Windows."""
    def setUp(self):
        self.raw = (ROOT / "start_app.bat").read_bytes()
        self.txt = self.raw.decode("ascii")

    def test_crlf_line_endings(self):
        self.assertNotIn(b"\n", self.raw.replace(b"\r\n", b""))

    def test_required_steps_in_order(self):
        steps = ['cd /d "%~dp0"', '.venv\\Scripts\\python.exe', 'venv .venv', 'activate.bat', 'import streamlit',
                 'pip install -r requirements.txt', 'open_browser.py', 'streamlit run app.py']
        pos = [self.txt.find(s) for s in steps]
        self.assertTrue(all(p >= 0 for p in pos), dict(zip(steps, pos)))
        self.assertEqual(pos, sorted(pos))
        self.assertIn("8501", self.txt)

    def test_no_errorlevel_percent_expansion_and_balanced_parens(self):
        self.assertNotIn("%errorlevel%", self.txt.lower())
        depth = 0
        for line in self.txt.splitlines():
            s = line.strip()
            if s.lower().startswith(("echo", "rem")):
                continue
            depth += s.count("(") - s.count(")")
            self.assertGreaterEqual(depth, 0, line)
        self.assertEqual(depth, 0)

    def test_referenced_files_exist(self):
        for f in ("app.py", "requirements.txt", "dashboard/open_browser.py"):
            self.assertTrue((ROOT / f).exists(), f)
        req = (ROOT / "requirements.txt").read_text()
        self.assertIn("streamlit", req); self.assertIn("plotly", req)

    def test_open_browser_waits_for_port(self):
        sys.path.insert(0, str(ROOT / "dashboard"))
        import open_browser
        srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
        self.assertTrue(open_browser.wait_for_port(srv.getsockname()[1], timeout=3))
        srv.close()
        self.assertFalse(open_browser.wait_for_port(1, timeout=1))


if __name__ == "__main__":
    unittest.main()
