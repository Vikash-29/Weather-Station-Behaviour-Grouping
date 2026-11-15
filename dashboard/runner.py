"""Starts and monitors the analysis in a background process. No Streamlit imports (testable).

Status is kept in outputs/metrics/run_status.json, written by dashboard/run_worker.py.
A run counts as successful ONLY if the pipeline exits with code 0 AND wrote run_complete.json
carrying this run's id.
"""
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from src import config

STALE_SECONDS = 180          # a 'running' status without heartbeat for this long is treated as stalled


def status_path():
    return config.METRIC_DIR / "run_status.json"


def log_path():
    return config.ROOT / "outputs" / "run_log.txt"


def read_status():
    p = status_path()
    if not p.exists():
        return {"state": "idle"}
    try:
        s = json.loads(p.read_text())
    except Exception:
        return {"state": "idle", "note": "run_status.json unreadable"}
    if s.get("state") == "running" and time.time() - s.get("updated_ts", 0) > STALE_SECONDS:
        s = {**s, "state": "stalled"}
    return s


def local_station_files(year=config.YEAR):
    d = config.RAW_DIR / f"gsod_{year}"
    return sorted(d.glob("*.csv")) if d.exists() else []


def log_tail(n=25):
    p = log_path()
    if not p.exists():
        return ""
    return "\n".join(p.read_text(errors="replace").splitlines()[-n:])


def start_run(year=config.YEAR, refresh_download=False):
    """Launch the worker detached from Streamlit. Returns (ok, message)."""
    if read_status().get("state") == "running":
        return False, "An analysis is already running."
    config.ensure_dirs()
    local = local_station_files(year)
    skip = bool(local) and not refresh_download     # reuse valid local NOAA files; never re-download them
    run_id = uuid.uuid4().hex[:12]
    cmd = [sys.executable, str(Path(__file__).with_name("run_worker.py")), "--year", str(year), "--run-id", run_id]
    if skip:
        cmd.append("--skip-download")
    kw = {}
    if sys.platform == "win32":
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    else:
        kw["start_new_session"] = True
    try:
        status_path().write_text(json.dumps(dict(state="running", run_id=run_id, stage=0, n_stages=7,
                                                 message="Starting...", started_ts=time.time(), updated_ts=time.time(),
                                                 use_local_data=skip)))
        subprocess.Popen(cmd, cwd=str(config.ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, env={**os.environ, "WSBG_ROOT": str(config.ROOT)}, **kw)
    except Exception as exc:
        status_path().write_text(json.dumps(dict(state="failed", run_id=run_id, error=f"Could not start worker: {exc}",
                                                 updated_ts=time.time())))
        return False, f"Could not start the analysis: {exc}"
    return True, ("Using the NOAA files already in data/raw (no download)." if skip
                  else "Downloading missing NOAA files, then running the analysis.")
