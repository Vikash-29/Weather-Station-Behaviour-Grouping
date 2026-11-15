"""Background worker: runs run_pipeline.py, streams its log and writes run_status.json.

Usage: python dashboard/run_worker.py --year 2023 --run-id abc [--skip-download]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from src import config  # noqa: E402

STATUS = config.METRIC_DIR / "run_status.json"
LOG = config.ROOT / "outputs" / "run_log.txt"
STAGE_RE = re.compile(r"\[STAGE (\d+)/(\d+)\] (.*)")


class Status:
    def __init__(self, run_id, use_local_data=False):
        self.lock = threading.Lock()
        self.d = dict(state="running", run_id=run_id, stage=0, n_stages=7, message="Starting...",
                      started_ts=time.time(), updated_ts=time.time(), use_local_data=use_local_data)

    def update(self, **kw):
        with self.lock:
            self.d.update(kw, updated_ts=time.time())
            tmp = STATUS.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.d))
            os.replace(tmp, STATUS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=config.YEAR)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    config.ensure_dirs()
    st = Status(a.run_id, a.skip_download)
    st.update()
    stop = threading.Event()
    threading.Thread(target=lambda: [st.update() for _ in iter(lambda: stop.wait(10), True)], daemon=True).start()
    cmd = [sys.executable, str(config.ROOT / "run_pipeline.py"), "--year", str(a.year)]
    if a.skip_download:
        cmd.append("--skip-download")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg", "WSBG_RUN_ID": a.run_id,
           "WSBG_ROOT": str(config.ROOT)}
    tail, rc = [], -1
    try:
        with open(LOG, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, cwd=str(config.ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=env, errors="replace")
            for line in proc.stdout:
                log.write(line); log.flush()
                tail = (tail + [line.rstrip()])[-40:]
                m = STAGE_RE.search(line)
                if m:
                    st.update(stage=int(m.group(1)), n_stages=int(m.group(2)), message=m.group(3))
            rc = proc.wait()
    except Exception as exc:
        tail.append(f"Worker error: {exc}")
    stop.set()
    marker = config.METRIC_DIR / "run_complete.json"
    ok = False
    if rc == 0 and marker.exists():
        try:
            ok = json.loads(marker.read_text()).get("run_id") == a.run_id
        except Exception:
            ok = False
    if ok:
        st.update(state="success", stage=st.d["n_stages"], message="Analysis completed", finished_ts=time.time())
    else:
        st.update(state="failed", message="Analysis FAILED", finished_ts=time.time(), return_code=rc,
                  error="\n".join(tail[-25:]) or "No output captured.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
