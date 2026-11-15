"""Run the complete analysis:  python run_pipeline.py [--skip-download] [--year 2023]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config, pipeline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true", help="use files already in data/raw/gsod_<year>/")
    ap.add_argument("--year", type=int, default=config.YEAR)
    a = ap.parse_args()
    pipeline.run_all(year=a.year, skip_download=a.skip_download)
