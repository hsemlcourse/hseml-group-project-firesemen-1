#!/usr/bin/env python
"""Train all models and save the final artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running this script directly with `python scripts/train.py` without
# installing the package first. This is handy for Windows/PowerShell users and
# still keeps the project compatible with `pip install -e .`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eth_price.config import DEFAULT_MAX_ROWS, DEFAULT_SAMPLE_PATH  # noqa: E402
from eth_price.train import run_training  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/eth_next_close_model.joblib"),
    )
    parser.add_argument("--metrics-path", type=Path, default=Path("report/experiment_results.csv"))
    parser.add_argument("--report-dir", type=Path, default=Path("report"))
    parser.add_argument("--figures-dir", type=Path, default=Path("report/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_training(
        data_path=args.data_path,
        max_rows=args.max_rows,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        report_dir=args.report_dir,
        figures_dir=args.figures_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
