"""Project configuration constants."""

from __future__ import annotations

from pathlib import Path

SEED = 812742
TARGET_COLUMN = "target_close_next"
TIME_COLUMN = "Open time"
DEFAULT_HORIZON = 1
DEFAULT_MAX_ROWS = 120_000

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "ethusd_1m_sample.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "eth_next_close_model.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "report" / "experiment_results.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "report"
DEFAULT_FIGURES_DIR = DEFAULT_REPORT_DIR / "figures"
