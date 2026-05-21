"""Data loading and raw-data validation utilities."""

from __future__ import annotations

import csv
import zipfile
from collections import deque
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from eth_price.config import TIME_COLUMN

DROP_COLUMNS = ["Close time", "Ignore"]
NUMERIC_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Quote asset volume",
    "Number of trades",
    "Taker buy base asset volume",
    "Taker buy quote asset volume",
]


def _tail_lines_from_zip(path: Path, max_rows: int) -> list[str]:
    """Read only the header and the last max_rows rows from a one-file ZIP CSV."""
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            msg = f"No CSV file found inside {path}"
            raise ValueError(msg)
        with archive.open(csv_names[0], "r") as handle:
            header = handle.readline().decode("utf-8").strip()
            rows: deque[str] = deque(maxlen=max_rows)
            for raw_line in handle:
                rows.append(raw_line.decode("utf-8").strip())
    return [header, *rows]


def _tail_lines_from_csv(path: Path, max_rows: int) -> list[str]:
    """Read only the header and last max_rows lines from an uncompressed CSV."""
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        rows: deque[str] = deque(maxlen=max_rows)
        for line in handle:
            rows.append(line.strip())
    return [header, *rows]


def load_raw_data(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load raw ETH minute candles.

    If ``max_rows`` is set, the loader keeps the most recent rows. That makes
    experiments reproducible and fast while preserving chronological order.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Data file not found: {path}"
        raise FileNotFoundError(msg)

    if max_rows is None:
        return pd.read_csv(path)

    if path.suffix.lower() == ".zip":
        lines = _tail_lines_from_zip(path, max_rows=max_rows)
        reader = csv.reader(lines)
        header = next(reader)
        return pd.DataFrame(reader, columns=header)

    lines = _tail_lines_from_csv(path, max_rows=max_rows)
    reader = csv.reader(lines)
    header = next(reader)
    return pd.DataFrame(reader, columns=header)


def clean_raw_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Clean raw exchange candles and return a compact cleaning report.

    The function handles impossible values globally: wrong dates, duplicate
    timestamps, non-numeric fields, and invalid OHLCV constraints. Distributional
    outliers are not removed here; they are clipped inside sklearn pipelines and
    fitted on train only to avoid leakage.
    """
    df = raw_df.copy()
    report: dict[str, object] = {"rows_initial": int(len(df))}

    if TIME_COLUMN not in df.columns:
        msg = f"Required time column '{TIME_COLUMN}' is missing"
        raise ValueError(msg)

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    report["bad_timestamps"] = int(df[TIME_COLUMN].isna().sum())
    df = df.dropna(subset=[TIME_COLUMN])

    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            msg = f"Required numeric column '{column}' is missing"
            raise ValueError(msg)
        df[column] = pd.to_numeric(df[column], errors="coerce")

    numeric_na = df[NUMERIC_COLUMNS].isna().sum().astype(int).to_dict()
    report["numeric_missing_before_drop"] = numeric_na
    df = df.dropna(subset=NUMERIC_COLUMNS)

    df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])
    before_duplicates = len(df)
    df = df.sort_values(TIME_COLUMN).drop_duplicates(subset=[TIME_COLUMN], keep="last")
    report["duplicate_timestamps_removed"] = int(before_duplicates - len(df))

    invalid_mask = (
        (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
        | (df["Volume"] < 0)
        | (df["Quote asset volume"] < 0)
        | (df["Number of trades"] < 0)
        | (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
    )
    report["invalid_ohlcv_rows_removed"] = int(invalid_mask.sum())
    df = df.loc[~invalid_mask].reset_index(drop=True)

    expected_delta = pd.Timedelta(minutes=1)
    gaps = df[TIME_COLUMN].diff().dropna()
    report["time_gaps_not_1_minute"] = int((gaps != expected_delta).sum())
    report["rows_final"] = int(len(df))
    report["time_min"] = str(df[TIME_COLUMN].min())
    report["time_max"] = str(df[TIME_COLUMN].max())
    report["columns_final"] = list(df.columns)

    return df, report


def train_val_test_split_by_time(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/validation/test split without shuffling."""
    if not 0 < train_size < 1 or not 0 < val_size < 1:
        msg = "train_size and val_size must be in (0, 1)"
        raise ValueError(msg)
    if train_size + val_size >= 1:
        msg = "train_size + val_size must be smaller than 1"
        raise ValueError(msg)

    df_sorted = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    n_rows = len(df_sorted)
    train_end = int(n_rows * train_size)
    val_end = int(n_rows * (train_size + val_size))

    train = df_sorted.iloc[:train_end].copy()
    val = df_sorted.iloc[train_end:val_end].copy()
    test = df_sorted.iloc[val_end:].copy()
    return train, val, test


def assert_no_temporal_overlap(parts: Iterable[pd.DataFrame]) -> None:
    """Raise an error if chronological split parts overlap in time."""
    previous_max: pd.Timestamp | None = None
    for part in parts:
        if part.empty:
            msg = "Split part is empty"
            raise ValueError(msg)
        current_min = part[TIME_COLUMN].min()
        current_max = part[TIME_COLUMN].max()
        if previous_max is not None and current_min <= previous_max:
            msg = "Temporal leakage: split parts overlap or are not ordered"
            raise ValueError(msg)
        previous_max = current_max


def describe_dataframe(df: pd.DataFrame) -> dict[str, object]:
    """Return compact dataset statistics for README/report generation."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "time_min": str(df[TIME_COLUMN].min()) if TIME_COLUMN in df else None,
        "time_max": str(df[TIME_COLUMN].max()) if TIME_COLUMN in df else None,
        "missing_values": df.isna().sum().astype(int).to_dict(),
    }


def detect_return_outliers(df: pd.DataFrame, z_threshold: float = 6.0) -> dict[str, float | int]:
    """Detect unusually large one-minute log returns for data-quality reporting."""
    log_returns = np.log(df["Close"]).diff().replace([np.inf, -np.inf], np.nan).dropna()
    if log_returns.empty:
        return {"count": 0, "share": 0.0, "z_threshold": z_threshold}
    z_scores = (log_returns - log_returns.mean()) / (log_returns.std(ddof=0) + 1e-12)
    count = int((z_scores.abs() > z_threshold).sum())
    return {
        "count": count,
        "share": float(count / len(log_returns)),
        "z_threshold": z_threshold,
        "max_abs_log_return": float(log_returns.abs().max()),
    }
