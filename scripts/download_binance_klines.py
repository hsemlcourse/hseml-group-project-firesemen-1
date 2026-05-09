#!/usr/bin/env python
"""Download Binance minute candles via public REST API.

This script is the reproducible data parser for the project. It is intentionally
kept separate from Kaggle data: Kaggle was convenient for CP1, while this parser
shows how to collect the same OHLCV candle structure from the source exchange.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
COLUMNS = [
    "Open time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close time",
    "Quote asset volume",
    "Number of trades",
    "Taker buy base asset volume",
    "Taker buy quote asset volume",
    "Ignore",
]


def parse_datetime(value: str) -> int:
    """Convert ISO datetime string to milliseconds since epoch."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    pause: float = 0.2,
) -> pd.DataFrame:
    """Fetch klines with pagination and basic retry logic."""
    all_rows: list[list[Any]] = []
    current_start = start_ms
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }
        for attempt in range(3):
            response = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
            if response.status_code == 200:
                break
            time.sleep(pause * (attempt + 1))
        response.raise_for_status()
        rows = response.json()
        if not rows:
            break
        all_rows.extend(rows)
        current_start = int(rows[-1][6]) + 1
        time.sleep(pause)
    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms", utc=True).dt.tz_localize(None)
    df["Close time"] = pd.to_datetime(df["Close time"], unit="ms", utc=True).dt.tz_localize(None)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETHUSDT", help="Binance symbol, e.g. ETHUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--start", required=True, help="ISO datetime, e.g. 2025-01-01T00:00:00Z")
    parser.add_argument("--end", required=True, help="ISO datetime, e.g. 2025-01-07T00:00:00Z")
    parser.add_argument("--output", type=Path, default=Path("data/raw/ETHUSDT_1m_Binance.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = fetch_klines(
        symbol=args.symbol,
        interval=args.interval,
        start_ms=parse_datetime(args.start),
        end_ms=parse_datetime(args.end),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df):,} rows to {args.output}")


if __name__ == "__main__":
    main()
