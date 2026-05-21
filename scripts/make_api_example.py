#!/usr/bin/env python
"""Create a JSON request example for the FastAPI deployment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SOURCE = Path("data/sample/ethusd_1m_sample.csv")
OUTPUT = Path("docs/assets/predict_request_example.json")


def main() -> None:
    df = pd.read_csv(SOURCE).tail(80)
    mapping = {
        "Open time": "open_time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Quote asset volume": "quote_asset_volume",
        "Number of trades": "number_of_trades",
        "Taker buy base asset volume": "taker_buy_base_asset_volume",
        "Taker buy quote asset volume": "taker_buy_quote_asset_volume",
    }
    request = {"candles": df[list(mapping)].rename(columns=mapping).to_dict(orient="records")}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(request, handle, ensure_ascii=False, indent=2)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
