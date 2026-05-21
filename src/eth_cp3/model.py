from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from .features import build_feature_frame, current_close

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


@lru_cache(maxsize=1)
def load_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact was not found: {path}")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_metadata(metadata_path: str | Path = DEFAULT_METADATA_PATH) -> dict[str, Any]:
    path = Path(metadata_path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return load_artifact()["metadata"]


def predict_next_close(candles: list[dict]) -> dict[str, Any]:
    artifact = load_artifact()
    metadata = artifact["metadata"]
    pipeline = artifact["pipeline"]
    feature_names = metadata["features"]
    feature_row = build_feature_frame(candles, feature_names)
    model_prediction = float(pipeline.predict(feature_row)[0])
    baseline_prediction = current_close(candles)

    baseline_rmse = metadata.get("baseline_test_metrics", {}).get("RMSE")
    model_rmse = metadata.get("test_metrics", {}).get("RMSE")
    if baseline_rmse is not None and model_rmse is not None and baseline_rmse <= model_rmse:
        recommended = baseline_prediction
        strategy = "persistence_baseline"
    else:
        recommended = model_prediction
        strategy = "ridge_model"

    if baseline_prediction:
        delta_model_pct = (model_prediction / baseline_prediction - 1.0) * 100
    else:
        delta_model_pct = None

    return {
        "recommended_prediction": recommended,
        "recommended_strategy": strategy,
        "model_prediction": model_prediction,
        "persistence_baseline_prediction": baseline_prediction,
        "delta_model_vs_current": model_prediction - baseline_prediction,
        "delta_model_vs_current_pct": delta_model_pct,
        "model_type": metadata.get("model_type"),
        "target": metadata.get("target"),
        "test_rmse_model": model_rmse,
        "test_rmse_baseline": baseline_rmse,
        "warning": (
            "This is an educational one-minute forecast. It is not financial advice "
            "and should not be used for trading decisions."
        ),
    }
