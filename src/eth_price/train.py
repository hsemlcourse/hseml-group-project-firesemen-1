"""End-to-end training pipeline used by scripts/train.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from eth_price.config import (
    DEFAULT_FIGURES_DIR,
    DEFAULT_HORIZON,
    DEFAULT_MAX_ROWS,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPORT_DIR,
    DEFAULT_SAMPLE_PATH,
    SEED,
    TARGET_COLUMN,
    TIME_COLUMN,
)
from eth_price.data import (
    assert_no_temporal_overlap,
    clean_raw_data,
    describe_dataframe,
    detect_return_outliers,
    load_raw_data,
    train_val_test_split_by_time,
)
from eth_price.features import (
    BASE_NUMERIC_COLUMNS,
    feature_columns,
    make_features,
    raw_baseline_frame,
)
from eth_price.metrics import regression_metrics
from eth_price.modeling import (
    ModelSpec,
    build_pipeline,
    default_model_specs,
    evaluate_on_test,
    predict_blend,
    target_series,
    tune_model,
    tune_naive_blend,
)
from eth_price.visualization import (
    plot_correlation,
    plot_feature_importance,
    plot_pca_explained_variance,
    plot_pca_projection,
    plot_predictions,
    plot_price_history,
    plot_return_distribution,
)


def _format_params(params: dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


def _evaluate_raw_linear_baseline(clean_df: pd.DataFrame) -> dict[str, Any]:
    baseline_df = raw_baseline_frame(clean_df, horizon=DEFAULT_HORIZON)
    train, val, test = train_val_test_split_by_time(baseline_df)
    assert_no_temporal_overlap([train, val, test])
    X_train = train[BASE_NUMERIC_COLUMNS]
    X_val = val[BASE_NUMERIC_COLUMNS]
    X_test = test[BASE_NUMERIC_COLUMNS]
    y_train = target_series(train)
    y_val = target_series(val)
    y_test = target_series(test)
    pipeline = build_pipeline(LinearRegression(), scale=True)
    pipeline.fit(X_train, y_train)
    return {
        "model": "Baseline LinearRegression without FE",
        "params": "{}",
        "hypothesis": (
            "Простая линейная модель на исходных OHLCV даёт точку отсчёта без feature engineering."
        ),
        "val_metrics": regression_metrics(y_val.to_numpy(), pipeline.predict(X_val)),
        "test_metrics": regression_metrics(y_test.to_numpy(), pipeline.predict(X_test)),
    }


def run_training(
    data_path: str | Path = DEFAULT_SAMPLE_PATH,
    max_rows: int | None = DEFAULT_MAX_ROWS,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> dict[str, Any]:
    """Run full data-cleaning, feature, tuning, ensemble, and reporting pipeline."""
    np.random.seed(SEED)
    data_path = Path(data_path)
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    report_dir = Path(report_dir)
    figures_dir = Path(figures_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data(data_path, max_rows=max_rows)
    clean_df, cleaning_report = clean_raw_data(raw_df)
    raw_description = describe_dataframe(clean_df)
    outlier_report = detect_return_outliers(clean_df)
    features_df = make_features(clean_df, horizon=DEFAULT_HORIZON)
    feature_cols = feature_columns(features_df)

    train, val, test = train_val_test_split_by_time(features_df)
    assert_no_temporal_overlap([train, val, test])
    X_train = train[feature_cols]
    X_val = val[feature_cols]
    X_test = test[feature_cols]
    y_train = target_series(train)
    y_val = target_series(val)
    y_test = target_series(test)

    rows: list[dict[str, Any]] = []
    raw_baseline = _evaluate_raw_linear_baseline(clean_df)
    rows.append(
        {
            "model": raw_baseline["model"],
            "hypothesis": raw_baseline["hypothesis"],
            "params": raw_baseline["params"],
            "val_mae": raw_baseline["val_metrics"]["mae"],
            "val_rmse": raw_baseline["val_metrics"]["rmse"],
            "val_r2": raw_baseline["val_metrics"]["r2"],
            "test_mae": raw_baseline["test_metrics"]["mae"],
            "test_rmse": raw_baseline["test_metrics"]["rmse"],
            "test_r2": raw_baseline["test_metrics"]["r2"],
            "comment": "Baseline без новых признаков.",
        }
    )

    naive_val_pred = X_val["Close"].to_numpy()
    naive_test_pred = X_test["Close"].to_numpy()
    naive_val_metrics = regression_metrics(y_val.to_numpy(), naive_val_pred)
    naive_test_metrics = regression_metrics(y_test.to_numpy(), naive_test_pred)
    rows.append(
        {
            "model": "Naive close(t)",
            "hypothesis": (
                "На минутном горизонте лучшая простая эвристика - "
                "цена следующей минуты примерно равна текущей."
            ),
            "params": "{}",
            "val_mae": naive_val_metrics["mae"],
            "val_rmse": naive_val_metrics["rmse"],
            "val_r2": naive_val_metrics["r2"],
            "test_mae": naive_test_metrics["mae"],
            "test_rmse": naive_test_metrics["rmse"],
            "test_r2": naive_test_metrics["r2"],
            "comment": "Сильный финансовый baseline для горизонта 1 минута.",
        }
    )

    best_model_name = ""
    best_model = None
    best_params: dict[str, Any] = {}
    best_val_mae = float("inf")
    fitted_models: dict[str, Any] = {}
    search_details: list[dict[str, Any]] = []

    for spec in default_model_specs():
        pipeline, params, val_metrics, search_rows = tune_model(
            spec,
            X_train,
            y_train,
            X_val,
            y_val,
        )
        test_metrics = evaluate_on_test(pipeline, X_test, y_test)
        fitted_models[spec.name] = pipeline
        search_details.extend(search_rows)
        rows.append(
            {
                "model": spec.name,
                "hypothesis": _hypothesis_for_model(spec),
                "params": _format_params(params),
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "comment": "Лучшие параметры выбраны по validation MAE без использования test.",
            }
        )
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_model_name = spec.name
            best_model = pipeline
            best_params = params

    if best_model is None:
        msg = "No ML model was fitted"
        raise RuntimeError(msg)

    naive_weight, blend_val_metrics = tune_naive_blend(best_model, X_val, y_val, step=0.01)
    blend_test_pred = predict_blend(best_model, X_test, naive_weight=naive_weight)
    blend_test_metrics = regression_metrics(y_test.to_numpy(), blend_test_pred)
    rows.append(
        {
            "model": f"Weighted blend: naive + {best_model_name}",
            "hypothesis": "Взвешенный ансамбль стабилизирует ML-прогноз сильным наивным baseline.",
            "params": _format_params(
                {"naive_weight": naive_weight, "ml_model": best_model_name, **best_params}
            ),
            "val_mae": blend_val_metrics["mae"],
            "val_rmse": blend_val_metrics["rmse"],
            "val_r2": blend_val_metrics["r2"],
            "test_mae": blend_test_metrics["mae"],
            "test_rmse": blend_test_metrics["rmse"],
            "test_r2": blend_test_metrics["r2"],
            "comment": "Финальная модель: ансамбль/блендинг, вес найден на validation.",
        }
    )

    results_df = pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)
    results_df.to_csv(metrics_path, index=False)
    pd.DataFrame(search_details).to_csv(
        report_dir / "hyperparameter_search_details.csv",
        index=False,
    )

    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": feature_cols,
        "naive_weight": naive_weight,
        "horizon": DEFAULT_HORIZON,
        "target_column": TARGET_COLUMN,
        "seed": SEED,
        "metrics": results_df.to_dict(orient="records"),
        "cleaning_report": cleaning_report,
        "outlier_report": outlier_report,
        "data_description": raw_description,
    }
    joblib.dump(artifact, model_path)

    with (report_dir / "cleaning_report.json").open("w", encoding="utf-8") as handle:
        json.dump(cleaning_report, handle, indent=2, ensure_ascii=False)
    with (report_dir / "outlier_report.json").open("w", encoding="utf-8") as handle:
        json.dump(outlier_report, handle, indent=2, ensure_ascii=False)
    with (report_dir / "dataset_description.json").open("w", encoding="utf-8") as handle:
        json.dump(raw_description, handle, indent=2, ensure_ascii=False)

    plot_price_history(clean_df, figures_dir / "price_history.png")
    plot_return_distribution(clean_df, figures_dir / "return_distribution.png")
    plot_correlation(features_df, feature_cols, figures_dir / "correlation_matrix.png")
    plot_predictions(
        test[TIME_COLUMN],
        y_test.to_numpy(),
        naive_test_pred,
        blend_test_pred,
        figures_dir / "predictions_test.png",
    )
    plot_feature_importance(best_model, feature_cols, figures_dir / "feature_importance.png")
    plot_pca_projection(X_train, y_train, figures_dir / "pca_projection.png")
    plot_pca_explained_variance(X_train, figures_dir / "pca_explained_variance.png")

    return {
        "data_path": str(data_path),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "best_ml_model": best_model_name,
        "naive_weight": naive_weight,
        "rows_after_cleaning": len(clean_df),
        "rows_after_features": len(features_df),
        "n_features": len(feature_cols),
        "results": results_df.to_dict(orient="records"),
    }


def _hypothesis_for_model(spec: ModelSpec) -> str:
    descriptions = {
        "LinearRegression": (
            "Линейная зависимость лагов и текущих цен может быть "
            "достаточной для очень короткого горизонта."
        ),
        "Ridge": (
            "L2-регуляризация должна стабилизировать линейную модель на скоррелированных признаках."
        ),
        "ElasticNet": (
            "L1+L2-регуляризация может занулить слабые признаки и снизить переобучение."
        ),
        "RandomForest": (
            "Нелинейный ансамбль деревьев может поймать режимы рынка и взаимодействия признаков."
        ),
        "ExtraTrees": (
            "Сильно рандомизированные деревья могут быть устойчивее "
            "RandomForest на шумных минутных данных."
        ),
        "HistGradientBoosting": (
            "Бустинг по деревьям может лучше приблизить нелинейные зависимости в OHLCV."
        ),
        "Ridge+PCA95": (
            "PCA уменьшает размерность скоррелированных лагов и rolling-признаков перед Ridge."
        ),
        "XGBoost": (
            "Градиентный бустинг с регуляризацией может улучшить качество на табличных признаках."
        ),
        "LightGBM": (
            "LightGBM может эффективно обработать много числовых признаков и нелинейности."
        ),
    }
    fallback = "Проверка альтернативной модели на одинаковом хронологическом split."
    return descriptions.get(spec.name, fallback)
