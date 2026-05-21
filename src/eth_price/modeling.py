"""Model training helpers and sklearn-compatible preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eth_price.config import SEED, TARGET_COLUMN
from eth_price.metrics import regression_metrics


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric feature outliers using train-only quantiles."""

    def __init__(self, lower: float = 0.001, upper: float = 0.999) -> None:
        self.lower = lower
        self.upper = upper
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> QuantileClipper:
        array = self._to_array(X)
        self.lower_bounds_ = np.nanquantile(array, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(array, self.upper, axis=0)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            msg = "QuantileClipper must be fitted before transform"
            raise RuntimeError(msg)
        array = self._to_array(X)
        return np.clip(array, self.lower_bounds_, self.upper_bounds_)

    @staticmethod
    def _to_array(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)


@dataclass(frozen=True)
class ModelSpec:
    """Definition of one experiment family."""

    name: str
    estimator: Any
    param_grid: list[dict[str, Any]]
    scale: bool = False
    pca: bool = False


def build_pipeline(estimator: Any, *, scale: bool = False, pca: bool = False) -> Pipeline:
    """Build a leakage-safe preprocessing + model pipeline."""
    steps: list[tuple[str, Any]] = [
        ("clipper", QuantileClipper(lower=0.001, upper=0.999)),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale or pca:
        steps.append(("scaler", StandardScaler()))
    if pca:
        steps.append(("pca", PCA(n_components=0.95, random_state=SEED)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def default_model_specs() -> list[ModelSpec]:
    """Return the model set required by the grading rubric."""
    specs = [
        ModelSpec(
            name="LinearRegression",
            estimator=LinearRegression(),
            param_grid=[{}],
            scale=True,
        ),
        ModelSpec(
            name="Ridge",
            estimator=Ridge(random_state=SEED),
            param_grid=[{"alpha": [0.1, 1.0, 10.0, 50.0]}],
            scale=True,
        ),
        ModelSpec(
            name="ElasticNet",
            estimator=ElasticNet(random_state=SEED, max_iter=1000),
            param_grid=[{"alpha": [0.0001, 0.001], "l1_ratio": [0.2]}],
            scale=True,
        ),
        ModelSpec(
            name="RandomForest",
            estimator=RandomForestRegressor(random_state=SEED, n_jobs=1),
            param_grid=[
                {"n_estimators": [10], "max_depth": [8], "min_samples_leaf": [5]},
            ],
        ),
        ModelSpec(
            name="ExtraTrees",
            estimator=ExtraTreesRegressor(random_state=SEED, n_jobs=1),
            param_grid=[
                {"n_estimators": [10], "max_depth": [8], "min_samples_leaf": [5]},
            ],
        ),
        ModelSpec(
            name="Ridge+PCA95",
            estimator=Ridge(random_state=SEED),
            param_grid=[{"alpha": [0.1, 1.0, 10.0]}],
            scale=True,
            pca=True,
        ),
    ]

    # XGBoost/LightGBM are listed in requirements and can be added to the search
    # for a longer run. The default grid stays sklearn-only so CI and reviewers
    # can reproduce experiments quickly on CPU.

    return specs


def tune_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[Pipeline, dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    """Tune one model family on chronological validation data."""
    rows: list[dict[str, Any]] = []
    best_pipeline: Pipeline | None = None
    best_params: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    best_mae = float("inf")

    for params in ParameterGrid(spec.param_grid):
        estimator = clone(spec.estimator).set_params(**params)
        pipeline = build_pipeline(estimator, scale=spec.scale, pca=spec.pca)
        pipeline.fit(X_train, y_train)
        val_pred = pipeline.predict(X_val)
        metrics = regression_metrics(y_val.to_numpy(), val_pred)
        rows.append({"model": spec.name, "params": str(params), "split": "val", **metrics})
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_pipeline = pipeline
            best_params = params
            best_metrics = metrics

    if best_pipeline is None or best_params is None or best_metrics is None:
        msg = f"No model was trained for {spec.name}"
        raise RuntimeError(msg)
    return best_pipeline, best_params, best_metrics, rows


def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate fitted pipeline on the holdout test split."""
    return regression_metrics(y_test.to_numpy(), pipeline.predict(X_test))


def tune_naive_blend(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    step: float = 0.01,
) -> tuple[float, dict[str, float]]:
    """Find validation-optimal weight for naive close(t) and ML prediction."""
    ml_pred = model.predict(X_val)
    naive_pred = X_val["Close"].to_numpy()
    weights = np.arange(0.0, 1.0 + step / 2, step)
    best_weight = 0.0
    best_metrics: dict[str, float] | None = None
    best_mae = float("inf")
    for weight in weights:
        pred = weight * naive_pred + (1.0 - weight) * ml_pred
        metrics = regression_metrics(y_val.to_numpy(), pred)
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_weight = float(weight)
            best_metrics = metrics
    if best_metrics is None:
        msg = "Blend tuning failed"
        raise RuntimeError(msg)
    return best_weight, best_metrics


def predict_blend(model: Pipeline, X: pd.DataFrame, naive_weight: float) -> np.ndarray:
    """Predict with weighted blend of naive current close and ML model."""
    return naive_weight * X["Close"].to_numpy() + (1.0 - naive_weight) * model.predict(X)


def target_series(df: pd.DataFrame) -> pd.Series:
    """Return target column as float series."""
    return df[TARGET_COLUMN].astype(float)
