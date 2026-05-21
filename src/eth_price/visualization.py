"""Project visualizations used in the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from eth_price.config import TARGET_COLUMN, TIME_COLUMN


def _save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_price_history(df: pd.DataFrame, path: Path) -> None:
    sample = df.iloc[:: max(1, len(df) // 4000)].copy()
    plt.figure(figsize=(10, 4.8))
    plt.plot(sample[TIME_COLUMN], sample["Close"], label="Close")
    rolling_close = sample["Close"].rolling(120, min_periods=1).mean()
    plt.plot(sample[TIME_COLUMN], rolling_close, label="Rolling mean")
    plt.title("ETH/USD close price over time")
    plt.xlabel("Time")
    plt.ylabel("USD")
    plt.legend()
    _save_current(path)


def plot_return_distribution(df: pd.DataFrame, path: Path) -> None:
    returns = np.log(df["Close"]).diff().replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(8, 4.8))
    plt.hist(returns, bins=100)
    plt.title("Distribution of one-minute log returns")
    plt.xlabel("log return")
    plt.ylabel("count")
    _save_current(path)


def plot_correlation(
    df: pd.DataFrame,
    feature_cols: list[str],
    path: Path,
    max_cols: int = 18,
) -> None:
    correlations = df[[*feature_cols, TARGET_COLUMN]].corr(numeric_only=True)[TARGET_COLUMN]
    selected = correlations.abs().sort_values(ascending=False).head(max_cols).index.tolist()
    corr_matrix = df[selected].corr(numeric_only=True)
    plt.figure(figsize=(9, 7))
    image = plt.imshow(corr_matrix.to_numpy(), aspect="auto")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(selected)), selected, rotation=90, fontsize=8)
    plt.yticks(range(len(selected)), selected, fontsize=8)
    plt.title("Correlation matrix for strongest target-related features")
    _save_current(path)


def plot_predictions(
    times: pd.Series,
    y_true: np.ndarray,
    naive_pred: np.ndarray,
    final_pred: np.ndarray,
    path: Path,
    n_points: int = 600,
) -> None:
    n_points = min(n_points, len(y_true))
    plt.figure(figsize=(10, 4.8))
    plt.plot(times.iloc[:n_points], y_true[:n_points], label="Actual")
    plt.plot(times.iloc[:n_points], naive_pred[:n_points], label="Naive")
    plt.plot(times.iloc[:n_points], final_pred[:n_points], label="Final blend")
    plt.title("Actual vs predicted close price on test split")
    plt.xlabel("Time")
    plt.ylabel("USD")
    plt.legend()
    _save_current(path)


def plot_feature_importance(model, feature_cols: list[str], path: Path, top_n: int = 20) -> None:
    estimator = model.named_steps.get("model")
    importances = None
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        importances = np.abs(np.asarray(estimator.coef_)).ravel()
    if importances is None:
        return
    if len(importances) != len(feature_cols):
        # PCA changes the feature space; skip misleading chart.
        return
    table = pd.DataFrame({"feature": feature_cols, "importance": importances})
    table = table.sort_values("importance", ascending=False).head(top_n).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(table["feature"], table["importance"])
    plt.title("Top feature importances / absolute coefficients")
    plt.xlabel("importance")
    _save_current(path)


def plot_pca_projection(X: pd.DataFrame, y: pd.Series, path: Path, max_rows: int = 10_000) -> None:
    sample_X = X.iloc[-max_rows:].copy()
    sample_y = y.iloc[-max_rows:].copy()
    values = SimpleImputer(strategy="median").fit_transform(sample_X)
    values = StandardScaler().fit_transform(values)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(values)
    plt.figure(figsize=(7, 5.5))
    scatter = plt.scatter(
        components[:, 0],
        components[:, 1],
        c=sample_y.to_numpy(),
        s=5,
        alpha=0.7,
    )
    plt.colorbar(scatter, label="target close")
    plt.title("PCA projection of engineered features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    _save_current(path)


def plot_pca_explained_variance(X: pd.DataFrame, path: Path, max_rows: int = 20_000) -> None:
    sample_X = X.iloc[-max_rows:].copy()
    values = SimpleImputer(strategy="median").fit_transform(sample_X)
    values = StandardScaler().fit_transform(values)
    pca = PCA(random_state=42)
    pca.fit(values)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 4.8))
    plt.plot(np.arange(1, len(cumulative) + 1), cumulative)
    plt.axhline(0.95, linestyle="--", linewidth=1)
    plt.title("PCA cumulative explained variance")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    _save_current(path)
