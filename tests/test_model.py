"""Tests for metrics, preprocessing and the time-ordered split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from airquality import config
from airquality.features import build_feature_matrix, feature_columns
from airquality.model import (
    build_preprocessor,
    candidate_models,
    episode_metrics,
    feature_importance,
    regression_metrics,
    time_split,
)


def test_regression_metrics_on_perfect_predictions() -> None:
    y_true = np.array([50.0, 100.0, 150.0])
    metrics = regression_metrics(y_true, y_true)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["within_25"] == 1.0
    assert metrics["bias"] == 0.0


def test_regression_metrics_report_bias() -> None:
    y_true = np.array([50.0, 60.0])
    metrics = regression_metrics(y_true, y_true + 10)

    assert metrics["bias"] == 10.0
    assert metrics["rmse"] == 10.0


def test_episode_metrics_counts_every_cell() -> None:
    y_true = np.array([50, 200, 300, 40])
    predictions = np.array([60, 210, 100, 30])
    metrics = episode_metrics(y_true, predictions, threshold=150)

    assert metrics["observed_episodes"] == 2
    assert metrics["flagged_hours"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5


def test_episode_metrics_handles_no_episodes() -> None:
    metrics = episode_metrics(np.array([10, 20]), np.array([15, 25]))

    assert metrics["observed_episodes"] == 0
    assert metrics["f1"] == 0.0


def test_preprocessor_produces_finite_matrix(clean_frame: pd.DataFrame) -> None:
    X, y = build_feature_matrix(clean_frame, include_lags=False)
    numeric, categorical = feature_columns(False)
    transformed = build_preprocessor(numeric, categorical).fit_transform(X, y)

    assert transformed.shape[0] == len(X)
    assert np.isfinite(transformed).all()


def test_time_split_keeps_the_future_out(clean_frame: pd.DataFrame,
                                         monkeypatch) -> None:
    monkeypatch.setattr(config, "TEST_START", "2012-01-31 00:00:00")
    X, y = build_feature_matrix(clean_frame, include_lags=True)
    X_train, X_test, y_train, y_test, timestamps = time_split(clean_frame, X, y)

    assert len(X_test) > 0
    train_times = timestamps.loc[X_train.index]
    test_times = timestamps.loc[X_test.index]
    assert train_times.max() < test_times.min()
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_ridge_pipeline_predicts_finite_values(clean_frame: pd.DataFrame) -> None:
    X, y = build_feature_matrix(clean_frame, include_lags=False)
    numeric, categorical = feature_columns(False)
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical)),
        ("model", Ridge(alpha=1.0)),
    ]).fit(X, y)

    predictions = pipeline.predict(X)
    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()


def test_feature_importance_is_sorted(clean_frame: pd.DataFrame) -> None:
    X, y = build_feature_matrix(clean_frame, include_lags=True)
    numeric, categorical = feature_columns(True)
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical)),
        ("model", candidate_models()["random_forest"]),
    ]).fit(X, y)

    importances = feature_importance(pipeline, top_n=5)
    values = [item["importance"] for item in importances]

    assert len(importances) == 5
    assert values == sorted(values, reverse=True)
    assert all("__" not in item["feature"] for item in importances)


def test_every_candidate_is_a_distinct_estimator() -> None:
    models = candidate_models()

    assert set(models) == {"ridge", "random_forest", "hist_gradient_boosting"}
    assert len({type(model).__name__ for model in models.values()}) == 3
    assert pytest.approx(config.EPISODE_THRESHOLD) == 150
