"""PM2.5 regression, time-ordered evaluation and pollution-episode detection."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import loguniform, randint  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (  # noqa: E402
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_validate,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from . import config  # noqa: E402
from .features import add_features, build_feature_matrix, feature_columns  # noqa: E402

plt.rcParams["figure.autolayout"] = True


# --------------------------------------------------------------------------- #
# Preprocessing and candidates
# --------------------------------------------------------------------------- #
def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric_pipeline, numeric), ("cat", categorical_pipeline, categorical)],
        remainder="drop", verbose_feature_names_out=True,
    )


def candidate_models() -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_depth=8, l2_regularization=1.0,
            random_state=config.RANDOM_STATE,
        ),
    }


SEARCH_SPACES = {
    "ridge": {"model__alpha": loguniform(1e-1, 1e3)},
    "random_forest": {
        "model__n_estimators": randint(250, 600),
        "model__max_depth": [None, 14, 22],
        "model__min_samples_leaf": randint(1, 6),
        "model__max_features": [0.4, 0.7, 1.0],
    },
    "hist_gradient_boosting": {
        "model__max_iter": randint(250, 600),
        "model__learning_rate": loguniform(0.02, 0.18),
        "model__max_depth": randint(4, 12),
        "model__l2_regularization": loguniform(1e-2, 10),
    },
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, predictions) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    error = predictions - y_true
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, predictions))), 3),
        "mae": round(float(mean_absolute_error(y_true, predictions)), 3),
        "r2": round(float(r2_score(y_true, predictions)), 4),
        "bias": round(float(error.mean()), 3),
        "within_25": round(float((np.abs(error) <= 25).mean()), 4),
        "hours": int(len(y_true)),
    }


def episode_metrics(y_true, predictions,
                    threshold: float = config.EPISODE_THRESHOLD) -> dict:
    """Detection quality for heavy-pollution hours.

    A missed episode is a public-health failure, so recall is read first; the
    precision cost of over-alerting is reported alongside it.
    """
    actual = (np.asarray(y_true, dtype=float) > threshold).astype(int)
    flagged = (np.asarray(predictions, dtype=float) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, flagged, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "observed_episodes": int(actual.sum()),
        "flagged_hours": int(flagged.sum()),
        "precision": round(float(precision_score(actual, flagged, zero_division=0)), 4),
        "recall": round(float(recall_score(actual, flagged, zero_division=0)), 4),
        "f1": round(float(f1_score(actual, flagged, zero_division=0)), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


# --------------------------------------------------------------------------- #
# Comparison and tuning
# --------------------------------------------------------------------------- #
def _folds() -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=config.CV_FOLDS)


def compare_models(X_train, y_train, include_lags: bool) -> list[dict]:
    numeric, categorical = feature_columns(include_lags)
    scoring = {"r2": "r2", "neg_rmse": "neg_root_mean_squared_error",
               "neg_mae": "neg_mean_absolute_error"}

    rows = []
    print(f"[model] cross-validating {4 if include_lags else 3} candidate families "
          f"({'with' if include_lags else 'without'} lags)")
    for name, estimator in candidate_models().items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", estimator),
        ])
        scores = cross_validate(pipeline, X_train, y_train, cv=_folds(),
                                scoring=scoring, n_jobs=1, error_score="raise")
        rows.append({
            "model": name,
            "r2_mean": round(float(scores["test_r2"].mean()), 4),
            "rmse_mean": round(float(-scores["test_neg_rmse"].mean()), 3),
            "mae_mean": round(float(-scores["test_neg_mae"].mean()), 3),
            "fit_seconds": round(float(scores["fit_time"].mean()), 1),
        })
        print(f"[model]   {name:<24} r2={rows[-1]['r2_mean']:.4f} "
              f"rmse={rows[-1]['rmse_mean']:.2f}")
    return sorted(rows, key=lambda row: row["rmse_mean"])


def tune_model(name: str, X_train, y_train, include_lags: bool
               ) -> tuple[Pipeline, dict]:
    numeric, categorical = feature_columns(include_lags)
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical)),
        ("model", candidate_models()[name]),
    ])
    search = RandomizedSearchCV(
        pipeline, SEARCH_SPACES[name], n_iter=config.SEARCH_ITER,
        scoring="neg_root_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=config.SEARCH_FOLDS),
        random_state=config.RANDOM_STATE, n_jobs=1, refit=True,
    )
    search.fit(X_train, y_train)
    params = {key.replace("model__", ""):
              (value.item() if hasattr(value, "item") else value)
              for key, value in search.best_params_.items()}
    print(f"[model] tuned {name}: cv rmse={-search.best_score_:.2f}")
    return search.best_estimator_, params


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
def time_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    timestamps = pd.to_datetime(df.loc[X.index, config.TIMESTAMP])
    is_train = (timestamps < pd.Timestamp(config.TEST_START)).to_numpy()
    return (
        X.loc[is_train], X.loc[~is_train],
        y.loc[is_train], y.loc[~is_train],
        timestamps,
    )


def run_scenarios(df: pd.DataFrame) -> dict:
    """Fit both feature settings and record their hold-out behaviour."""
    results: dict[str, dict] = {}

    for scenario, description in config.SCENARIOS.items():
        include_lags = scenario == "with_lags"
        X, y = build_feature_matrix(df, include_lags=include_lags)
        X_train, X_test, y_train, y_test, timestamps = time_split(df, X, y)

        comparison = compare_models(X_train, y_train, include_lags)
        best_name = comparison[0]["model"]
        pipeline, params = tune_model(best_name, X_train, y_train, include_lags)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        results[scenario] = {
            "description": description,
            "include_lags": include_lags,
            "features": int(X.shape[1]),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "comparison": comparison,
            "model": best_name,
            "params": params,
            "test": regression_metrics(y_test, predictions),
            "episodes": episode_metrics(y_test, predictions),
            "_pipeline": pipeline,
            "_y_test": y_test,
            "_predictions": predictions,
            "_timestamps": pd.to_datetime(df.loc[X_test.index, config.TIMESTAMP]),
        }
        print(f"[model] {scenario:<12} rmse={results[scenario]['test']['rmse']:.2f} "
              f"r2={results[scenario]['test']['r2']:.3f} "
              f"episode recall={results[scenario]['episodes']['recall']:.3f}")
    return results


def baseline_metrics(df: pd.DataFrame) -> dict:
    """Two reference forecasts that any model must beat.

    - *climatology*: predict the 2010-2013 average, the weather-only floor;
    - *persistence*: predict the previous hour, the forecast floor once lags exist.
    """
    X, y = build_feature_matrix(df, include_lags=True)
    X_train, X_test, y_train, y_test, _ = time_split(df, X, y)

    climatology = np.full(len(y_test), float(y_train.mean()))
    persistence = X_test["pm_lag_1"].to_numpy()

    use = ~np.isnan(persistence)
    return {
        "climatology": regression_metrics(y_test, climatology),
        "persistence": regression_metrics(y_test.to_numpy()[use], persistence[use]),
        "persistence_episodes": episode_metrics(y_test.to_numpy()[use], persistence[use]),
    }
