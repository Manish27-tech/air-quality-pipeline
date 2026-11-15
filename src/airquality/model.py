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
