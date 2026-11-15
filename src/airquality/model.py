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


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(config.FIGURE_DIR / filename, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_comparison(scenarios: dict, baselines: dict) -> None:
    frame = pd.DataFrame([
        {"scenario": name, "RMSE": block["test"]["rmse"], "R²": block["test"]["r2"]}
        for name, block in scenarios.items()
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, metric, baseline_key, colour in (
        (axes[0], "RMSE", "rmse", "#C44E52"),
        (axes[1], "R²", "r2", "#4C72B0"),
    ):
        sns.barplot(data=frame, x="scenario", y=metric, ax=ax, color=colour)
        ax.bar_label(ax.containers[0], fmt="%.3f" if metric == "R²" else "%.2f",
                     fontsize=9)
        baseline = baselines["persistence"][baseline_key]
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1.3,
                   label=f"persistence = {baseline:.2f}")
        ax.set_xlabel("")
        ax.set_title(f"Hold-out 2014 — {metric}")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Lagged PM2.5 only just beats 'tomorrow equals today'", fontsize=13)
    _save(fig, "09_scenario_comparison.png")


def plot_forecast_window(scenarios: dict) -> None:
    start = pd.Timestamp("2014-01-05")
    end = start + pd.Timedelta(days=21)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    for scenario, colour in (("weather_only", "#4C72B0"), ("with_lags", "#DD8452")):
        block = scenarios[scenario]
        timestamps = block["_timestamps"]
        window = (timestamps >= start) & (timestamps <= end)
        if window.any():
            ax.plot(timestamps[window], np.asarray(block["_predictions"])[window],
                    linewidth=1.6, color=colour, label=f"{scenario} prediction")

    timestamps = scenarios["with_lags"]["_timestamps"]
    actual = scenarios["with_lags"]["_y_test"]
    window = (timestamps >= start) & (timestamps <= end)
    ax.plot(timestamps[window], np.asarray(actual)[window], color="black",
            linewidth=1.6, label="observed")

    ax.axhline(config.EPISODE_THRESHOLD, color="grey", linestyle=":", linewidth=1)
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Three weeks of January 2014, hourly")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, "10_forecast_window.png")


def plot_monthly_error(scenarios: dict) -> None:
    block = scenarios["with_lags"]
    timestamps = pd.to_datetime(block["_timestamps"])
    error = np.asarray(block["_predictions"]) - np.asarray(block["_y_test"])
    frame = pd.DataFrame({"month": timestamps.dt.month, "abs_error": np.abs(error),
                          "error": error})

    grouped = frame.groupby("month").agg(mae=("abs_error", "mean"),
                                         bias=("error", "mean")).round(2)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(grouped.index - 0.18, grouped["mae"], width=0.34, color="#4C72B0",
           label="MAE")
    ax.bar(grouped.index + 0.18, grouped["bias"], width=0.34, color="#C44E52",
           label="bias (pred − actual)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month of 2014")
    ax.set_ylabel("µg/m³")
    ax.set_title("Winter errors are larger in absolute terms")
    ax.legend()
    _save(fig, "11_monthly_error.png")


def feature_importance(pipeline: Pipeline, top_n: int = 16) -> list[dict]:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = list(preprocessor.get_feature_names_out())

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:  # pragma: no cover
        return []

    order = np.argsort(values)[::-1][:top_n]
    return [{"feature": names[index].split("__", 1)[-1],
             "importance": round(float(values[index]), 5)} for index in order]


def plot_importance(importances: list[dict]) -> None:
    frame = pd.DataFrame(importances).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=frame, x="importance", y="feature", hue="feature", legend=False,
                palette="crest", ax=ax)
    ax.set_title("What the lagged forecast relies on")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    _save(fig, "12_feature_importance.png")


def plot_episode_detection(scenarios: dict, baselines: dict) -> None:
    rows = [
        {"source": name, "precision": block["episodes"]["precision"],
         "recall": block["episodes"]["recall"]}
        for name, block in scenarios.items()
    ]
    rows.append({"source": "persistence", "precision":
                 baselines["persistence_episodes"]["precision"],
                 "recall": baselines["persistence_episodes"]["recall"]})
    frame = pd.DataFrame(rows).melt(id_vars=["source"], var_name="metric",
                                    value_name="score")

    fig, ax = plt.subplots(figsize=(9, 4.4))
    sns.barplot(data=frame, x="source", y="score", hue="metric", ax=ax,
                palette=["#4C72B0", "#C44E52"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_title(f"Detecting hours above {config.EPISODE_THRESHOLD} µg/m³ in 2014")
    ax.legend(title="")
    _save(fig, "13_episode_detection.png")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train(df: pd.DataFrame) -> dict:
    started = time.perf_counter()

    scenarios = run_scenarios(df)
    baselines = baseline_metrics(df)
    importances = feature_importance(scenarios["with_lags"]["_pipeline"])

    plot_scenario_comparison(scenarios, baselines)
    plot_forecast_window(scenarios)
    plot_monthly_error(scenarios)
    plot_importance(importances)
    plot_episode_detection(scenarios, baselines)

    public = {
        name: {key: value for key, value in block.items() if not key.startswith("_")}
        for name, block in scenarios.items()
    }

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "rows": int(len(df)),
            "train_end": config.TRAIN_END,
            "test_start": config.TEST_START,
            "episode_threshold": config.EPISODE_THRESHOLD,
        },
        "split": "time-ordered: train <= 2013, test = 2014 (no random split)",
        "scenarios": public,
        "baselines": baselines,
        "feature_importance": importances,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }

    path = config.REPORT_DIR / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bundle = {
        "weather_only": scenarios["weather_only"]["_pipeline"],
        "with_lags": scenarios["with_lags"]["_pipeline"],
        "threshold": config.EPISODE_THRESHOLD,
        "metrics": metrics,
        "trained_at": metrics["generated_at"],
    }
    model_path = config.ARTIFACT_DIR / "pm25_model.joblib"
    joblib.dump(bundle, model_path)

    print(f"[model] weather_only rmse={public['weather_only']['test']['rmse']:.2f} "
          f"| with_lags rmse={public['with_lags']['test']['rmse']:.2f} "
          f"| persistence rmse={baselines['persistence']['rmse']:.2f}")
    print(f"[model] metrics -> {path.relative_to(config.ROOT)}")
    print(f"[model] model   -> {model_path.relative_to(config.ROOT)}")
    return metrics


def load_model(path=None) -> dict:
    path = path or (config.ARTIFACT_DIR / "pm25_model.joblib")
    return joblib.load(path)


def predict(df: pd.DataFrame, scenario: str = "weather_only",
            bundle: dict | None = None) -> np.ndarray:
    """Score a raw-format frame with the saved model."""
    bundle = bundle or load_model()
    include_lags = scenario == "with_lags"
    numeric, categorical = feature_columns(include_lags)
    frame = add_features(df)
    if include_lags:
        from .features import add_lag_features

        frame = add_lag_features(frame)
    return bundle[scenario].predict(frame[numeric + categorical])
