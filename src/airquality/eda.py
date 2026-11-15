"""Exploratory data analysis for the Beijing PM2.5 series."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from . import config  # noqa: E402
from .clean import missing_runs  # noqa: E402
from .features import SEASON_BY_MONTH, add_features  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["figure.autolayout"] = True


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def compute_summary(df: pd.DataFrame) -> dict:
    engineered = add_features(df)
    target = df[config.TARGET]
    timestamps = pd.to_datetime(df[config.TIMESTAMP])
    episodes = target > config.EPISODE_THRESHOLD

    yearly = (
        df.groupby(timestamps.dt.year)[config.TARGET]
        .agg(["mean", "median", "max", "count"]).round(2)
    )
    monthly = (
        engineered.groupby(engineered[config.TIMESTAMP].dt.month)[config.TARGET]
        .mean().round(2)
    )
    hourly = (
        engineered.groupby("hour")[config.TARGET].mean().round(2)
    )
    by_direction = (
        engineered.groupby("cbwd")[config.TARGET]
        .agg(["mean", "size"]).round(2)
    )
    by_season = (
        engineered.groupby("season")[config.TARGET]
        .agg(["mean", "median", "size"]).round(2)
    )

    numeric_columns = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir",
                       "dew_point_depression", "log_wind", "pressure_anomaly",
                       config.TARGET]
    correlations = (
        engineered[numeric_columns].corr(numeric_only=True)[config.TARGET]
        .drop(config.TARGET)
        .sort_values(key=lambda series: series.abs(), ascending=False)
    )

    return {
        "rows": int(len(df)),
        "start": str(timestamps.min()),
        "end": str(timestamps.max()),
        "mean_pm25": round(float(target.mean()), 2),
        "median_pm25": float(target.median()),
        "p95_pm25": float(target.quantile(0.95)),
        "max_pm25": float(target.max()),
        "missing": missing_runs(df),
        "episodes": {
            "over_150": int(episodes.sum()),
            "over_150_share": round(float(episodes.mean()), 4),
            "over_250": int((target > 250).sum()),
            "over_250_share": round(float((target > 250).mean()), 4),
        },
        "yearly": {
            str(int(year)): {"mean": float(row["mean"]),
                             "median": float(row["median"]),
                             "max": float(row["max"]),
                             "hours": int(row["count"])}
            for year, row in yearly.iterrows()
        },
        "monthly_mean": {int(month): float(value) for month, value in monthly.items()},
        "hourly_mean": {int(hour): float(value) for hour, value in hourly.items()},
        "by_direction": {
            str(level): {"mean": float(row["mean"]), "hours": int(row["size"])}
            for level, row in by_direction.iterrows()
        },
        "by_season": {
            str(level): {"mean": float(row["mean"]), "median": float(row["median"]),
                         "hours": int(row["size"])}
            for level, row in by_season.iterrows()
        },
        "weekend_vs_weekday": {
            "weekend": round(float(engineered.loc[engineered["is_weekend"] == 1,
                                                  config.TARGET].mean()), 2),
            "weekday": round(float(engineered.loc[engineered["is_weekend"] == 0,
                                                  config.TARGET].mean()), 2),
        },
        "top_correlations_with_target": [
            {"feature": name, "correlation": round(float(value), 4)}
            for name, value in correlations.head(8).items()
        ],
    }
