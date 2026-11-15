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


# --------------------------------------------------------------------------- #
# Distribution and time figures
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, filename: str) -> None:
    path = config.FIGURE_DIR / filename
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] figure -> {path.relative_to(config.ROOT)}")


def plot_target_distribution(df: pd.DataFrame) -> None:
    values = df[config.TARGET].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

    axes[0].hist(values.clip(upper=400), bins=80, color="#4C72B0", edgecolor="none")
    axes[0].set_yscale("log")
    for threshold, colour in ((150, "#DD8452"), (250, "#C44E52")):
        axes[0].axvline(threshold, color=colour, linestyle="--", linewidth=1.2)
        axes[0].text(threshold + 4, axes[0].get_ylim()[1] * 0.4,
                     f"{threshold}", color=colour, fontsize=9)
    axes[0].set_xlabel("PM2.5 (µg/m³, clipped at 400)")
    axes[0].set_ylabel("Hours (log scale)")
    axes[0].set_title("Highly right-skewed with a long pollution tail")

    by_season = df.assign(season=pd.to_datetime(df[config.TIMESTAMP]).dt.month
                          .map(SEASON_BY_MONTH))
    order = ["Winter", "Spring", "Summer", "Autumn"]
    sns.boxplot(data=by_season, x="season", y=config.TARGET, order=order, ax=axes[1],
                showfliers=False, hue="season", legend=False,
                palette=[config.SEASON_COLOURS[s] for s in order])
    axes[1].set_ylim(0, 350)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("PM2.5 (µg/m³)")
    axes[1].set_title("Winter is the pollution season")
    _save(fig, "01_target_distribution.png")


def plot_series(df: pd.DataFrame) -> None:
    series = (df.set_index(config.TIMESTAMP)[config.TARGET]
              .resample("D").mean())
    rolling = series.rolling(30, min_periods=10).mean()

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(series.index, series.values, color="#9FB6CD", linewidth=0.7,
            label="daily mean")
    ax.plot(rolling.index, rolling.values, color="#C44E52", linewidth=2,
            label="30-day rolling mean")
    ax.axhline(config.EPISODE_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.text(series.index[10], config.EPISODE_THRESHOLD + 8,
            f"heavy pollution threshold = {config.EPISODE_THRESHOLD}",
            fontsize=8)
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Five years of hourly PM2.5 at the US Embassy monitor")
    ax.legend(loc="upper right")
    _save(fig, "02_timeseries.png")


def plot_seasonal_diurnal(df: pd.DataFrame) -> None:
    engineered = add_features(df)
    monthly = (engineered.groupby(engineered[config.TIMESTAMP].dt.month)[config.TARGET]
               .agg(["mean", "median",
                     lambda s: s.quantile(0.25),
                     lambda s: s.quantile(0.75)]))
    monthly.columns = ["mean", "median", "q25", "q75"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].fill_between(monthly.index, monthly["q25"], monthly["q75"],
                         color="#4C72B0", alpha=0.2, label="IQR")
    axes[0].plot(monthly.index, monthly["mean"], marker="o", color="#4C72B0",
                 linewidth=2, label="mean")
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("PM2.5 (µg/m³)")
    axes[0].set_title("Monthly profile")
    axes[0].legend()

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        subset = (engineered[engineered["season"] == season]
                  .groupby("hour")[config.TARGET].mean())
        axes[1].plot(subset.index, subset.values, marker="o", markersize=3,
                     linewidth=1.8, color=config.SEASON_COLOURS[season],
                     label=season)
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Mean PM2.5 (µg/m³)")
    axes[1].set_title("Diurnal profile by season")
    axes[1].legend()
    _save(fig, "03_seasonal_diurnal.png")


# --------------------------------------------------------------------------- #
# Weather relationships
# --------------------------------------------------------------------------- #
def plot_wind_effect(df: pd.DataFrame) -> None:
    engineered = add_features(df)
    wind_bins = pd.qcut(engineered["Iws"], 5, duplicates="drop")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    order = ["NW", "NE", "SE", "cv"]
    sns.boxplot(data=engineered, x="cbwd", y=config.TARGET, order=order, ax=axes[0],
                showfliers=False, hue="cbwd", legend=False,
                palette=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    axes[0].set_xlabel("Wind direction")
    axes[0].set_ylabel("PM2.5 (µg/m³)")
    axes[0].set_title("Northwesterly air is clean; calm air is not")

    means = engineered.groupby(wind_bins, observed=True)[config.TARGET].mean()
    centres = [interval.mid for interval in means.index]
    axes[1].plot(centres, means.values, marker="o", color="#C44E52", linewidth=2)
    axes[1].set_xlabel("Cumulative wind run Iws (quintile midpoint)")
    axes[1].set_ylabel("Mean PM2.5 (µg/m³)")
    axes[1].set_title("More wind, less pollution")
    _save(fig, "04_wind_effect.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    engineered = add_features(df)
    columns = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "dew_point_depression",
               "log_wind", "pressure_anomaly", "hour", "is_weekend", config.TARGET]
    correlation = engineered[columns].corr(numeric_only=True)
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    sns.heatmap(correlation, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, cbar_kws={"shrink": 0.75}, annot_kws={"size": 8}, ax=ax)
    ax.set_title("Correlation matrix of weather and calendar features")
    _save(fig, "05_correlation_heatmap.png")


# --------------------------------------------------------------------------- #
# Missingness and episodes
# --------------------------------------------------------------------------- #
def plot_missing_heatmap(df: pd.DataFrame) -> None:
    timestamps = pd.to_datetime(df[config.TIMESTAMP])
    table = (df.assign(missing=df[config.TARGET].isna(),
                       year=timestamps.dt.year, month=timestamps.dt.month)
             .pivot_table(index="year", columns="month", values="missing",
                          aggfunc="mean") * 100)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    sns.heatmap(table, annot=True, fmt=".1f", cmap="rocket_r", cbar_kws={
        "label": "% hours missing"}, ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title("Missing target hours cluster in multi-day runs")
    _save(fig, "06_missing_heatmap.png")


def plot_episodes(df: pd.DataFrame) -> None:
    timestamps = pd.to_datetime(df[config.TIMESTAMP])
    frame = df.assign(
        year=timestamps.dt.year,
        month=timestamps.dt.month,
        episode=df[config.TARGET] > config.EPISODE_THRESHOLD,
    )

    yearly = frame.groupby("year")["episode"].agg(["mean", "sum"])
    monthly = (frame.groupby(["year", "month"])["episode"].mean().reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].bar(yearly.index.astype(str), yearly["mean"] * 100, color="#C44E52")
    for index, value in enumerate(yearly["mean"] * 100):
        axes[0].text(index, value + 0.6, f"{value:.1f}%", ha="center", fontsize=9)
    axes[0].set_ylabel("Hours above 150 µg/m³ (%)")
    axes[0].set_title("Heavy-pollution burden by year")

    axes[1].plot(pd.to_datetime(monthly.assign(day=1)[["year", "month", "day"]]),
                 monthly["episode"] * 100, color="#4C72B0", linewidth=1.6)
    axes[1].set_ylabel("Hours above 150 µg/m³ (%)")
    axes[1].set_title("Monthly episode share")
    axes[1].tick_params(axis="x", rotation=30)
    _save(fig, "07_episodes.png")


def plot_weather_scatter(df: pd.DataFrame) -> None:
    engineered = add_features(df)
    sample = engineered.dropna(subset=[config.TARGET]).sample(
        4_000, random_state=config.RANDOM_STATE)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (column, label) in zip(
        axes, [("log_wind", "log1p(wind run)"), ("dew_point_depression",
                                                 "TEMP − DEWP (°C)"),
               ("PRES", "Pressure (hPa)")], strict=True
    ):
        ax.scatter(sample[column], sample[config.TARGET], s=5, alpha=0.25,
                   color="#4C72B0", edgecolor="none")
        ax.set_xlabel(label)
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_ylim(0, 500)
    fig.suptitle("Meteorology explains the broad strokes, not the spikes", fontsize=13)
    _save(fig, "08_weather_scatter.png")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame) -> dict:
    summary = compute_summary(df)

    plot_target_distribution(df)
    plot_series(df)
    plot_seasonal_diurnal(df)
    plot_wind_effect(df)
    plot_correlation_heatmap(df)
    plot_missing_heatmap(df)
    plot_episodes(df)
    plot_weather_scatter(df)

    path = config.REPORT_DIR / "eda_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[eda] summary -> {path.relative_to(config.ROOT)}")
    return summary
