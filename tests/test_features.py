"""Tests for calendar features, weather features and leakage-safe lags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airquality import config
from airquality.features import (
    add_features,
    add_lag_features,
    add_time_features,
    add_weather_features,
    build_feature_matrix,
    episode_target,
    feature_columns,
)


def test_time_features_are_cyclic(clean_frame: pd.DataFrame) -> None:
    engineered = add_time_features(clean_frame)

    assert set(engineered["season"]) <= {"Winter", "Spring", "Summer", "Autumn"}
    assert set(engineered["part_of_day"]) <= {"night", "morning", "afternoon", "evening"}
    assert engineered["hour_sin"].between(-1, 1).all()
    assert engineered["hour_cos"].between(-1, 1).all()
    assert engineered["is_weekend"].isin([0, 1]).all()
    # 1 January 2012 was a Sunday.
    assert engineered.loc[0, "is_weekend"] == 1


def test_weather_features_are_interpretable(clean_frame: pd.DataFrame) -> None:
    engineered = add_weather_features(clean_frame)
    expected = engineered["TEMP"] - engineered["DEWP"]

    pd.testing.assert_series_equal(engineered["dew_point_depression"], expected,
                                   check_names=False)
    assert (engineered["log_wind"] >= 0).all()
    assert np.allclose(engineered["log_wind"],
                       np.log1p(engineered["Iws"].clip(lower=0)))


def test_lag_one_is_the_previous_hour(clean_frame: pd.DataFrame) -> None:
    engineered = add_lag_features(clean_frame)
    expected = clean_frame[config.TARGET].shift(1)

    pd.testing.assert_series_equal(engineered["pm_lag_1"], expected,
                                   check_names=False)
    # One warm-up hour plus one hour after every missing target block.
    assert engineered["pm_lag_1"].isna().sum() == (
        clean_frame[config.TARGET].isna().sum() + 1
    )


def test_rolling_mean_never_sees_the_current_hour(clean_frame: pd.DataFrame) -> None:
    engineered = add_lag_features(clean_frame)
    series = clean_frame[config.TARGET]

    for position in (200, 500, 1_000):
        expected = series.iloc[position - 24:position].mean()
        assert engineered["pm_roll_24"].iloc[position] == pytest.approx(
            expected, rel=1e-9, abs=1e-9)


def test_lag_difference_uses_two_past_values(clean_frame: pd.DataFrame) -> None:
    engineered = add_lag_features(clean_frame)

    expected = engineered["pm_lag_1"] - engineered["pm_lag_24"]
    pd.testing.assert_series_equal(engineered["pm_diff_24"], expected,
                                   check_names=False)


def test_feature_columns_add_lags_only_when_requested() -> None:
    base_numeric, base_categorical = feature_columns()
    lagged_numeric, lagged_categorical = feature_columns(include_lags=True)

    assert set(config.LAG_COLUMNS) <= set(lagged_numeric)
    assert not set(config.LAG_COLUMNS) & set(base_numeric)
    assert base_categorical == lagged_categorical
    assert "cbwd" in base_categorical


def test_feature_matrix_drops_missing_targets(clean_frame: pd.DataFrame) -> None:
    features, target = build_feature_matrix(clean_frame, include_lags=False)

    assert target.notna().all()
    assert len(features) == len(clean_frame) - 26
    assert not features.isna().all(axis=1).any()


def test_lagged_matrix_drops_the_warmup_rows(clean_frame: pd.DataFrame) -> None:
    base_features, base_target = build_feature_matrix(clean_frame, include_lags=False)
    lagged_features, lagged_target = build_feature_matrix(clean_frame, include_lags=True)

    assert len(lagged_features) < len(base_features)
    assert lagged_target.index.equals(lagged_features.index)
    assert lagged_features[config.LAG_COLUMNS].notna().all().all()
    assert lagged_features.isna().sum().sum() == 0


def test_episode_target_uses_the_threshold(clean_frame: pd.DataFrame) -> None:
    flags = episode_target(clean_frame[config.TARGET])

    assert flags.dtype == int
    assert flags.sum() == int((clean_frame[config.TARGET] >
                              config.EPISODE_THRESHOLD).sum())
    assert set(flags.unique()) <= {0, 1}


def test_add_features_combines_both_steps(clean_frame: pd.DataFrame) -> None:
    engineered = add_features(clean_frame)

    assert "season" in engineered.columns
    assert "dew_point_depression" in engineered.columns
    assert "pm_lag_1" not in engineered.columns
