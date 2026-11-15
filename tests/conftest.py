"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airquality import config


@pytest.fixture()
def raw_frame() -> pd.DataFrame:
    """Sixty days of plausible hourly weather with a few missing target hours."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    timestamps = pd.date_range("2012-01-01", periods=60 * 24, freq="h")
    hours = np.arange(len(timestamps))
    seasonal = np.sin(2 * np.pi * hours / (365 * 24))
    diurnal = np.sin(2 * np.pi * (hours % 24) / 24)

    pm = 60 + 35 * seasonal + 12 * diurnal + rng.normal(0, 15, len(timestamps))
    pm = np.clip(pm, 0, 500)

    frame = pd.DataFrame({
        "no": np.arange(1, len(timestamps) + 1),
        "year": timestamps.year,
        "month": timestamps.month,
        "day": timestamps.day,
        "hour": timestamps.hour,
        "pm2.5": pm,
        "DEWP": np.round(-5 + 12 * seasonal + rng.normal(0, 2, len(timestamps)), 1),
        "TEMP": np.round(5 + 15 * seasonal + 6 * diurnal
                         + rng.normal(0, 1.5, len(timestamps)), 1),
        "PRES": np.round(1016 + 6 * rng.normal(0, 1, len(timestamps)), 1),
        "cbwd": np.resize(["NW", "NE", "SE", "cv"], len(timestamps)),
        "Iws": np.round(np.abs(rng.lognormal(1.2, 1.0, len(timestamps))), 2),
        "Is": np.where(rng.random(len(timestamps)) < 0.5, 1.0, 0.0),
        "Ir": np.where(rng.random(len(timestamps)) < 0.9, 0.0, 1.0),
    })

    # One 24-hour outage plus two isolated dropouts: 1.8% missing overall.
    frame.loc[100:123, "pm2.5"] = np.nan
    frame.loc[500, "pm2.5"] = np.nan
    frame.loc[900, "pm2.5"] = np.nan
    return frame


@pytest.fixture()
def clean_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    from airquality.clean import clean

    cleaned, _ = clean(raw_frame)
    return cleaned
