"""Feature engineering for hourly PM2.5 forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import add_timestamp

SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
PART_OF_DAY = {
    0: "night", 1: "night", 2: "night", 3: "night", 4: "night", 5: "night",
    6: "morning", 7: "morning", 8: "morning", 9: "morning", 10: "morning",
    11: "morning",
    12: "afternoon", 13: "afternoon", 14: "afternoon", 15: "afternoon",
    16: "afternoon", 17: "afternoon",
    18: "evening", 19: "evening", 20: "evening", 21: "evening", 22: "evening",
    23: "evening",
}


# --------------------------------------------------------------------------- #
# Calendar features
# --------------------------------------------------------------------------- #
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar features from the hourly timestamp.

    Highlighting hourly sin/cos keeps the cyclic structure (23:00 is next to 00:00)
    instead of making the model learn an artificial jump.
    """
    out = add_timestamp(df)
    timestamps = pd.to_datetime(out[config.TIMESTAMP])

    out["hour"] = timestamps.dt.hour
    out["dayofweek"] = timestamps.dt.dayofweek
    out["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype(int)
    out["dayofyear"] = timestamps.dt.dayofyear
    out["season"] = timestamps.dt.month.map(SEASON_BY_MONTH)
    out["part_of_day"] = out["hour"].map(PART_OF_DAY)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out


# --------------------------------------------------------------------------- #
# Weather features
# --------------------------------------------------------------------------- #
def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw meteorological columns into interpretable quantities.

    ``TEMP - DEWP`` is the dew-point depression: near zero means humid air and a
    high chance of haze. ``Iws`` is a *cumulative* wind run, so the raw value grows
    with the averaging window and is heavily right-skewed; the log stabilises it.
    """
    out = df.copy()

    out["dew_point_depression"] = out["TEMP"] - out["DEWP"]
    out["log_wind"] = np.log1p(out["Iws"].clip(lower=0))
    out["pressure_anomaly"] = out["PRES"] - 1016.45

    # Wind direction stays categorical and is one-hot encoded inside the pipeline,
    # so no dummy block is created here.
    return out
