"""Central configuration: paths, dataset location, time split and constants."""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
ARTIFACT_DIR = ROOT / "artifacts"
NOTEBOOK_DIR = ROOT / "notebooks"

for _directory in (RAW_DIR, PROCESSED_DIR, REPORT_DIR, FIGURE_DIR, ARTIFACT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
# UCI dataset 381 (Liang et al., 2015): hourly observations from the US Embassy
# air-quality monitor in Beijing, 1 January 2010 to 31 December 2014.
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/"
    "PRSA_data_2010.1.1-2014.12.31.csv"
)
RAW_FILENAME = "PRSA_data_2010.1.1-2014.12.31.csv"

COLUMNS = ["no", "year", "month", "day", "hour", "pm2.5", "DEWP", "TEMP",
           "PRES", "cbwd", "Iws", "Is", "Ir"]
TARGET = "pm2.5"
TIME_COLUMNS = ["year", "month", "day", "hour"]
TIMESTAMP = "timestamp"

# The target is the only column with missing values; it uses the literal token
# "NA" rather than an empty field.
MISSING_TOKEN = "NA"

# A time-ordered split is mandatory: a random split lets the model interpolate
# between neighbouring hours and reports a skill it cannot reproduce in
# production. Everything up to the end of 2013 trains; 2014 is held out.
TRAIN_END = "2013-12-31 23:00:00"
TEST_START = "2014-01-01 00:00:00"

# Chinese AQI "heavy pollution" band starts at 150 µg/m³; used for episode
# detection, where missing an event costs more than a false alarm.
EPISODE_THRESHOLD = 150

CBWD_LEVELS = ["NW", "NE", "SE", "cv"]

# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
CATEGORICAL_FEATURES = ["cbwd", "season"]

ENGINEERED_NUMERIC = [
    "dew_point_depression",
    "log_wind",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "pressure_anomaly",
]
ENGINEERED_CATEGORICAL = ["part_of_day"]

# Pollutant history. Every lag and rolling statistic is shifted by one hour so the
# value at time t never contains information from time t.
LAGS = [1, 3, 6, 12, 24, 48]
ROLLING_WINDOWS = [24, 72]
LAG_COLUMNS = (
    [f"pm_lag_{lag}" for lag in LAGS]
    + [f"pm_roll_{window}" for window in ROLLING_WINDOWS]
    + ["pm_diff_24"]
)

SCENARIOS = {
    "weather_only": "Meteorology and calendar only — works without a PM sensor",
    "with_lags": "Adds recent PM2.5 history — a short-term forecast setting",
}

VALID_RANGES = {
    "pm2.5": (0, 1_100),
    "DEWP": (-45, 35),
    "TEMP": (-25, 50),
    "PRES": (985, 1_055),
    "Iws": (0, 650),
    "Is": (0, 150),
    "Ir": (0, 150),
}

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
CV_FOLDS = 5
SEARCH_ITER = 12
SEARCH_FOLDS = 3

FIG_DPI = 140
SEASON_COLOURS = {
    "Winter": "#4C72B0",
    "Spring": "#55A868",
    "Summer": "#DD8452",
    "Autumn": "#C44E52",
}
