"""Cleaning: type coercion, range guards and an auditable summary."""

from __future__ import annotations

import pandas as pd

from . import config
from .data import add_timestamp


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, rebuild the timestamp and clamp values to plausible ranges.

    Idempotent: a cleaned frame passes through unchanged.
    """
    out = add_timestamp(df)

    for column in ["no", *config.TIME_COLUMNS]:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")

    out[config.TARGET] = pd.to_numeric(out[config.TARGET], errors="coerce")

    for column in ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["cbwd"] = out["cbwd"].astype("string").str.strip()

    for column, (low, high) in config.VALID_RANGES.items():
        out[column] = out[column].clip(lower=low, upper=high)

    return out.sort_values(config.TIMESTAMP).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def missing_runs(df: pd.DataFrame) -> dict:
    """Describe how the 2,067 missing target hours cluster in time.

    Long runs matter: they are not random sensor dropouts but multi-day gaps that
    any lag-based model must skip rather than interpolate through.
    """
    missing = df[config.TARGET].isna().to_numpy()
    runs: list[int] = []
    current = 0
    for flag in missing:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)

    runs.sort(reverse=True)
    return {
        "missing_hours": int(missing.sum()),
        "missing_share": round(float(missing.mean()), 4),
        "runs": len(runs),
        "longest_runs": runs[:5],
        "singletons": int(sum(1 for run in runs if run == 1)),
    }


def audit(df: pd.DataFrame) -> dict:
    timestamps = pd.to_datetime(df[config.TIMESTAMP])
    deltas = timestamps.diff().dropna()
    yearly = (
        df.groupby(df[config.TIMESTAMP].dt.year)[config.TARGET]
        .agg(["mean", "median", "count"]).round(2)
    )
    monthly_missing = (
        df.assign(missing=df[config.TARGET].isna())
        .groupby([df[config.TIMESTAMP].dt.year, df[config.TIMESTAMP].dt.month])["missing"]
        .mean().round(4)
    )

    return {
        "rows_in": int(len(df)),
        "rows_out": int(len(df)),
        "target_missing": int(df[config.TARGET].isna().sum()),
        "target_missing_share": round(float(df[config.TARGET].isna().mean()), 4),
        "missing_runs": missing_runs(df),
        "gaps_over_one_hour": int((deltas > pd.Timedelta(hours=1)).sum()),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "episode_hours": int((df[config.TARGET] > config.EPISODE_THRESHOLD).sum()),
        "episode_share": round(
            float((df[config.TARGET] > config.EPISODE_THRESHOLD).mean()), 4),
        "yearly": {
            str(int(year)): {"mean": float(row["mean"]),
                             "median": float(row["median"]),
                             "hours": int(row["count"])}
            for year, row in yearly.iterrows()
        },
        "worst_missing_months": [
            {"year": int(year), "month": int(month), "share": float(share)}
            for (year, month), share in monthly_missing.nlargest(5).items()
        ],
        "imputation_policy": (
            "no imputation: a missing target cannot be trained on, and "
            "interpolating it would fake the very spikes the model is meant to catch"
        ),
    }


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cleaned = normalise(df)
    log = audit(cleaned)
    cleaned.attrs["cleaning_log"] = log
    return cleaned, log


def save_processed(df: pd.DataFrame) -> str:
    destination = config.PROCESSED_DIR / "beijing_pm25_clean.csv"
    df.to_csv(destination, index=False)
    return str(destination)
