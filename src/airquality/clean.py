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
