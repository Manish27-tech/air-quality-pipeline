"""Dataset acquisition, loading and validation."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

from . import config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #
def download(force: bool = False, verbose: bool = True) -> Path:
    destination = config.RAW_DIR / config.RAW_FILENAME
    if destination.exists() and not force:
        if verbose:
            print(f"[data] cached: {destination.name} "
                  f"({destination.stat().st_size / 1e6:.1f} MB)")
        return destination

    if verbose:
        print(f"[data] downloading {config.DATASET_URL}")
    request = urllib.request.Request(
        config.DATASET_URL, headers={"User-Agent": "air-quality-pipeline/0.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except Exception as exc:  # noqa: BLE001 - surfaced as one clear error
        raise RuntimeError(f"Could not fetch the dataset: {exc}") from exc

    if len(payload) < 1_000_000:
        raise RuntimeError(f"response looks truncated ({len(payload)} bytes)")
    destination.write_bytes(payload)
    if verbose:
        print(f"[data] saved {destination.name} "
              f"(sha256={_sha256(destination)[:12]}...)")
    return destination


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the CSV, parse the calendar columns and build a timestamp.

    Only the literal ``NA`` token is treated as missing — ``keep_default_na`` is
    disabled so that the wind-direction code ``cv`` and numeric strings are never
    reinterpreted.
    """
    path = path or config.RAW_DIR / config.RAW_FILENAME
    if not path.exists():
        download(verbose=False)

    frame = pd.read_csv(
        path,
        na_values=[config.MISSING_TOKEN],
        keep_default_na=False,
    )
    frame.columns = [column.strip() for column in frame.columns]
    frame = frame.rename(columns={"No": "no"})
    frame = add_timestamp(frame)
    return frame


def add_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a parsed ``timestamp`` column (idempotent)."""
    out = frame.copy()
    if config.TIMESTAMP in out.columns:
        out[config.TIMESTAMP] = pd.to_datetime(out[config.TIMESTAMP])
        return out
    out[config.TIMESTAMP] = pd.to_datetime(out[config.TIME_COLUMNS].astype(int))
    return out


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class DataValidationError(ValueError):
    """Raised when the raw dataset does not satisfy the expected schema."""


def validate(df: pd.DataFrame) -> dict:
    """Run schema, continuity and domain checks, raising on critical failures."""
    checks: list[dict] = []
    critical: list[str] = []

    def record(name: str, passed: bool, detail: str, is_critical: bool = False) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail,
                       "critical": is_critical})
        if not passed and is_critical:
            critical.append(name)

    missing = [column for column in config.COLUMNS if column not in df.columns]
    record("expected_columns", not missing,
           "all 13 columns present" if not missing else f"missing: {missing}",
           is_critical=True)
    if critical:
        raise DataValidationError("failed critical validation: " + ", ".join(critical))

    record("non_empty", len(df) > 0, f"{len(df):,} rows", is_critical=True)

    expected_hours = 5 * 365 * 24 + 24  # 2010-2014 inclusive, 2012 is a leap year
    record("expected_scale", abs(len(df) - expected_hours) <= 24,
           f"{len(df):,} rows (expected {expected_hours:,} hourly records)")

    df = add_timestamp(df)
    timestamps = pd.to_datetime(df[config.TIMESTAMP])
    record("timestamp_unique", timestamps.is_unique,
           f"{int(timestamps.duplicated().sum())} duplicate timestamps",
           is_critical=True)
    record("timestamp_monotonic", timestamps.is_monotonic_increasing,
           "records are ordered in time")

    deltas = timestamps.sort_values().diff().dropna()
    gaps = int((deltas > pd.Timedelta(hours=1)).sum())
    record("hourly_continuity", gaps == 0,
           f"{gaps} gaps larger than one hour in the series")

    numeric_target = pd.to_numeric(df[config.TARGET], errors="coerce")
    low, high = config.VALID_RANGES[config.TARGET]
    record("target_range", bool(numeric_target.dropna().between(low, high).all()),
           f"{config.TARGET} within {low}-{high}", is_critical=True)

    nulls = {column: int(count) for column, count in df.isna().sum().items() if count}
    unexpected = {column: count for column, count in nulls.items()
                  if column != config.TARGET}
    record("missingness_confined_to_target", not unexpected,
           f"missing values: {nulls or 'none'}", is_critical=True)

    missing_share = float(numeric_target.isna().mean())
    record("missing_share_bounded", missing_share < 0.06,
           f"{missing_share:.2%} of target values missing "
           f"({int(numeric_target.isna().sum()):,} hours)", is_critical=True)

    bad_direction = sorted(set(df["cbwd"].unique()) - set(config.CBWD_LEVELS))
    record("wind_direction_domain", not bad_direction,
           "NW/NE/SE/cv only" if not bad_direction else f"unexpected: {bad_direction}",
           is_critical=True)

    record("target_starts_2010", timestamps.min() == pd.Timestamp("2010-01-01 00:00:00"),
           f"first record: {timestamps.min()}")
    record("target_ends_2014", timestamps.max() == pd.Timestamp("2014-12-31 23:00:00"),
           f"last record: {timestamps.max()}")

    if critical:
        raise DataValidationError("failed critical validation: " + ", ".join(critical))

    report = {
        "rows": int(len(df)),
        "columns": len(config.COLUMNS),
        "start": str(timestamps.min()),
        "end": str(timestamps.max()),
        "missing_target": int(numeric_target.isna().sum()),
        "missing_share": round(missing_share, 4),
        "mean_pm25": round(float(numeric_target.mean()), 2),
        "median_pm25": float(numeric_target.median()),
        "max_pm25": float(numeric_target.max()),
        "episode_hours": int((numeric_target > config.EPISODE_THRESHOLD).sum()),
        "episode_share": round(float((numeric_target > config.EPISODE_THRESHOLD).mean()), 4),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "gaps": gaps,
        "checks": checks,
    }
    df.attrs["validation"] = report
    return report
