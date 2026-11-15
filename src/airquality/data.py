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
