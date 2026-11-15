"""Tests for cleaning, the missing-run audit and the imputation policy."""

from __future__ import annotations

import pandas as pd

from airquality import config
from airquality.clean import audit, clean, missing_runs, normalise


def test_normalise_builds_datetime_and_sorts(raw_frame: pd.DataFrame) -> None:
    shuffled = raw_frame.sample(frac=1.0, random_state=1)
    cleaned = normalise(shuffled)

    assert pd.api.types.is_datetime64_any_dtype(cleaned[config.TIMESTAMP])
    assert cleaned[config.TIMESTAMP].is_monotonic_increasing
    assert len(cleaned) == len(raw_frame)


def test_normalise_is_idempotent(clean_frame: pd.DataFrame) -> None:
    again = normalise(clean_frame)

    pd.testing.assert_frame_equal(clean_frame, again)


def test_normalise_clips_ranges(raw_frame: pd.DataFrame) -> None:
    frame = raw_frame.copy()
    frame.loc[0, "PRES"] = 1_500
    cleaned = normalise(frame)

    assert cleaned.loc[0, "PRES"] == config.VALID_RANGES["PRES"][1]


def test_missing_runs_detects_the_outage(clean_frame: pd.DataFrame) -> None:
    runs = missing_runs(clean_frame)

    assert runs["missing_hours"] == 26
    assert runs["runs"] == 3
    assert runs["longest_runs"][0] == 24
    assert runs["singletons"] == 2


def test_audit_reports_yearly_and_episode_stats(clean_frame: pd.DataFrame) -> None:
    log = audit(clean_frame)

    assert set(log["yearly"]) == {"2012"}
    assert log["target_missing"] == 26
    assert 0 < log["target_missing_share"] < 0.06
    assert log["episode_hours"] == int(
        (clean_frame[config.TARGET] > config.EPISODE_THRESHOLD).sum())
    assert "no imputation" in log["imputation_policy"]


def test_audit_counts_missing_months(clean_frame: pd.DataFrame) -> None:
    log = audit(clean_frame)

    assert log["worst_missing_months"]
    top = log["worst_missing_months"][0]
    assert top["share"] > 0
    assert 1 <= top["month"] <= 12


def test_clean_attaches_the_log(raw_frame: pd.DataFrame) -> None:
    cleaned, log = clean(raw_frame)

    assert cleaned.attrs["cleaning_log"] is log
    assert log["duplicate_timestamps"] == 0
    assert log["gaps_over_one_hour"] == 0
