"""Tests for schema validation and loading."""

from __future__ import annotations

import pandas as pd
import pytest

from airquality import config, data


def test_validate_reports_continuity_and_missingness(raw_frame: pd.DataFrame) -> None:
    report = data.validate(raw_frame)

    assert report["rows"] == len(raw_frame)
    assert report["duplicate_timestamps"] == 0
    assert report["missing_target"] == 26
    assert report["gaps"] == 0
    assert report["episode_hours"] >= 0


def test_validate_raises_on_duplicate_timestamps(raw_frame: pd.DataFrame) -> None:
    frame = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)
    frame = data.add_timestamp(frame)
    with pytest.raises(data.DataValidationError, match="timestamp_unique"):
        data.validate(frame.sort_values(config.TIMESTAMP))


def test_validate_raises_on_missing_weather_column(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(data.DataValidationError, match="expected_columns"):
        data.validate(raw_frame.drop(columns=["PRES"]))


def test_validate_raises_when_weather_is_missing(raw_frame: pd.DataFrame) -> None:
    frame = raw_frame.copy()
    frame.loc[5, "TEMP"] = None
    with pytest.raises(data.DataValidationError, match="missingness_confined"):
        data.validate(frame)


def test_validate_raises_on_unknown_wind_direction(raw_frame: pd.DataFrame) -> None:
    frame = raw_frame.copy()
    frame.loc[5, "cbwd"] = "SW"
    with pytest.raises(data.DataValidationError, match="wind_direction_domain"):
        data.validate(frame)


def test_validate_raises_when_target_range_is_broken(raw_frame: pd.DataFrame) -> None:
    frame = raw_frame.copy()
    frame.loc[5, "pm2.5"] = 5_000
    with pytest.raises(data.DataValidationError, match="target_range"):
        data.validate(frame)


def test_load_raw_parses_na_token_and_builds_timestamp(tmp_path) -> None:
    path = tmp_path / "pm25.csv"
    rows = [
        "No,year,month,day,hour,pm2.5,DEWP,TEMP,PRES,cbwd,Iws,Is,Ir",
        "1,2010,1,1,0,NA,-21,-11,1021,NW,1.79,0,0",
        "2,2010,1,1,1,120,-20,-10,1020,SE,2.50,0,0",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    frame = data.load_raw(path)

    assert len(frame) == 2
    assert frame.loc[0, config.TARGET] != frame.loc[0, config.TARGET]  # NaN
    assert frame.loc[1, config.TARGET] == 120
    assert frame.loc[0, config.TIMESTAMP] == pd.Timestamp("2010-01-01 00:00:00")
    assert frame.loc[1, "cbwd"] == "SE"
