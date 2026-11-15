# Data provenance

The raw file is **not** committed. `python run.py download` fetches it from UCI and caches
it in `data/raw/` (git-ignored). This file records exactly what was used.

## Source

- Dataset: **PM2.5 Data of Five Chinese Cities**, UCI Machine Learning Repository, ID 381.
- Original paper: X. Liang et al. (2015), *Assessing Beijing's PM2.5 pollution: severity,
  weather, APEC and winter heating effects*, Proceedings of the Royal Society A.
- The Beijing file contains hourly observations from the US Embassy monitor.
- URL: <https://archive.ics.uci.edu/ml/machine-learning-databases/00381/>

## File

| File | Rows | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `PRSA_data_2010.1.1-2014.12.31.csv` | 43,824 | 2,010,494 | `4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c` |

Period: **2010-01-01 00:00 → 2014-12-31 23:00** (five calendar years, including the
2012 leap day). Exactly 43,824 hourly rows, zero duplicate timestamps, zero missing
hours.

## Quirks the loader handles

1. **The target is the only incomplete column.** 2,067 of 43,824 `pm2.5` values are
   missing (4.72%), written as the literal token `NA`, and they cluster in multi-day
   runs (longest: 132 hours). The loader parses only that token as missing and disables
   pandas' default NA list so `cv` and numeric strings are never reinterpreted.
2. **No timestamp column.** The date is split into `year`, `month`, `day`, `hour`;
   `timestamp` is reconstructed and validated for uniqueness and hourly continuity.
3. **`cbwd` is categorical.** Values are `NW`, `NE`, `SE` and `cv` (calm/variable).
   Treating it as ordinal would impose a false ordering.
4. **`Iws` is a cumulative wind run**, not a wind speed: its scale depends on the
   averaging window, it is heavily right-skewed (max 585.6), so it is log-transformed.
5. **Unit changes.** `Is`/`Ir` are flagged as hours of snow/rain; they are kept as
   0/1-style indicators, not as continuous measurements.
6. **Missing targets are not imputed.** They are dropped at the feature-matrix stage,
   because interpolation would fabricate the pollution spikes the model is meant to
   detect.

## Licence / terms

The dataset is distributed by UCI for research use; see the repository page for the
current terms. The code in this repository is MIT licensed.
