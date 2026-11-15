# air-quality-pipeline

End-to-end pipeline for the UCI Beijing PM2.5 dataset (43,824 hourly records,
2010-2014). Regression of hourly PM2.5 concentration plus episode detection
(PM2.5 > 150 ug/m3), with time-ordered evaluation.

## Headline results

| scenario | model | test RMSE | test R2 | episode recall |
|---|---|---|---|---|
| weather only | hist_gradient_boosting | 69.06 | 0.455 | 0.473 |
| with lags | ridge | 21.24 | 0.948 | 0.901 |

- Persistence baseline RMSE = 22.09; climatology RMSE = 93.53.
- Episode detection (with lags): precision 0.906, recall 0.901, F1 0.904.

## Dataset

- 43,824 hourly rows, 2010-2014, 5 meteorological variables + PM2.5.
- 2,067 missing targets (4.72%) in 214 runs; longest run 132 h.
- Yearly means: 104.05, 99.07, 90.55, 101.71, 97.73 ug/m3.
- Episodes above 150 ug/m3: 8,991 hours (21.5%).

## Split and validation

- Time-ordered: train = 2010-2013 (33,096 rows), test = 2014 (8,661 rows).
- No random split. TimeSeriesSplit (5 folds) for CV.
- Lag features use shift(1) so no target leakage.

## Reproduce

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python run.py all          # download -> EDA -> train
    python -m pytest          # 33 tests
    python -m ruff check src tests

## Layout

    src/airquality/   config, data, clean, features, eda, model, report, cli
    tests/            33 tests
    reports/          figures (8), eda_summary.json, metrics.json, REPORT.md, model_card.md

License: MIT.
