# Model Card — Beijing PM2.5 Hourly Forecast

## Overview
| Field | Value |
| --- | --- |
| Version | 0.1.0 |
| Task | Hourly PM2.5 regression + heavy-pollution episode detection |
| Algorithms | `hist_gradient_boosting` (weather-only), `ridge` (with lags) |
| Trained | 2026-11-15T13:35:51+00:00 |
| Data | UCI Beijing PM2.5, 43,824 hours, 2010–2014 |
| Split | Time-ordered: train ≤ 2013, test = 2014 |

## Intended use
Teaching and portfolio demonstration of time-series regression, lag construction and
honest baselining. **Not** a public-health warning system. Any operational use needs
multiple monitors, emissions data, uncertainty quantification and human review.

## Performance (2014 hold-out)
| Scenario | RMSE (µg/m³) |
| --- | --- |
| weather_only | 69.06 |
| with_lags | 21.24 |
| persistence baseline | 22.09 |

| Scenario | R² |
| --- | --- |
| weather_only | 0.455 |
| with_lags | 0.948 |

## Episode detection (>150 µg/m³)
| Scenario | Recall | Precision |
| --- | --- | --- |
| weather_only | 47.3% | 61.8% |
| with_lags | 90.1% | 90.6% |

## Limitations
- Single monitor in one city; no emissions or boundary-layer inputs.
- Peak concentrations are systematically under-predicted by squared-error models.
- 4.72% of target hours are missing, clustered in winter.
- Persistence is a strong baseline; the reported skill is measured against it.
- One test year; no uncertainty intervals are produced.

## Ethical considerations
- A missed heavy-pollution hour has health consequences; a false alarm erodes trust.
  Neither error is acceptable without a cost analysis the data cannot supply.
- Predictions must not be used to downplay official air-quality warnings.
