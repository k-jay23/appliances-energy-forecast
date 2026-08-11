# Household Appliance Energy Forecasting

A case study forecasting hourly household appliance energy use 24 hours ahead, comparing
naive benchmarks, SARIMAX, a feature-based gradient-boosted tree model (XGBoost), and a
pretrained time-series foundation model (Chronos).

Full write-up, methodology, and discussion: [`report/appliances_energy_forecast_report.docx`](report/appliances_energy_forecast_report.docx)

## Dataset

[UCI Appliances Energy Prediction dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
(Candanedo, Feldheim & Deramaix, 2017) — 10-minute readings of appliance energy use, plus
indoor temperature/humidity (9 rooms) and outdoor weather, from a low-energy house in
Belgium, over 4.5 months (11 Jan – 27 May 2016). Resampled to hourly resolution for this
project (3,290 rows).

## Results summary

24-hour-ahead forecast, ranked by the 14-day rolling-average RMSE (lower is better):

| Model | RMSE (single 24h) | RMSE (14-day rolling) | MAE (rolling) | MAPE % (rolling) |
|---|---|---|---|---|
| **SARIMAX(1,0,6)(1,1,1,24)** | **159.6** | **328.1** | **214.3** | 33.2 |
| XGBoost (true forecast) | 238.9 | 373.0 | 240.9 | 38.5 |
| XGBoost (conditional*) | 231.3 | 379.3 | 284.5 | 53.8 |
| Mean | 300.0 | 391.2 | 296.1 | 53.6 |
| Weekly Seasonal Naive | 292.8 | 404.9 | 254.6 | 37.1 |
| Chronos (foundation model) | 305.2 | n/a† | n/a† | n/a† |
| Daily Seasonal Naive | 708.3 | 455.8 | 288.6 | 43.8 |
| Naive | 1485.6 | 584.7 | 512.0 | 113.4 |
| Drift | 1492.5 | 586.1 | 513.6 | 113.8 |

\* uses actual future weather/indoor sensor readings — a conditional, not a true, forecast (see report §10, Q5)
† Chronos was evaluated on the single 24h test only; see [Limitations](#limitations-known-gaps)

**Headline finding**: SARIMAX gives the most accurate forecasts and beats every other model
on both evaluation regimes, despite XGBoost and Chronos being considerably more complex.
Full discussion of *why* is in the report.

## Repository structure

```
├── src/                          # all analysis code, run in this order
│   ├── data_prep.py              # Part 1 — load, clean, resample to hourly
│   ├── eda.py                    # Part 1 — exploratory plots, STL decomposition
│   ├── stationarity.py           # Part 1 — ADF test, ACF/PACF, differencing
│   ├── forecast_utils.py         # Part 2 — train/test split, metrics, rolling evaluation
│   ├── benchmarks.py             # Part 3 — mean/naive/seasonal-naive/drift benchmarks
│   ├── sarimax_grid_search.py    # Part 4 — 147-combination SARIMAX grid search (resumable)
│   ├── sarimax_final.py          # Part 4 — refit best model, diagnostics, forecast
│   ├── sarimax_rolling.py        # Part 4 — 14-day rolling robustness check (resumable)
│   ├── features.py               # Part 5 — feature engineering (two variants)
│   ├── ml_model.py               # Part 6 — XGBoost training and evaluation
│   ├── foundation_model_chronos.py  # Part 7 — Chronos (run in Colab, see note below)
│   └── evaluate_all.py           # Part 8 — consolidated comparison table + plot
├── data/
│   ├── raw/                      # original energydata_complete.csv
│   └── processed/                # cleaned/resampled parquet files
├── outputs/
│   ├── figures/                  # all generated plots (PNG)
│   └── metrics/                  # all generated results (CSV)
├── report/
│   ├── appliances_energy_forecast_report.docx   # the full 8-page report
│   ├── part9_answers.md          # Part 9 — 6 discussion questions (source text)
│   └── build_report.js           # script that generates the .docx report
└── requirements.txt
```

## How to reproduce

```bash
pip install -r requirements.txt

cd src
python data_prep.py            # Part 1
python eda.py
python stationarity.py
python benchmarks.py           # Part 3 (also runs Part 2's split)
python sarimax_grid_search.py  # Part 4 — resumable; re-run until it prints "ALL DONE"
python sarimax_final.py
python sarimax_rolling.py      # resumable; re-run until it prints "ALL DONE"
python ml_model.py             # Parts 5 & 6
python evaluate_all.py         # Part 8
```

**Note on `sarimax_grid_search.py` and `sarimax_rolling.py`**: these are deliberately
resumable — each run picks up where the last one stopped (tracked via the output CSV) — because
the full grid search / rolling evaluation is compute-heavy (147 SARIMAX fits, and 14 more
refits respectively). Just re-run the script until you see `ALL DONE`.

**Note on `foundation_model_chronos.py`**: this must be run in an environment with access to
HuggingFace (e.g. Google Colab), since it downloads pretrained model weights. See the comment
block at the top of the file for exact copy-paste instructions. The results already obtained
this way are saved in `outputs/metrics/part7_chronos_metrics.csv`.

## Requirements

Python 3.12, plus: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `scikit-learn`, `xgboost`,
`lightgbm`, `pyarrow`. See `requirements.txt` for exact versions used.

For Part 7 only (run separately in Colab): `chronos-forecasting`, `torch`.

## Limitations (known gaps)

- The SARIMAX grid search used a reduced 45-day window and a capped iteration limit for
  compute reasons; 90 of 147 combinations did not fully converge (the selected winner did).
- No exogenous-variable SARIMAX variant was tested.
- The XGBoost model is a single "direct" model covering all 24 forecast horizons at once,
  rather than one specialised model per horizon.
- Chronos was evaluated only on a single 24-hour window (not the 14-day rolling check used
  for the other models), and only the "small" model size was tried.
- Analysis covers one house over one 4.5-month period; findings may not generalise to other
  homes, climates, or seasons.

Full discussion of these points, and proposed future work, is in the report (§11).

## References

- Candanedo, L. M., Feldheim, V. & Deramaix, D. (2017). Data driven prediction models of
  energy use of appliances in a low-energy house. *Energy and Buildings*, 140, 81–97.
- Ansari, A. F. et al. (2024). Chronos: Learning the Language of Time Series. arXiv:2403.07815.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung, G. M. (2015). *Time Series Analysis:
  Forecasting and Control* (5th ed.). Wiley.
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
