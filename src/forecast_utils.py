"""
forecast_utils.py
------------------
Part 2: Shared definitions used by EVERY model in this project, so all
models are trained/tested/scored in exactly the same way (this is what
makes their results comparable to each other).

THE FORECASTING PROBLEM (plain English):
- Target variable : "Appliances" -> hourly energy used by appliances (in Wh)
- Forecast horizon : 24 hours ahead (predict a full day, one hour at a time)
- Train/test split : the LAST 14 days (336 hours) of the dataset are held
  back as a "test" set that no model is allowed to see during training.
  Everything before that is the "training" set.
- Primary evaluation: each model is trained on the training set, then asked
  to forecast the 24 hours immediately after training ends (which is also
  the first day of the test set). We compare the forecast to the real
  values for those 24 hours.
- We ALSO run a more thorough check later (Part 8) where we repeat this
  24-hour forecast 14 times, moving forward one day at a time through the
  whole test set (a "rolling" evaluation). This gives a more reliable
  picture than judging a model on a single day.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

TARGET = "Appliances"
TEST_DAYS = 14
HORIZON = 24  # hours


def load_hourly() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "energy_hourly.parquet")


def train_test_split(df: pd.DataFrame, test_days: int = TEST_DAYS):
    """
    Split into train / test by TIME (never randomly - this is a time series).
    The last `test_days` days become the test set; everything before is train.
    """
    n_test = test_days * 24
    train = df.iloc[:-n_test].copy()
    test = df.iloc[-n_test:].copy()
    return train, test


def regression_metrics(y_true, y_pred) -> dict:
    """
    RMSE  - average error size, in Wh, penalises big misses more (main metric)
    MAE   - average error size, in Wh, treats all misses equally (easy to read)
    MAPE  - average error as a %, easy to explain to non-technical people
            (can be noisy here because Appliances is sometimes small at night)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    # avoid divide-by-zero for MAPE
    nonzero = y_true != 0
    mape = float(np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE_%": mape}


def rolling_evaluate(train: pd.Series, test: pd.Series, forecast_fn, horizon: int = HORIZON) -> pd.DataFrame:
    """
    A fairer, more robust evaluation than judging a model on a single day.

    Plain English: instead of forecasting ONE 24-hour block and hoping it's
    representative, we slide through the whole 14-day test set one day at a
    time. At each of the 14 "origins", the model gets to see everything up to
    that point (training data + real test data already passed) and must
    forecast the next 24 hours. We then average the error across all 14 days.

    forecast_fn must have the signature: forecast_fn(history: pd.Series, horizon: int) -> np.ndarray
    """
    n_days = len(test) // horizon
    rows = []
    for day in range(n_days):
        start = day * horizon
        end = start + horizon
        history = pd.concat([train, test.iloc[:start]]) if start > 0 else train
        actual = test.iloc[start:end].values
        pred = forecast_fn(history, horizon)
        m = regression_metrics(actual, pred)
        m["origin_day"] = day + 1
        m["origin_date"] = str(test.index[start].date())
        rows.append(m)
    return pd.DataFrame(rows)[["origin_day", "origin_date", "RMSE", "MAE", "MAPE_%"]]


if __name__ == "__main__":
    df = load_hourly()
    train, test = train_test_split(df)
    print(f"Full dataset : {len(df)} hourly rows ({df.index.min()} -> {df.index.max()})")
    print(f"Train set    : {len(train)} rows ({train.index.min()} -> {train.index.max()})")
    print(f"Test set     : {len(test)} rows ({test.index.min()} -> {test.index.max()})")
    print(f"First 24h of test set (what we'll forecast first): "
          f"{test.index[0]} -> {test.index[23]}")
