"""
Shared stuff used across every model script - train/test split, metrics,
and the rolling evaluation function. Put this in one place so every model
gets scored the exact same way, otherwise the comparison at the end
wouldn't really be fair.

Quick recap of the setup: forecasting "Appliances" (Wh) 24h ahead, last 14
days held out as test, rest is training. I originally just scored everyone
on a single 24h forecast but that felt like it could get lucky/unlucky
depending which day it landed on (it did - see benchmarks), so added the
rolling check below too as a sanity check.
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
    # NOTE: has to be a time-based split, not random - shuffling a time
    # series before splitting would leak future info into training
    n_test = test_days * 24
    train = df.iloc[:-n_test].copy()
    test = df.iloc[-n_test:].copy()
    return train, test


def regression_metrics(y_true, y_pred) -> dict:
    # RMSE is the main one I'm using to rank models (punishes big misses
    # more), MAE for an easier-to-read number, MAPE as a % just because
    # it's what non-technical readers usually expect to see
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
    # walks through all 14 test days one at a time, refitting/re-forecasting
    # at each "origin" with an expanding history, then averages the error.
    # more work than just scoring one day but way less noisy.
    # forecast_fn signature: forecast_fn(history: pd.Series, horizon: int) -> np.ndarray
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
