"""
features.py
------------
Part 5: Build features for the feature-based ML model (Part 6).

APPROACH: "direct multi-horizon" tabular forecasting.
Instead of one row per hour, we build one row per (origin, hours_ahead) pair,
covering hours_ahead = 1..24. Each row represents: "standing at time
`origin`, what is appliance use `hours_ahead` hours later?" This lets a
single regression model learn to forecast any point in the next 24 hours.

FEATURES (all designed to be knowable at the forecast origin - see notes):
- hours_ahead            : how far into the future this row forecasts (1-24)
- target_hour_sin/cos    : hour-of-day of the TARGET time, cyclically encoded
- target_dow_sin/cos     : day-of-week of the TARGET time, cyclically encoded
- is_weekend             : whether the TARGET time falls on Sat/Sun
- recent_lag_1           : last actually-observed Appliances value AT THE ORIGIN
- recent_lag_24          : Appliances value 24h before the origin
- recent_lag_168         : Appliances value 168h (1 week) before the origin
- rolling_mean_24        : mean of the last 24h of Appliances, ending at origin
- rolling_std_24         : std of the last 24h of Appliances, ending at origin
- rolling_mean_168       : mean of the last 168h (1 week), ending at origin
- target_lag24           : Appliances value exactly 24h before the TARGET time
                            (always knowable: target - 24h <= origin whenever
                            hours_ahead <= 24, i.e. always in this dataset)
- target_lag168          : Appliances value exactly 168h before the TARGET time
                            (always knowable for the same reason)

TWO FEATURE SETS ARE BUILT (this matters for Part 9, Question 5):
- "true_forecast" set  : only the features above (all genuinely knowable
                          in advance without needing to already know the
                          future weather).
- "conditional" set    : the true_forecast features PLUS the actual indoor
                          (T1-T9, RH_1-RH_9) and outdoor (T_out, RH_out,
                          Windspeed, Visibility, Press_mm_hg, Tdewpoint)
                          sensor readings AT THE TARGET TIME. These are
                          real historical readings, so they make the model
                          look artificially good - in a real deployment you
                          would not know tomorrow's indoor temperature
                          exactly, only a weather FORECAST for outdoor
                          variables. This is intentionally included so we
                          can measure and discuss the gap it creates.
"""

import numpy as np
import pandas as pd
from pathlib import Path

WEATHER_COLS = (
    ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
    + [f"T{i}" for i in range(1, 10)]
    + [f"RH_{i}" for i in range(1, 10)]
)

MIN_HISTORY_HOURS = 168  # need a full week of history for lag_168


def _row_features(df: pd.DataFrame, y: pd.Series, origin_idx: int, h: int,
                   use_future_weather: bool) -> dict:
    target_idx = origin_idx + h
    target_time = df.index[target_idx]

    row = {
        "origin_time": df.index[origin_idx],
        "target_time": target_time,
        "hours_ahead": h,
        "target_hour_sin": np.sin(2 * np.pi * target_time.hour / 24),
        "target_hour_cos": np.cos(2 * np.pi * target_time.hour / 24),
        "target_dow_sin": np.sin(2 * np.pi * target_time.dayofweek / 7),
        "target_dow_cos": np.cos(2 * np.pi * target_time.dayofweek / 7),
        "is_weekend": int(target_time.dayofweek >= 5),
        "recent_lag_1": y.iloc[origin_idx],
        "recent_lag_24": y.iloc[origin_idx - 23],
        "recent_lag_168": y.iloc[origin_idx - 167],
        "rolling_mean_24": y.iloc[origin_idx - 23: origin_idx + 1].mean(),
        "rolling_std_24": y.iloc[origin_idx - 23: origin_idx + 1].std(),
        "rolling_mean_168": y.iloc[origin_idx - 167: origin_idx + 1].mean(),
        "target_lag24": y.iloc[target_idx - 24],
        "target_lag168": y.iloc[target_idx - 168],
        "target": y.iloc[target_idx],
    }
    if use_future_weather:
        for c in WEATHER_COLS:
            row[f"future_{c}"] = df[c].iloc[target_idx]
    return row


def build_dataset(df: pd.DataFrame, origin_positions, horizon: int = 24,
                   use_future_weather: bool = True) -> pd.DataFrame:
    """Build the direct multi-horizon feature table for a list of origin positions (iloc ints)."""
    y = df["Appliances"]
    rows = [
        _row_features(df, y, origin_idx, h, use_future_weather)
        for origin_idx in origin_positions
        for h in range(1, horizon + 1)
    ]
    return pd.DataFrame(rows)


def training_origin_positions(n_train: int, horizon: int = 24):
    """All valid origins strictly within the training period."""
    return list(range(MIN_HISTORY_HOURS, n_train - horizon))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from forecast_utils import load_hourly, train_test_split

    df = load_hourly()
    train, test = train_test_split(df)
    origins = training_origin_positions(len(train))
    print(f"Building feature table for {len(origins)} training origins x 24 horizons "
          f"= {len(origins)*24} rows ...")

    feat_true = build_dataset(df, origins, use_future_weather=False)
    print(f"True-forecast feature set: {feat_true.shape}")
    print(feat_true.columns.tolist())

    feat_cond = build_dataset(df, origins[:5], use_future_weather=True)  # small sample to check
    print(f"\nConditional feature set (sample): {feat_cond.shape}")
    print(feat_cond.columns.tolist())
