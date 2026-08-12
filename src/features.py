"""
Building the feature table for the XGBoost model in Part 6.

Went with a "direct multi-horizon" setup - one row per (origin,
hours_ahead) combo instead of one row per hour, so a single model can
predict any hour in the next day (hours_ahead just becomes a feature).
Took me a bit to land on this design - the tricky part is that features
like "value 1 hour ago" don't actually work for a 24h horizon because by
hour 5 you don't know what happened at hour 4 yet. Worked around that by
using lags relative to the TARGET hour instead of the origin, for anything
where lag >= max horizon (24h and 168h lags are always safe this way,
1h lag isn't so that stays relative to the origin only).

Two versions of the feature set get built (see build_dataset's
use_future_weather flag):
- honest version: only stuff you'd actually know standing at the origin
  (lags, rolling stats, time-of-day/day-of-week of the target)
- "conditional" version: same but also includes the REAL future
  weather/indoor sensor readings, i.e. cheating a bit. Wanted to measure
  how much this actually helps (spoiler: not much, it's discussed in the
  report under Q5 - this isn't a real forecast if you're using future
  weather you couldn't have actually known yet).
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
    # origin_positions are just row indices (iloc-style ints), one feature
    # row gets built per origin per hour ahead (1 to horizon)
    y = df["Appliances"]
    rows = [
        _row_features(df, y, origin_idx, h, use_future_weather)
        for origin_idx in origin_positions
        for h in range(1, horizon + 1)
    ]
    return pd.DataFrame(rows)


def training_origin_positions(n_train: int, horizon: int = 24):
    # need MIN_HISTORY_HOURS before an origin (for the lag_168 feature) and
    # `horizon` hours after it (so the target actually exists in training data)
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
