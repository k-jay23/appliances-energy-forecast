"""
sarimax_rolling.py (resumable)
--------------------------------
Part 4/8: robust check - repeat the 24h SARIMAX forecast across all 14
held-out days (refitting each time with the growing history), same idea as
the rolling check used for the Part 3 benchmarks. Resumable in batches.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import time
import pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_utils import load_hourly, train_test_split, regression_metrics, TARGET, HORIZON

METRICS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "metrics"
OUT_CSV = METRICS_DIR / "part4_sarimax_rolling14day.csv"

BEST_ORDER = (1, 0, 6)
BEST_SEASONAL_ORDER = (1, 1, 1, 24)
TIME_BUDGET_SECONDS = float(os.environ.get("TIME_BUDGET_SECONDS", 170))


def fit_and_forecast(history, horizon=HORIZON):
    model = SARIMAX(history, order=BEST_ORDER, seasonal_order=BEST_SEASONAL_ORDER,
                     enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False, method="lbfgs", maxiter=75)
    return res.get_forecast(steps=horizon).predicted_mean.values


def main():
    df = load_hourly()
    train, test = train_test_split(df)
    y_train = train[TARGET]
    y_test = test[TARGET]

    n_days = len(y_test) // HORIZON  # 14

    done_days = set()
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        done_days = set(prev["origin_day"])
    else:
        with open(OUT_CSV, "w") as f:
            f.write("origin_day,origin_date,RMSE,MAE,MAPE_%\n")

    print(f"Days total={n_days}, already done={len(done_days)}")
    t_start = time.time()

    for day in range(n_days):
        if (day + 1) in done_days:
            continue
        if time.time() - t_start > TIME_BUDGET_SECONDS:
            print(f"Time budget reached. Stopping. Remaining days: "
                  f"{n_days - len(done_days)}")
            return

        start = day * HORIZON
        end = start + HORIZON
        history = pd.concat([y_train, y_test.iloc[:start]]) if start > 0 else y_train
        actual = y_test.iloc[start:end].values

        t0 = time.time()
        pred = fit_and_forecast(history)
        m = regression_metrics(actual, pred)
        elapsed = time.time() - t0

        with open(OUT_CSV, "a") as f:
            f.write(f"{day+1},{y_test.index[start].date()},{m['RMSE']},{m['MAE']},{m['MAPE_%']}\n")

        done_days.add(day + 1)
        print(f"Day {day+1}/{n_days} ({y_test.index[start].date()}) "
              f"RMSE={m['RMSE']:.1f} MAE={m['MAE']:.1f} fit_time={elapsed:.1f}s", flush=True)

    print("ALL DONE")


if __name__ == "__main__":
    main()
