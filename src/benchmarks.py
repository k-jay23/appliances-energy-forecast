"""
The 5 baseline forecasts (mean, naive, daily/weekly seasonal naive, drift).
These are supposed to be dumb on purpose - the point is everything fancier
I build later actually needs to beat these to be worth using.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from forecast_utils import load_hourly, train_test_split, regression_metrics, rolling_evaluate, TARGET, HORIZON

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
METRICS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams["figure.dpi"] = 110


def mean_forecast(train: pd.Series, horizon: int = HORIZON) -> np.ndarray:
    # just guess the average, every hour
    return np.full(horizon, train.mean())


def naive_forecast(train: pd.Series, horizon: int = HORIZON) -> np.ndarray:
    # guess tomorrow = whatever the last value was (flat line)
    return np.full(horizon, train.iloc[-1])


def seasonal_naive_forecast(train: pd.Series, horizon: int = HORIZON, season: int = 24) -> np.ndarray:
    # season=24 -> "copy yesterday", season=168 -> "copy same day last week"
    last_season = train.iloc[-season:].values
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season, reps)[:horizon]


def drift_forecast(train: pd.Series, horizon: int = HORIZON) -> np.ndarray:
    # naive but extrapolates a straight line through the whole training set
    y0, yT = train.iloc[0], train.iloc[-1]
    T = len(train) - 1
    slope = (yT - y0) / T if T > 0 else 0
    steps = np.arange(1, horizon + 1)
    return yT + slope * steps


MODELS = {
    "Mean": lambda train, h: mean_forecast(train, h),
    "Naive": lambda train, h: naive_forecast(train, h),
    "Daily Seasonal Naive": lambda train, h: seasonal_naive_forecast(train, h, season=24),
    "Weekly Seasonal Naive": lambda train, h: seasonal_naive_forecast(train, h, season=24 * 7),
    "Drift": lambda train, h: drift_forecast(train, h),
}


def run():
    df = load_hourly()
    train, test = train_test_split(df)
    y_train = train[TARGET]
    y_test_24h = test[TARGET].iloc[:HORIZON]

    results = []
    forecasts = {}
    for name, fn in MODELS.items():
        pred = fn(y_train, HORIZON)
        forecasts[name] = pred
        m = regression_metrics(y_test_24h.values, pred)
        m["model"] = name
        results.append(m)

    results_df = pd.DataFrame(results)[["model", "RMSE", "MAE", "MAPE_%"]].sort_values("RMSE")
    results_df.to_csv(METRICS_DIR / "part3_benchmark_metrics.csv", index=False)
    print("=== Benchmark model results (24h-ahead forecast, ranked best first) ===")
    print(results_df.to_string(index=False))

    # plot everyone on top of the actual values so you can see how bad naive/drift are
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(y_test_24h.index, y_test_24h.values, label="Actual", color="black", lw=2)
    colors = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#d53f8c"]
    for (name, pred), color in zip(forecasts.items(), colors):
        ax.plot(y_test_24h.index, pred, label=name, lw=1.3, color=color, alpha=0.85)
    ax.set_title("Part 3: Benchmark Forecasts vs Actual (first 24h of test period)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend(fontsize=8, ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_benchmark_forecasts.png")
    plt.close(fig)

    print(f"\nSaved metrics to {METRICS_DIR / 'part3_benchmark_metrics.csv'}")
    print(f"Saved plot to {FIG_DIR / '09_benchmark_forecasts.png'}")

    # single-day test above was a bit suspicious (weekly seasonal naive won
    # but naive/drift were terrible - turned out training just happened to
    # end right on an evening peak). doing the rolling check to be sure.
    print("\n=== Robust check: average error across all 14 held-out days (rolling) ===")
    rolling_summary = []
    for name, fn in MODELS.items():
        roll_df = rolling_evaluate(y_train, test[TARGET], lambda h, hz, fn=fn: fn(h, hz))
        summary = roll_df[["RMSE", "MAE", "MAPE_%"]].mean()
        summary["model"] = name
        rolling_summary.append(summary)
    rolling_df = pd.DataFrame(rolling_summary)[["model", "RMSE", "MAE", "MAPE_%"]].sort_values("RMSE")
    rolling_df.to_csv(METRICS_DIR / "part3_benchmark_metrics_rolling14day.csv", index=False)
    print(rolling_df.to_string(index=False))
    print(f"\nSaved rolling metrics to {METRICS_DIR / 'part3_benchmark_metrics_rolling14day.csv'}")

    return results_df, forecasts, rolling_df


if __name__ == "__main__":
    run()
