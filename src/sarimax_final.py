"""
Takes the winning (p,d,q) from the grid search and actually uses it -
refits on the full training data (grid search only used the last 45 days
for speed), checks the residuals look ok, then forecasts the 24h test window.

BEST_ORDER is hardcoded here from the grid search results rather than read
dynamically - could have wired that up automatically but honestly just
copy-pasted the winning order in after eyeballing the results csv.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_utils import (load_hourly, train_test_split, regression_metrics,
                             rolling_evaluate, TARGET, HORIZON)

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
METRICS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "metrics"
plt.rcParams["figure.dpi"] = 110

BEST_ORDER = (1, 0, 6)
BEST_SEASONAL_ORDER = (1, 1, 1, 24)


def fit_sarimax(y_train, order=BEST_ORDER, seasonal_order=BEST_SEASONAL_ORDER):
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False, method="lbfgs", maxiter=100)


def main():
    df = load_hourly()
    train, test = train_test_split(df)
    y_train = train[TARGET]
    y_test_24h = test[TARGET].iloc[:HORIZON]

    print(f"Refitting SARIMA{BEST_ORDER}x{BEST_SEASONAL_ORDER} on full training set "
          f"({len(y_train)} obs)...")
    res = fit_sarimax(y_train)
    print(res.summary().tables[0])

    # dropping the first bit of residuals since the model needs some data
    # to "warm up" before its predictions are meaningful (burn-in period)
    resid = res.resid[BEST_SEASONAL_ORDER[3] + max(BEST_ORDER):]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(resid, lags=48, ax=axes[0])
    axes[0].set_title("ACF of SARIMAX Residuals")
    axes[1].hist(resid, bins=40, color="#2b6cb0", edgecolor="white")
    axes[1].set_title("Distribution of SARIMAX Residuals")
    axes[1].set_xlabel("Residual (Wh)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_sarimax_residual_diagnostics.png")
    plt.close(fig)

    resid_stats = {
        "mean": float(resid.mean()),
        "std": float(resid.std()),
        "skew": float(resid.skew()),
        "kurtosis": float(resid.kurtosis()),
    }
    print(f"\nResidual stats: {resid_stats}")

    fc = res.get_forecast(steps=HORIZON)
    pred_mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)  # 95% CI - heads up, this goes negative
    # sometimes which obviously can't happen for energy use. noted this
    # in the report, it's a known thing with gaussian CIs on skewed data

    metrics = regression_metrics(y_test_24h.values, pred_mean.values)
    print(f"\n24h-ahead forecast metrics: {metrics}")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(y_test_24h.index, y_test_24h.values, label="Actual", color="black", lw=2)
    ax.plot(y_test_24h.index, pred_mean.values, label="SARIMAX forecast", color="#c05621", lw=1.5)
    ax.fill_between(y_test_24h.index, ci.iloc[:, 0], ci.iloc[:, 1],
                     color="#c05621", alpha=0.2, label="95% CI")
    ax.set_title(f"Part 4: SARIMA{BEST_ORDER}x{BEST_SEASONAL_ORDER} — 24h Forecast vs Actual")
    ax.set_xlabel("Time")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_sarimax_forecast.png")
    plt.close(fig)

    pd.DataFrame([metrics]).to_csv(METRICS_DIR / "part4_sarimax_single24h_metrics.csv", index=False)
    print(f"\nSaved figures to {FIG_DIR}, metrics to {METRICS_DIR}")
    print("(Rolling 14-day evaluation is run separately in sarimax_rolling.py)")

    return res, metrics


if __name__ == "__main__":
    main()
