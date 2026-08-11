"""
stationarity.py
----------------
Part 1 (continued): Formal stationarity testing for the hourly Appliances series.
- Augmented Dickey-Fuller (ADF) test on the raw series
- ACF / PACF plots on the raw series
- First-order and seasonal (24h) differencing
- ADF re-test on differenced series
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
METRICS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110


def adf_report(series: pd.Series, label: str) -> dict:
    series = series.dropna()
    result = adfuller(series, autolag="AIC")
    report = {
        "series": label,
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags_used": result[2],
        "n_obs": result[3],
        "crit_1%": result[4]["1%"],
        "crit_5%": result[4]["5%"],
        "crit_10%": result[4]["10%"],
        "stationary_at_5%": result[1] < 0.05,
    }
    return report


def plot_acf_pacf(series: pd.Series, label: str, filename: str, lags: int = 72):
    series = series.dropna()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF — {label}")
    plot_pacf(series, lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"PACF — {label}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_DIR / "energy_hourly.parquet")
    y = df["Appliances"]

    reports = []

    # 1. Raw series
    reports.append(adf_report(y, "raw (level)"))
    plot_acf_pacf(y, "Raw series", "05_acf_pacf_raw.png")

    # 2. First-order differenced (removes stochastic trend / short-run non-stationarity)
    y_diff1 = y.diff()
    reports.append(adf_report(y_diff1, "first-difference (d=1)"))
    plot_acf_pacf(y_diff1, "First difference (d=1)", "06_acf_pacf_diff1.png")

    # 3. Seasonal differenced (24h) - removes daily seasonality
    y_seasonal_diff = y.diff(24)
    reports.append(adf_report(y_seasonal_diff, "seasonal difference (24h)"))
    plot_acf_pacf(y_seasonal_diff, "Seasonal difference (24h)", "07_acf_pacf_seasonal_diff.png")

    # 4. Both first + seasonal differenced
    y_both = y.diff().diff(24)
    reports.append(adf_report(y_both, "diff(1) + seasonal diff(24)"))
    plot_acf_pacf(y_both, "diff(1) + seasonal diff(24)", "08_acf_pacf_diff1_seasonal.png")

    report_df = pd.DataFrame(reports)
    report_df.to_csv(METRICS_DIR / "adf_stationarity_report.csv", index=False)

    print(report_df.to_string(index=False))
    print(f"\nSaved ADF report to {METRICS_DIR / 'adf_stationarity_report.csv'}")
    print(f"Saved ACF/PACF plots to {FIG_DIR}")
