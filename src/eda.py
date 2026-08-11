"""
eda.py
------
Part 1 (continued): Exploratory analysis of the hourly Appliances series.
- Full time series plot
- Average daily profile (hour-of-day seasonality)
- Average weekly profile (day-of-week seasonality)
- STL / classical seasonal decomposition
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110


def load_hourly() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "energy_hourly.parquet")


def plot_full_series(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["Appliances"], lw=0.7, color="#2b6cb0")
    ax.set_title("Hourly Appliance Energy Use — Full Series (Jan–May 2016)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliances (Wh)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_full_series.png")
    plt.close(fig)


def plot_daily_profile(df: pd.DataFrame):
    hourly_avg = df.groupby(df.index.hour)["Appliances"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hourly_avg.index, hourly_avg["mean"], marker="o", color="#2b6cb0")
    ax.fill_between(
        hourly_avg.index,
        hourly_avg["mean"] - hourly_avg["std"],
        hourly_avg["mean"] + hourly_avg["std"],
        alpha=0.2,
        color="#2b6cb0",
    )
    ax.set_title("Average Appliance Energy Use by Hour of Day (±1 std)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Appliances (Wh)")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_daily_profile.png")
    plt.close(fig)


def plot_weekly_profile(df: pd.DataFrame):
    dow_avg = df.groupby(df.index.dayofweek)["Appliances"].mean()
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, dow_avg.values, color="#2b6cb0")
    ax.set_title("Average Appliance Energy Use by Day of Week")
    ax.set_ylabel("Appliances (Wh)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_weekly_profile.png")
    plt.close(fig)


def plot_decomposition(df: pd.DataFrame, period: int = 24):
    """STL decomposition using a daily (24h) period."""
    stl = STL(df["Appliances"], period=period, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(df.index, res.observed, lw=0.6, color="#2b6cb0")
    axes[0].set_ylabel("Observed")
    axes[1].plot(df.index, res.trend, lw=0.8, color="#c05621")
    axes[1].set_ylabel("Trend")
    axes[2].plot(df.index, res.seasonal, lw=0.6, color="#2f855a")
    axes[2].set_ylabel("Seasonal\n(24h)")
    axes[3].scatter(df.index, res.resid, s=2, color="#718096")
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Date")
    fig.suptitle("STL Decomposition — Hourly Appliance Energy Use (period=24h)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_stl_decomposition.png")
    plt.close(fig)
    return res


if __name__ == "__main__":
    df = load_hourly()
    plot_full_series(df)
    plot_daily_profile(df)
    plot_weekly_profile(df)
    res = plot_decomposition(df)

    print("Seasonal component strength check:")
    import numpy as np
    var_resid = res.resid.var()
    var_seasonal_resid = (res.seasonal + res.resid).var()
    strength = max(0, 1 - var_resid / var_seasonal_resid)
    print(f"  Seasonal strength (daily, 24h): {strength:.3f}  (0=none, 1=strong)")
    print(f"Figures saved to {FIG_DIR}")
