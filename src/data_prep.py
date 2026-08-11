"""
data_prep.py
------------
Part 1: Load the raw Appliances Energy Prediction dataset, parse timestamps,
check for missing values, and resample from 10-minute to hourly resolution.

Source data: UCI Appliances Energy Prediction dataset (Candanedo et al., 2017),
downloaded from the original authors' repository (mirror of the UCI archive file).
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "energydata_complete.csv"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw 10-minute resolution CSV and parse the timestamp column."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df


def check_missing(df: pd.DataFrame) -> pd.Series:
    """Return count of missing values per column (0 expected for this dataset)."""
    return df.isna().sum()


def check_time_regularity(df: pd.DataFrame) -> dict:
    """Check that timestamps are evenly spaced at 10-minute intervals."""
    diffs = df.index.to_series().diff().dropna()
    expected = pd.Timedelta(minutes=10)
    n_gaps = (diffs != expected).sum()
    return {
        "n_rows": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "expected_interval": expected,
        "n_irregular_intervals": int(n_gaps),
        "min_interval": diffs.min(),
        "max_interval": diffs.max(),
    }


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample from 10-minute to hourly resolution.

    Appliances and lights (energy, Wh per 10-min reading) are SUMMED to give
    total Wh consumed in each hour. All other sensor/weather columns are
    AVERAGED, since they are instantaneous readings (temperature, humidity,
    pressure, etc.), not additive quantities.
    """
    energy_cols = ["Appliances", "lights"]
    other_cols = [c for c in df.columns if c not in energy_cols]

    hourly_energy = df[energy_cols].resample("h").sum()
    hourly_other = df[other_cols].resample("h").mean()

    hourly = pd.concat([hourly_energy, hourly_other], axis=1)
    hourly = hourly[df.columns]  # preserve original column order
    return hourly


def run(save: bool = True) -> pd.DataFrame:
    df_raw = load_raw()
    missing = check_missing(df_raw)
    regularity = check_time_regularity(df_raw)

    print("=== Raw data summary ===")
    print(f"Shape: {df_raw.shape}")
    print(f"Date range: {regularity['start']} -> {regularity['end']}")
    print(f"Irregular 10-min intervals: {regularity['n_irregular_intervals']}")
    print(f"Total missing values across all columns: {missing.sum()}")

    df_hourly = resample_hourly(df_raw)
    print("\n=== Hourly resampled data ===")
    print(f"Shape: {df_hourly.shape}")
    print(f"Missing values after resampling: {df_hourly.isna().sum().sum()}")

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df_raw.to_parquet(PROCESSED_DIR / "energy_10min.parquet")
        df_hourly.to_parquet(PROCESSED_DIR / "energy_hourly.parquet")
        print(f"\nSaved to {PROCESSED_DIR}")

    return df_hourly


if __name__ == "__main__":
    run()
