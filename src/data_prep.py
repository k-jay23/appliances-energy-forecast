"""
Part 1 - loading and cleaning the raw dataset.

Just reads the UCI Appliances Energy Prediction csv, parses the timestamps,
and bins it up to hourly since the raw data is every 10 mins which is way
too fine-grained for what we need. Checked it for missing values / gaps too,
turns out this dataset is basically spotless (no NaNs, no missing timestamps)
which made this part easy.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "energydata_complete.csv"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    # date column comes in as a string by default, need it as an actual
    # datetime so we can resample later
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df


def check_missing(df: pd.DataFrame) -> pd.Series:
    # sanity check, should be all zeros for this dataset
    return df.isna().sum()


def check_time_regularity(df: pd.DataFrame) -> dict:
    """Make sure there aren't any gaps in the 10-min readings before we resample."""
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
    # Appliances/lights are Wh readings so they need to be SUMMED across the
    # hour (adding up energy used makes sense). Everything else (temp,
    # humidity, pressure etc) is an instantaneous reading so averaging is
    # the right call there - you can't "sum" a temperature.
    energy_cols = ["Appliances", "lights"]
    other_cols = [c for c in df.columns if c not in energy_cols]

    hourly_energy = df[energy_cols].resample("h").sum()
    hourly_other = df[other_cols].resample("h").mean()

    hourly = pd.concat([hourly_energy, hourly_other], axis=1)
    hourly = hourly[df.columns]  # keep the same column order as original
    return hourly


def run(save: bool = True) -> pd.DataFrame:
    df_raw = load_raw()
    missing = check_missing(df_raw)
    regularity = check_time_regularity(df_raw)

    # just printing everything out so I can eyeball it looks right
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
