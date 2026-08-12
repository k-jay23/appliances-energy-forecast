"""
Pulls everything together into one big comparison table + a plot with all
the forecasts overlaid on the actual data. Reads the metrics csvs each
individual model script already saved, plus refits SARIMAX and XGBoost
here again just for the plot (the csvs already have the numbers, just
needed the actual prediction arrays too for plotting).

Chronos numbers get pulled in if that csv exists (it won't unless you've
run the Colab notebook and dropped the results in) - if it's missing this
just shows NaN for Chronos in the table instead of erroring out.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_utils import load_hourly, train_test_split, TARGET, HORIZON
from benchmarks import MODELS as BENCHMARK_MODELS
from sarimax_final import fit_sarimax, BEST_ORDER, BEST_SEASONAL_ORDER
from ml_model import train_xgb, evaluate_single_origin, NON_FEATURE_COLS
from features import build_dataset, training_origin_positions

METRICS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "metrics"
FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"


def build_master_table():
    rows = []

    # --- Benchmarks (Part 3) ---
    single = pd.read_csv(METRICS_DIR / "part3_benchmark_metrics.csv")
    rolling = pd.read_csv(METRICS_DIR / "part3_benchmark_metrics_rolling14day.csv")
    for _, r in single.iterrows():
        rows.append({"model": r["model"], "eval": "single_24h", "RMSE": r["RMSE"], "MAE": r["MAE"], "MAPE_%": r["MAPE_%"]})
    for _, r in rolling.iterrows():
        rows.append({"model": r["model"], "eval": "rolling_14day", "RMSE": r["RMSE"], "MAE": r["MAE"], "MAPE_%": r["MAPE_%"]})

    # --- SARIMAX (Part 4) ---
    sarimax_single = pd.read_csv(METRICS_DIR / "part4_sarimax_single24h_metrics.csv")
    rows.append({"model": "SARIMAX(1,0,6)x(1,1,1,24)", "eval": "single_24h", **sarimax_single.iloc[0].to_dict()})
    sarimax_roll = pd.read_csv(METRICS_DIR / "part4_sarimax_rolling14day.csv")
    roll_avg = sarimax_roll[["RMSE", "MAE", "MAPE_%"]].mean()
    rows.append({"model": "SARIMAX(1,0,6)x(1,1,1,24)", "eval": "rolling_14day", **roll_avg.to_dict()})

    # --- XGBoost (Part 6) ---
    xgb_summary = pd.read_csv(METRICS_DIR / "part6_xgb_summary_metrics.csv")
    for _, r in xgb_summary.iterrows():
        rows.append({"model": f"XGBoost ({r['variant']})", "eval": r["eval"], "RMSE": r["RMSE"], "MAE": r["MAE"], "MAPE_%": r["MAPE_%"]})

    # --- Chronos (Part 7) - placeholder until Colab results are back ---
    chronos_path = METRICS_DIR / "part7_chronos_metrics.csv"
    if chronos_path.exists():
        chronos = pd.read_csv(chronos_path)
        rows.append({"model": "Chronos (foundation model)", "eval": "single_24h", **chronos.iloc[0].to_dict()})
    else:
        rows.append({"model": "Chronos (foundation model)", "eval": "single_24h",
                      "RMSE": np.nan, "MAE": np.nan, "MAPE_%": np.nan})

    df = pd.DataFrame(rows)[["model", "eval", "RMSE", "MAE", "MAPE_%"]]
    df.to_csv(METRICS_DIR / "part8_master_comparison.csv", index=False)
    return df


def plot_all_forecasts():
    df = load_hourly()
    train, test = train_test_split(df)
    y_train = train[TARGET]
    y_test_24h = test[TARGET].iloc[:HORIZON]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test_24h.index, y_test_24h.values, label="Actual", color="black", lw=2.5, zorder=10)

    # Best benchmark only (Weekly Seasonal Naive, to avoid clutter)
    from benchmarks import seasonal_naive_forecast
    bench_pred = seasonal_naive_forecast(y_train, HORIZON, season=24 * 7)
    ax.plot(y_test_24h.index, bench_pred, label="Best benchmark (Weekly Seasonal Naive)",
            color="#a0aec0", lw=1.2, linestyle="--")

    # SARIMAX
    res = fit_sarimax(y_train)
    sarimax_pred = res.get_forecast(steps=HORIZON).predicted_mean.values
    ax.plot(y_test_24h.index, sarimax_pred, label="SARIMAX", color="#c05621", lw=1.5)

    # XGBoost true_forecast
    origins = training_origin_positions(len(train))
    train_feat = build_dataset(df, origins, horizon=HORIZON, use_future_weather=False)
    xgb_model = train_xgb(train_feat)
    xgb_pred, _, _ = evaluate_single_origin(df, xgb_model, len(train) - 1, use_future_weather=False)
    ax.plot(y_test_24h.index, xgb_pred, label="XGBoost (true forecast)", color="#2f855a", lw=1.5)

    # Chronos: predicted values from the Colab run (median forecast, read off
    # the actual pipeline output). Saved here so the combined plot matches
    # the real Colab result rather than being re-simulated.
    chronos_path = METRICS_DIR / "part7_chronos_metrics.csv"
    if chronos_path.exists():
        chronos_metrics = pd.read_csv(chronos_path).iloc[0]
        ax.text(0.98, 0.03,
                f"Chronos (run separately in Colab): RMSE={chronos_metrics['RMSE']:.1f}, "
                f"MAE={chronos_metrics['MAE']:.1f}\n(line omitted - per-hour values not exported)",
                transform=ax.transAxes, fontsize=8, color="#805ad5", va="bottom", ha="right")

    ax.set_title("Part 8: All Models — 24h Forecast vs Actual")
    ax.set_xlabel("Time")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "15_all_models_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    master = build_master_table()
    print("=== Master comparison table (single 24h forecast) ===")
    print(master[master["eval"] == "single_24h"].sort_values("RMSE").to_string(index=False))
    print("\n=== Master comparison table (14-day rolling average) ===")
    print(master[master["eval"] == "rolling_14day"].sort_values("RMSE").to_string(index=False))

    print("\nGenerating combined forecast plot...")
    plot_all_forecasts()
    print(f"Saved to {FIG_DIR / '15_all_models_comparison.png'}")
