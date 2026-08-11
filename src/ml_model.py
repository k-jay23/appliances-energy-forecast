"""
ml_model.py
-----------
Part 6: Feature-based ML model (XGBoost) using the features built in Part 5.

We train TWO versions to directly answer Part 9 Question 5 later:
  1. "true_forecast" - only genuinely-knowable-in-advance features
  2. "conditional"   - same, PLUS actual future weather/indoor sensor readings
                       (this is a conditional forecast, not a real one - see
                       features.py docstring)

Both are evaluated the same way as every previous model: a single 24h-ahead
forecast, AND a 14-day rolling average, for fair comparison.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_utils import load_hourly, train_test_split, regression_metrics, TARGET, HORIZON
from features import build_dataset, training_origin_positions

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
METRICS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "metrics"
plt.rcParams["figure.dpi"] = 110

NON_FEATURE_COLS = ["origin_time", "target_time", "target"]


def train_xgb(train_feat: pd.DataFrame) -> XGBRegressor:
    X = train_feat.drop(columns=NON_FEATURE_COLS)
    y = train_feat["target"]
    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X, y)
    return model


def evaluate_single_origin(df, model, origin_pos, use_future_weather, horizon=HORIZON):
    feat = build_dataset(df, [origin_pos], horizon=horizon, use_future_weather=use_future_weather)
    X = feat.drop(columns=NON_FEATURE_COLS)
    pred = model.predict(X)
    actual = feat["target"].values
    return pred, actual, feat


def rolling_evaluate_ml(df, model, start_origin_pos, use_future_weather, n_days=14, horizon=HORIZON):
    rows = []
    for day in range(n_days):
        origin_pos = start_origin_pos + day * horizon
        pred, actual, _ = evaluate_single_origin(df, model, origin_pos, use_future_weather, horizon)
        m = regression_metrics(actual, pred)
        m["origin_day"] = day + 1
        m["origin_date"] = str(df.index[origin_pos + 1].date())
        rows.append(m)
    return pd.DataFrame(rows)[["origin_day", "origin_date", "RMSE", "MAE", "MAPE_%"]]


def run():
    df = load_hourly()
    train, test = train_test_split(df)
    train_origin_pos = len(train) - 1  # standard single 24h forecast origin

    origins = training_origin_positions(len(train))
    print(f"Training on {len(origins)} origins ({len(origins)*24} rows per feature set)")

    results_summary = []
    models = {}
    for variant, use_fw in [("true_forecast", False), ("conditional", True)]:
        print(f"\n=== Training XGBoost - {variant} feature set (use_future_weather={use_fw}) ===")
        train_feat = build_dataset(df, origins, horizon=HORIZON, use_future_weather=use_fw)
        model = train_xgb(train_feat)
        models[variant] = model

        # single 24h forecast
        pred, actual, feat_eval = evaluate_single_origin(df, model, train_origin_pos, use_fw)
        single_metrics = regression_metrics(actual, pred)
        print(f"Single 24h forecast: {single_metrics}")

        # rolling 14-day
        roll_df = rolling_evaluate_ml(df, model, train_origin_pos, use_fw)
        roll_df.to_csv(METRICS_DIR / f"part6_xgb_{variant}_rolling14day.csv", index=False)
        roll_avg = roll_df[["RMSE", "MAE", "MAPE_%"]].mean()
        print(f"14-day rolling average: {roll_avg.to_dict()}")

        results_summary.append({"variant": variant, "eval": "single_24h", **single_metrics})
        results_summary.append({"variant": variant, "eval": "rolling_14day", **roll_avg.to_dict()})

        # Plot single 24h forecast
        target_times = feat_eval["target_time"]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(target_times, actual, label="Actual", color="black", lw=2)
        ax.plot(target_times, pred, label=f"XGBoost ({variant})", color="#2f855a", lw=1.5)
        ax.set_title(f"Part 6: XGBoost ({variant}) — 24h Forecast vs Actual")
        ax.set_xlabel("Time")
        ax.set_ylabel("Appliances (Wh)")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"12_xgb_{variant}_forecast.png")
        plt.close(fig)

        # Feature importance
        X_cols = train_feat.drop(columns=NON_FEATURE_COLS).columns
        importances = pd.Series(model.feature_importances_, index=X_cols).sort_values(ascending=False)
        importances.to_csv(METRICS_DIR / f"part6_xgb_{variant}_feature_importance.csv")

        fig, ax = plt.subplots(figsize=(8, max(4, len(importances) * 0.25)))
        importances.head(15).sort_values().plot.barh(ax=ax, color="#2f855a")
        ax.set_title(f"Top 15 Feature Importances — XGBoost ({variant})")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"13_xgb_{variant}_feature_importance.png")
        plt.close(fig)

    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(METRICS_DIR / "part6_xgb_summary_metrics.csv", index=False)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

    return models, summary_df


if __name__ == "__main__":
    run()
