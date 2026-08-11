"""
foundation_model_chronos.py
-----------------------------
Part 7: Zero-shot forecast using Amazon's "Chronos" time-series foundation
model. Chronos is pretrained on a huge collection of public time series and
can forecast a NEW series (ours) with no training at all - you just give it
recent history ("context") and ask for a forecast.

*** RUN THIS IN GOOGLE COLAB, NOT IN THE PROJECT SANDBOX ***
(The sandbox used for the rest of this project blocks HuggingFace, which is
where the pretrained Chronos weights are hosted, so this script cannot run
there. Colab has normal internet access.)

HOW TO RUN IN COLAB (5 minutes, no file uploads needed):
1. Go to https://colab.research.google.com , click "New notebook".
2. In the FIRST cell, paste and run just this line:
   !pip install -q chronos-forecasting torch pandas matplotlib
3. In a SECOND cell, paste everything below from "import pandas" onwards, and run it.
   (It downloads the dataset itself and does the resampling, so no upload needed.)
4. It will print something like:
   Chronos 24h-ahead forecast: RMSE=xxx.xx  MAE=xxx.xx  MAPE%=xx.xx
   Copy that line and paste it back to me, along with a description of the
   plot if you'd like (or download 14_chronos_forecast.png from the file
   panel on the left and share it).
"""

# !pip install -q chronos-forecasting torch pandas matplotlib   # run this in its own cell first

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import ChronosPipeline

TARGET = "Appliances"
HORIZON = 24
TEST_DAYS = 14
CONTEXT_HOURS = 24 * 14  # give the model the last 14 days as context

# --- Load + prepare data directly (no file upload needed - Colab has internet) ---
url = "https://raw.githubusercontent.com/LuisM78/Appliances-energy-prediction-data/master/energydata_complete.csv"
raw = pd.read_csv(url)
raw["date"] = pd.to_datetime(raw["date"])
raw = raw.sort_values("date").set_index("date")

energy_cols = ["Appliances", "lights"]
other_cols = [c for c in raw.columns if c not in energy_cols]
hourly = pd.concat([raw[energy_cols].resample("h").sum(),
                     raw[other_cols].resample("h").mean()], axis=1)

n_test = TEST_DAYS * 24
train, test = hourly.iloc[:-n_test], hourly.iloc[-n_test:]
y_train = train[TARGET]
y_test_24h = test[TARGET].iloc[:HORIZON]
print(f"Train: {len(train)} rows, Test: {len(test)} rows. "
      f"Forecasting {y_test_24h.index[0]} -> {y_test_24h.index[-1]}")

# --- Load pretrained Chronos model (downloads weights from HuggingFace, ~small) ---
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",   # small = good speed/accuracy tradeoff on CPU
    device_map="cpu",
    torch_dtype=torch.float32,
)

# --- Forecast: give it the last CONTEXT_HOURS of TRAINING data, ask for 24h ahead ---
context = torch.tensor(y_train.iloc[-CONTEXT_HOURS:].values, dtype=torch.float32)
forecast = pipeline.predict(context=context, prediction_length=HORIZON, num_samples=100)
# forecast shape: [num_series, num_samples, horizon] -> take median as point forecast
pred_median = np.median(forecast[0].numpy(), axis=0)
pred_low = np.quantile(forecast[0].numpy(), 0.05, axis=0)
pred_high = np.quantile(forecast[0].numpy(), 0.95, axis=0)

# --- Evaluate ---
actual = y_test_24h.values
errors = actual - pred_median
rmse = float(np.sqrt(np.mean(errors ** 2)))
mae = float(np.mean(np.abs(errors)))
mape = float(np.mean(np.abs(errors[actual != 0] / actual[actual != 0])) * 100)
print(f"Chronos 24h-ahead forecast: RMSE={rmse:.2f}  MAE={mae:.2f}  MAPE%={mape:.2f}")

pd.DataFrame([{"RMSE": rmse, "MAE": mae, "MAPE_%": mape}]).to_csv(
    "part7_chronos_metrics.csv", index=False)

# --- Plot ---
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(y_test_24h.index, actual, label="Actual", color="black", lw=2)
ax.plot(y_test_24h.index, pred_median, label="Chronos forecast (median)", color="#805ad5", lw=1.5)
ax.fill_between(y_test_24h.index, pred_low, pred_high, color="#805ad5", alpha=0.2, label="90% interval")
ax.set_title("Part 7: Chronos Foundation Model — 24h Forecast vs Actual")
ax.set_xlabel("Time")
ax.set_ylabel("Appliances (Wh)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("14_chronos_forecast.png")
plt.show()
print("Saved part7_chronos_metrics.csv and 14_chronos_forecast.png — download these from Colab's file panel.")
