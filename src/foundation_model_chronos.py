"""
Part 7 - Chronos foundation model. Zero-shot forecasting, meaning no
training on our data at all, just feed it recent history and it predicts
forward based on patterns it picked up from a huge pile of other time
series during pretraining.

Couldn't run this one in the same place as everything else - my main
environment blocks HuggingFace (that's where the pretrained weights live),
so I had to run this separately in Google Colab instead. Kind of annoying
but it only took a few minutes:

1. colab.research.google.com -> new notebook
2. cell 1: !pip install -q chronos-forecasting torch pandas matplotlib
3. cell 2: paste everything below (from "import pandas" down) and run it.
   Grabs the data itself off github so no need to upload anything.
4. prints RMSE/MAE/MAPE at the end - that's what went into the report.
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

# feed it the last CONTEXT_HOURS as history, ask for 24h ahead
# (note: context has to be positional here, not a kwarg - the installed
# version of chronos-forecasting doesn't accept context= and throws a
# "got an unexpected keyword argument" error otherwise, took me a minute
# to figure that one out)
context = torch.tensor(y_train.iloc[-CONTEXT_HOURS:].values, dtype=torch.float32)
forecast = pipeline.predict(context, prediction_length=HORIZON, num_samples=100)
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
