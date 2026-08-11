# Part 9 — Discussion Questions (answers for the report)

*(All 6 questions answered, grounded in the actual results produced by this
project — including the real Chronos numbers from the Colab run.)*

---

### Q1. Which benchmark model is strongest — naive, daily seasonal naive, weekly seasonal naive, or drift — and what does this tell you about the structure of appliance energy use?

On a single 24h test, Weekly Seasonal Naive looked best (RMSE 292.8), but that
result depended heavily on which hour training happened to end on — Naive and
Drift both collapsed because the final training hour was an unrepresentative
evening peak. The fairer 14-day rolling average tells a more honest story:
**Mean (391.2) and Weekly Seasonal Naive (404.9) are roughly tied as the
strongest**, Daily Seasonal Naive is clearly worse (455.8), and Naive/Drift
are far worse (584–586).

This is informative: appliance use *does* have a repeating average daily and
weekly rhythm (confirmed in the Part 1 EDA), but the exact timing of usage on
any given day is noisy — "copy exactly what happened yesterday" is
unreliable, because it's driven by unpredictable occupant behaviour, not a
clean mechanical cycle. The very poor showing of Naive/Drift (which anchor on
the single last observed value) also confirms the Part 1 stationarity
finding: the series is mean-reverting with no persistent trend, so the last
value carries almost no information about the future.

### Q2. Does the SARIMAX model improve on the strongest seasonal benchmark? Discuss whether daily seasonality, autocorrelation, and exogenous variables are adequately captured.

Yes, clearly. SARIMAX(1,0,6)(1,1,1,24) achieves RMSE 328 on the 14-day
rolling average versus ~391 for the best benchmark (a ~16% improvement), and
159.6 vs 292.8 on the single 24h test (~45% improvement). The residual
diagnostics (ACF plot) show no significant leftover autocorrelation, meaning
the seasonal AR/MA structure captures the daily cycle and short-run
autocorrelation adequately. No exogenous variables were included in this
SARIMAX (that was deliberately left to the feature-based model in Part 6);
given how little the weather/indoor-sensor features helped XGBoost (see Q3),
it's plausible an exogenous SARIMAX would not have improved much further —
though this wasn't directly tested and is a reasonable extension.

One real limitation: the 95% confidence intervals dip below zero at points,
which is physically impossible for energy use — a known weakness of Gaussian
SARIMAX intervals applied to skewed, non-negative data.

### Q3. Does the XGBoost/feature-based model improve when lag, rolling-window, time-of-day, and sensor/weather variables are added? Which feature groups appear most useful?

Feature importance rankings show **time-of-day (hour sin/cos) and
day-of-week (dow sin/cos), plus the 24h/168h lag and rolling-mean features,
dominate** — these account for the majority of importance in the
"true_forecast" model. Adding the actual future weather and indoor sensor
readings (the "conditional" model) did **not** improve the robust rolling
score (379 vs 373 RMSE — very slightly worse), despite using real future
information the honest model didn't have access to. This suggests that, for
this specific house, weather/indoor sensor readings carry little independent
predictive power for appliance use beyond what time-of-day and recent/seasonal
lags already capture — consistent with the original dataset paper
(Candanedo et al., 2017), which found weather contributed relatively little
to predicting this specific target.

Neither XGBoost variant beat SARIMAX on the robust evaluation (373–379 vs
328 RMSE). Two plausible reasons: SARIMAX's seasonal ARIMA structure may
simply suit this kind of repeating-but-noisy series better than a generic
tree-based regressor, and/or the training set (~115 days) may be too small
for a single global XGBoost model — covering all 24 forecast horizons at
once — to learn as robustly as a model purpose-built for seasonal time series.

### Q4. Does the foundation model outperform the simpler benchmark, SARIMAX, and feature-based models? Is the improvement, if any, large enough to justify the extra complexity?

**No.** Chronos (`amazon/chronos-t5-small`, zero-shot, no training) scored
**RMSE=305.2, MAE=215.1, MAPE=26.4%** on the single 24h test — worse than
SARIMAX (159.6), both XGBoost variants (231–239), *and* even the simple
Weekly Seasonal Naive benchmark (292.8) and Mean benchmark (300.0). It only
beats Daily Seasonal Naive, Naive, and Drift.

Looking at the forecast plot, Chronos produced a smooth, low-variance median
forecast that tracked the general overnight-low / evening-rise shape but
substantially **under-predicted the magnitude of both sharp peaks** in the
test window — the actual series exceeds even the upper bound of its own 90%
prediction interval near the end of the day. This is a sensible failure mode
for a *general-purpose* pretrained model: Chronos was trained on a huge,
diverse corpus of public time series and has never seen this specific
house's occupant behaviour, so it defaults to a cautious, averaged-out
prediction rather than the sharp, bursty spikes that are specific to this
data (and which SARIMAX and XGBoost, both fit directly on this house's
history, learned to anticipate).

**Conclusion**: for this task — a single, well-established time series with
4+ months of clean history — the extra complexity (and setup friction: this
required Google Colab since the sandbox blocked HuggingFace, plus a ~185MB
model download) is not justified. A zero-shot foundation model becomes more
attractive in scenarios this project doesn't actually face: forecasting
brand-new households with little to no history, or forecasting many series
at once without wanting to fit/maintain a separate model per series. Only a
single 24h evaluation was run for Chronos (not the full 14-day rolling
check used for the other models), due to the practical friction of running
it outside the main pipeline — a natural extension noted in the report's
limitations section.

### Q5. Which variables would genuinely be known at the forecast origin? If you use future indoor temperature, humidity, or weather values from the test set, is this a true forecast or a conditional forecast?

Genuinely knowable at the forecast origin: all past/lagged Appliances values
up to that point, and any purely calendar-based feature (hour-of-day,
day-of-week) for future timestamps, since calendars are known in advance.
**Not** genuinely knowable: actual future indoor temperature/humidity
(depends on future weather, heating/cooling decisions, and occupancy) or
actual future outdoor weather (temperature, humidity, wind, pressure,
visibility) — in a real deployment these would have to come from an external
weather *forecast* (itself imperfect), not the ground-truth values.

This means our "conditional" XGBoost model, which used the real realized
future weather/indoor readings from the test set, is **not a true forecast**
— it's a conditional/hindsight estimate ("what would appliance use have been,
*given* we already knew the exact future conditions"), which overstates
what's actually achievable in production. Interestingly, per Q3, this
hindsight information didn't even help — but the leakage risk is real and
must be flagged any time exogenous variables are used without first
forecasting those variables themselves.

### Q6. Based on accuracy, interpretability, uncertainty, computational cost, and ease of deployment, which model would you recommend for practical smart-home energy forecasting, and why?

**SARIMAX(1,0,6)(1,1,1,24)** is the recommendation from current evidence:

- **Accuracy**: best of everything tested (RMSE 328 rolling, 159.6 single-shot) — including beating the Chronos foundation model (305.2), which actually finished behind two of the simple benchmarks
- **Interpretability**: moderate — AR/MA coefficients are less intuitive
  than tree feature importances, but the seasonal structure is conceptually
  clear and residual diagnostics are easy to interpret
- **Uncertainty**: provides native confidence intervals (with the
  below-zero caveat noted in Q2); Chronos also gives prediction intervals
  natively, but its interval on the test window did not even contain the
  actual peak values, which is a bigger practical uncertainty failure
- **Computational cost**: SARIMAX's weak point — the grid search took a
  long time (147 fits) and each new forecast requires a refit (~50s in our
  tests); manageable for one household forecasting once a day, but doesn't
  scale cheaply to many households at once. Chronos needs no training at
  all, but does need a one-time ~185MB model download and GPU/CPU inference
  capacity
- **Ease of deployment**: straightforward, stable library, no GPU or
  complex feature pipeline required — unlike XGBoost, which needs a
  lag/rolling-window feature pipeline maintained at inference time, or
  Chronos, which needs a deep learning runtime (PyTorch) and, in our case,
  couldn't even run inside the assignment's own compute environment due to
  a network restriction on HuggingFace

If deploying across many homes simultaneously (rather than one well-established
house with 4+ months of history), the calculus would likely flip: a single
XGBoost model amortizes its training cost across many households, and a
zero-shot foundation model becomes attractive for brand-new homes with no
history to fit a per-home SARIMAX on yet — though based on this project's
results, its out-of-the-box accuracy would need to be weighed carefully
against that convenience.
