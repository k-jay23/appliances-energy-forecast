const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, BorderStyle, ShadingType, AlignmentType, ImageRun, PageBreak,
  Header, Footer, PageNumber, LevelFormat, convertInchesToTwip
} = require("docx");

const FIG = (name) => fs.readFileSync(`${__dirname}/figs/${name}`);

// ---------- helpers ----------
function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 220, after: 90 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 160, after: 80 } });
}
function p(text, opts = {}) {
  return new Paragraph({
    children: Array.isArray(text) ? text : [new TextRun(text)],
    spacing: { after: 120, line: 264 },
    ...opts,
  });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60, line: 264 } });
}
function bold(text) { return new TextRun({ text, bold: true }); }
function ital(text) { return new TextRun({ text, italics: true }); }

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 1500, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "2B6CB0" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text: String(text), bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000", size: opts.size || 19 })],
    })],
  });
}

function dataTable(headers, rows, widths) {
  const colWidths = widths || headers.map(() => Math.floor(9360 / headers.length));
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({ children: headers.map((hd, i) => cell(hd, { header: true, width: colWidths[i], center: true })), tableHeader: true }),
      ...rows.map((r) => new TableRow({ children: r.map((v, i) => cell(v, { width: colWidths[i], center: i > 0 })) })),
    ],
  });
}

function figure(name, widthPx, heightPx, caption) {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: FIG(name), transformation: { width: widthPx, height: heightPx } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: caption, italics: true, size: 18 })],
    }),
  ];
}

// ---------- content ----------
const children = [];

// Title block
children.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
    children: [new TextRun({ text: "Modelling and Forecasting Household Appliance Energy Use", bold: true, size: 32 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 40 },
    children: [new TextRun({ text: "A Case Study Comparing Benchmark, Statistical, Machine-Learning, and Foundation-Model Forecasting Approaches", size: 24, italics: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 220 },
    children: [new TextRun({ text: "Data Science Analysis Techniques and AI — Assignment Report", size: 20, color: "555555" })],
  })
);

// 1. Introduction
children.push(h1("1. Introduction"));
children.push(p(
  "This report works through a fairly practical question: can appliance electricity use in a home be forecast 24 hours ahead, and does it actually pay off to use something more sophisticated than a simple rule of thumb? Four approaches were built and compared under the same test conditions: basic statistical benchmarks, a seasonal ARIMA model (SARIMAX), a gradient-boosted tree model (XGBoost) with engineered features, and a pretrained foundation model (Chronos) that needs no training at all. The results turned out to be more interesting than expected going in - the simplest classical model beat the more elaborate ones, which is worth digging into rather than glossing over."
));
children.push(p(
  "The data comes from the UCI \u201cAppliances Energy Prediction\u201d dataset (Candanedo, Feldheim & Deramaix, 2017): 10-minute sensor readings from a low-energy house in Belgium, covering indoor temperature/humidity across nine rooms plus outdoor weather, over about 4.5 months. The rest of the report follows the structure of the assignment - data preparation, stationarity testing, defining the forecasting problem, the four modelling approaches in turn, a consolidated comparison, the six set discussion questions, limitations, and a final recommendation."
));

// 2. Data & Preprocessing
children.push(h1("2. Data and Preprocessing"));
children.push(p(
  "The raw file has 19,735 readings taken every 10 minutes between 11 January and 27 May 2016, across 28 columns. The target is Appliances (Wh); there is also a secondary lights load, nine pairs of indoor temperature/humidity sensors (T1\u2013T9, RH_1\u2013RH_9), six outdoor weather variables (T_out, RH_out, Press_mm_hg, Windspeed, Visibility, Tdewpoint), and two columns of pure random noise (rv1, rv2) that the original authors added deliberately, as a check on whether feature-selection methods would correctly ignore them. rv1/rv2 were dropped from every model built here for the same reason - they are noise by construction, so there is nothing real to learn from them."
));
children.push(p(
  "Missing values were not an issue at all, and neither were gaps in the timestamps - this dataset is unusually clean. For the actual modelling the series was resampled from 10-minute to hourly resolution (3,290 rows), summing the two energy columns since Wh is additive, and averaging everything else since those are instantaneous sensor readings rather than accumulating quantities."
));
children.push(h2("Exploratory findings"));
children.push(p(
  "The hourly series has an obvious daily shape: quiet overnight, climbing from around 7am, peaking near 18:00, with a smaller weekly pattern layered on top (Mondays, Fridays and Saturdays run a bit higher than mid-week). An STL decomposition backs this up with a moderate seasonal strength (\u2248 0.32 on a 0\u20131 scale) - moderate rather than strong, because the residual is genuinely noisy. Appliance use is bursty by nature, not a clean repeating wave."
));
children.push(...figure("02_daily_profile.png", 400, 200, "Figure 1. Average hourly appliance energy use by hour of day (\u00b11 std. dev.)."));

// 3. Stationarity
children.push(h1("3. Stationarity Analysis"));
children.push(p(
  "Running an Augmented Dickey-Fuller test on the raw hourly series gives a clear result: no unit root (ADF statistic = \u22129.13, p \u2248 3.1\u00d710\u207b\u00b9\u2075), so the series is already stationary in the classical sense - it does not drift, it just oscillates around a fairly steady level. First-order and 24-hour seasonal differencing were tested too and stay stationary, meaning differencing is not technically required just to satisfy the ADF criterion."
));
children.push(p(
  "That is a bit of a trap though, because ADF only checks for a unit root, not for seasonality. Looking at the ACF/PACF, there is clear correlation at lag 24 (and its multiples, 48 and 72), and once the series is seasonally differenced at 24h, a sharp negative spike shows up at lag 24 in both plots - the classic signature of a seasonal AR(1)/MA(1) term at s=24. That evidence fed directly into the SARIMAX specification used in Section 6."
));
children.push(...figure("07_acf_pacf_seasonal_diff.png", 440, 264, "Figure 2. ACF and PACF after 24-hour seasonal differencing, showing the significant lag-24 spike."));

// 4. Problem definition
children.push(h1("4. Forecasting Problem Definition"));
children.push(p([bold("Target variable: "), new TextRun("Appliances \u2014 hourly appliance energy use (Wh).")]));
children.push(p([bold("Forecast horizon: "), new TextRun("24 hours ahead.")]));
children.push(p([bold("Train/test split: "), new TextRun("the final 14 days (336 hours) are held out as a test set never seen during training; the preceding 2,954 hours form the training set.")]));
children.push(p([bold("Evaluation design: "), new TextRun(
  "every model is scored two ways. First, a single 24-hour forecast made right at the end of training - this is literally what the brief asks for. Second, a more thorough 14-day rolling check, where that same 24-hour forecast is repeated across all 14 held-out days and averaged. The second check was added after the single-day benchmark numbers (Section 5) turned out to be a bit misleading - one day is not really a reliable enough sample to judge a model on."
)]));
children.push(p([bold("Metrics: "), new TextRun("RMSE (primary), MAE, and MAPE.")]));

// 5. Benchmarks
children.push(h1("5. Benchmark Models"));
children.push(p(
  "Five standard baseline forecasts were implemented: Mean (the overall training average), Naive (repeat the last observed value), Daily Seasonal Naive (copy the same hour yesterday), Weekly Seasonal Naive (copy the same hour, same day, last week), and Drift (naive plus a straight-line trend extrapolated across the whole training set)."
));
children.push(dataTable(
  ["Model", "RMSE (single)", "MAE (single)", "RMSE (rolling)", "MAE (rolling)"],
  [
    ["Mean", "300.0", "236.3", "391.2", "296.1"],
    ["Naive", "1485.6", "1455.8", "584.7", "512.0"],
    ["Daily Seasonal Naive", "708.3", "481.7", "455.8", "288.6"],
    ["Weekly Seasonal Naive", "292.8", "180.0", "404.9", "254.6"],
    ["Drift", "1492.5", "1463.3", "586.1", "513.6"],
  ],
  [2400, 1740, 1740, 1740, 1740]
));
children.push(p(
  "The single-day test made Weekly Seasonal Naive look best, while Naive and Drift basically fell apart - both anchor on the final training value, and that value happened to sit right at an evening peak, so they predicted \u201chigh, forever.\u201d Not entirely fair on them. The rolling 14-day check is a lot more balanced: Mean and Weekly Seasonal Naive end up roughly tied for strongest among the simple benchmarks, Daily Seasonal Naive lags behind both despite the clear average shape seen in Figure 1, and Naive/Drift stay well behind everything else. Put together, this says appliance use has a genuine average daily/weekly rhythm but no persistent short-term momentum (consistent with the stationarity result in Section 3), and that the exact timing of usage on any given day is too noisy for \u201cjust copy yesterday\u201d to hold up reliably.",
  { spacing: { after: 140, line: 264 } }
));

// 6. SARIMAX
children.push(h1("6. SARIMAX Model"));
children.push(p(
  "A SARIMA(p,d,q)(P,D,Q,24) model was fitted using statsmodels. Per the assignment spec, the non-seasonal orders were grid-searched exhaustively (p 0\u20136, d 0\u20132, q 0\u20136, 147 combinations total), selecting by AIC. The seasonal part was fixed at (1,1,1,24) rather than also searched: Section 3 already gave a fairly clean seasonal signature at lag 24, and searching the seasonal terms as well would have pushed the search space well past what was practical to run (some of the higher-order fits already took close to 40 seconds each on the hardware available)."
));
children.push(p(
  "To keep 147 fits actually finishable, the search was run on just the last 45 days of training data rather than the full 2,954 hours, with the winning order then refit on the complete training set for the real forecast. The winner by AIC was SARIMA(1,0,6)(1,1,1,24) (AIC = 14872.0 on the search subset), which converged cleanly. Worth flagging: 90 of the 147 combinations did not fully converge within the reduced iteration cap used to keep the search fast - the winning model was not one of them, but a slower, fuller search might turn up something marginally better."
));
children.push(p(
  "Refitting on the full training set, the residual ACF (Figure 3, left) shows nothing significant left over, suggesting the seasonal/AR structure is doing its job capturing the daily cycle and short-run dependence. The residual distribution (right) is right-skewed (skew = 1.84) with heavy tails (kurtosis = 8.16) and a near-zero mean (\u22122.5 Wh), which fits the bursty nature of the data - the model tracks the general level well but tends to under-predict the occasional big spike."
));
children.push(...figure("10_sarimax_residual_diagnostics.png", 540, 180, "Figure 3. SARIMAX residual ACF (left) and residual distribution (right)."));
children.push(dataTable(
  ["Evaluation", "RMSE", "MAE", "MAPE %"],
  [["Single 24h forecast", "159.6", "112.7", "19.5"], ["14-day rolling average", "328.1", "214.3", "33.2"]],
  [3960, 1800, 1800, 1800]
));
children.push(p(
  "SARIMAX beats the best benchmark clearly on both evaluations - about a 16% RMSE improvement on the rolling check (328.1 vs. 391.2) and an even bigger gap on the single-day test. One honest limitation worth naming: the 95% interval dips below zero in places, which obviously cannot happen for real energy use - a known issue with Gaussian confidence intervals applied to skewed, non-negative data.",
  { spacing: { after: 140, line: 264 } }
));

// 7. Feature-based ML
children.push(h1("7. Feature-Based Machine Learning Model: XGBoost"));
children.push(p(
  "For the feature-based model, a \u201cdirect multi-horizon\u201d design was used: one training row per (origin, hours-ahead) pair, covering hours-ahead = 1\u201324, so a single model can predict any point across the next day rather than needing 24 separate models. The features are restricted to what would genuinely be known standing at the forecast origin: cyclically-encoded hour-of-day and day-of-week for the target time, a weekend flag, lagged Appliances values (1h/24h/168h relative to the origin), rolling mean and std (24h/168h), plus lags measured relative to the specific target hour itself (24h/168h before it, which stays valid across the whole 24-hour horizon)."
));
children.push(p(
  "Two versions of this feature set were built to directly test what exogenous covariates actually add. The \u201ctrue-forecast\u201d version (17 features) only uses what is described above. The \u201cconditional\u201d version (41 features) also includes the real indoor (T1\u2013T9, RH_1\u2013RH_9) and outdoor weather readings at the target time - which is something of a cheat, since a live system would not actually know tomorrow\u2019s exact indoor temperature (more on this under Question 5). Both were trained as XGBoost regressors (400 trees, depth 5, learning rate 0.05) on 66,288 rows, using training-period data only."
));
children.push(dataTable(
  ["Model variant", "RMSE (single)", "MAE (single)", "RMSE (rolling)", "MAE (rolling)"],
  [["True forecast (17 features)", "238.9", "172.9", "373.0", "240.9"], ["Conditional (41 features)", "231.3", "187.7", "379.3", "284.5"]],
  [3160, 1550, 1550, 1550, 1550]
));
children.push(p(
  "Feature importance (Figure 4) shows the honest model leans almost entirely on the cyclic day-of-week/hour-of-day encodings plus the 24h/168h lag and rolling-mean features - seasonal and recency structure dominate everything else. What is a little surprising is that the conditional model, even with access to the real future weather and indoor readings, did not actually do better on the rolling check (379.3 vs. 373.0 - if anything, slightly worse). That lines up with the original dataset paper\u2019s own finding: weather does not carry much independent signal for appliance-specific energy use, as opposed to heating load. Neither XGBoost variant beat SARIMAX on the fair rolling comparison either (373\u2013379 vs. 328), despite being considerably more complex to build - discussed further in Section 10."
));
children.push(...figure("13_xgb_true_forecast_feature_importance.png", 400, 200, "Figure 4. Top 15 feature importances, XGBoost (true-forecast variant)."));

// 8. Foundation model
children.push(h1("8. Foundation Model: Chronos"));
children.push(p(
  "Chronos (amazon/chronos-t5-small; Ansari et al., 2024) is a pretrained transformer built for zero-shot forecasting: no training on the target series required, just a window of recent history (\u201ccontext\u201d) fed in directly. The model was given the last 336 hours (14 days) of training data as context and asked for a 24-hour probabilistic forecast (100 sampled trajectories, median taken as the point forecast, 5th\u201395th percentile as a 90% interval)."
));
children.push(p(
  "This one could not run in the same place as everything else - the sandbox used for the rest of this project blocks the HuggingFace domain the pretrained weights are hosted on, so it had to be run separately in Google Colab. Because of that extra friction, only the single 24-hour evaluation was carried out, not the 14-day rolling check used for the other models."
));
children.push(dataTable(["Evaluation", "RMSE", "MAE", "MAPE %"], [["Single 24h forecast", "305.2", "215.1", "26.4"]], [3960, 1800, 1800, 1800]));
children.push(...figure("14_chronos_forecast.png", 440, 198, "Figure 5. Chronos 24-hour forecast (median and 90% interval) vs. actual."));
children.push(p(
  "Figure 5 shows what happened: Chronos produces a smooth, cautious forecast that gets the broad overnight-low/evening-rise shape roughly right but badly underestimates the size of both sharp peaks - actual usage goes above even the top of its own 90% interval late in the day. That is a fairly understandable failure mode for a general-purpose zero-shot model that has never seen this specific house before; it defaults to a safe, averaged-out guess rather than committing to a spike. Accuracy-wise it lands close to the Mean and Weekly Seasonal Naive benchmarks, clearly behind both SARIMAX and XGBoost.",
  { spacing: { after: 140, line: 264 } }
));

// 9. Comparison
children.push(h1("9. Consolidated Model Comparison"));
children.push(p("Table 3 and Figure 6 put every model on the same footing so they can be compared directly."));
children.push(dataTable(
  ["Model", "RMSE (single)", "RMSE (rolling)", "MAE (rolling)", "MAPE % (rolling)"],
  [
    ["SARIMAX(1,0,6)(1,1,1,24)", "159.6", "328.1", "214.3", "33.2"],
    ["XGBoost (true forecast)", "238.9", "373.0", "240.9", "38.5"],
    ["XGBoost (conditional)", "231.3", "379.3", "284.5", "53.8"],
    ["Mean", "300.0", "391.2", "296.1", "53.6"],
    ["Weekly Seasonal Naive", "292.8", "404.9", "254.6", "37.1"],
    ["Chronos (foundation model)", "305.2", "n/a*", "n/a*", "n/a*"],
    ["Daily Seasonal Naive", "708.3", "455.8", "288.6", "43.8"],
    ["Naive", "1485.6", "584.7", "512.0", "113.4"],
    ["Drift", "1492.5", "586.1", "513.6", "113.8"],
  ],
  [3060, 1575, 1575, 1575, 1575]
));
children.push(p("*Chronos was evaluated only on the single 24h test (see Section 8 and Limitations).", { children: [ital("*Chronos was evaluated only on the single 24h test (see Section 8 and Limitations).")], spacing: { after: 200, line: 260 } }));
children.push(...figure("15_all_models_comparison.png", 530, 265, "Figure 6. Actual vs. forecast for the best benchmark, SARIMAX, and XGBoost, on the single 24h test window."));
children.push(p(
  "SARIMAX wins outright on both evaluation regimes. XGBoost, despite being considerably more complex to build, does not manage to beat it here, and the foundation model - despite needing no training at all - only really keeps pace with the simplest benchmarks."
));

// 10. Discussion
children.push(h1("10. Critical Discussion"));

children.push(h2("Q1. Which benchmark model is strongest, and what does this tell you about the structure of appliance energy use?"));
children.push(p(
  "The single-day test made Weekly Seasonal Naive look best, but that was mostly luck of the draw - training happened to end right on an unrepresentative hour, which is exactly why Naive and Drift cratered. Once the rolling check is used instead, Mean and Weekly Seasonal Naive come out roughly tied for strongest, with Daily Seasonal Naive clearly behind both. What this says: appliance use has a genuine average daily/weekly rhythm, but the exact timing on any specific day is too noisy for \u201ccopy yesterday exactly\u201d to be dependable, and the very poor showing of Naive/Drift confirms there is no real short-term momentum in the series either (ties back to Section 3)."
));

children.push(h2("Q2. Does SARIMAX improve on the strongest benchmark? Are daily seasonality, autocorrelation, and exogenous variables adequately captured?"));
children.push(p(
  "Yes, by a decent margin - about 16% lower RMSE on the rolling check (328.1 vs. 391.2), and an even bigger gap on the single-day test. The residual ACF came back clean, no significant autocorrelation left over, so the seasonal AR/MA structure is capturing daily seasonality and short-run dependence reasonably well. No exogenous variables were included in this SARIMAX - that was deliberately left for the feature-based model instead. Given how little the weather/indoor sensor data ended up helping XGBoost (see Q3), an exogenous SARIMAX probably would not add a huge amount either, though this was not directly tested."
));

children.push(h2("Q3. Does XGBoost improve when lag, rolling-window, time-of-day, and sensor/weather variables are added? Which feature groups are most useful?"));
children.push(p(
  "The feature importance plot makes this fairly clear: time-of-day and day-of-week encodings, plus the 24h/168h lag and rolling-mean features, dominate everything else by a wide margin. Adding the real future weather/indoor sensor readings (the conditional variant) did not improve the rolling score at all - it actually came in slightly worse (379.3 vs. 373.0), even with information the honest model never had access to. So weather/indoor readings do not seem to carry much independent value here beyond what time-of-day and lag structure already capture. Neither version of XGBoost beat SARIMAX on the fair comparison either, which suggests SARIMAX\u2019s seasonal structure is simply a better fit for this kind of series than a generic tree-based regressor, and/or that \u2248115 days of training history is not quite enough for one global model to learn all 24 forecast horizons well at once."
));

children.push(h2("Q4. Does the foundation model outperform the simpler benchmark, SARIMAX, and feature-based models? Is any improvement large enough to justify the extra complexity?"));
children.push(p(
  "No, not close. Chronos scored 305.2 RMSE on the single 24h test - worse than SARIMAX (159.6) and both XGBoost variants (231\u2013239), and only about level with the plain Mean benchmark (300.0). Looking at the forecast plot, it visibly missed the sharp evening peak and defaulted to a smooth, cautious prediction, which makes sense for a model that has never seen this particular house\u2019s history before. For a single, well-established series with 4+ months of clean data behind it, the extra complexity - and, in this project, the extra hassle of needing a separate Colab environment just to run it - is not worth it on accuracy grounds. Where a foundation model like this would actually earn its keep is a cold-start situation: a brand-new home with no history yet to fit a per-home SARIMAX or XGBoost on, where getting an instant forecast without any training matters more than squeezing out the best possible accuracy."
));

children.push(h2("Q5. Which variables are genuinely known at the forecast origin? Is using future weather/indoor values from the test set a true or conditional forecast?"));
children.push(p(
  "Genuinely knowable at the forecast origin: all past Appliances values, and anything purely calendar-based (hour-of-day, day-of-week), since the calendar is always known in advance. Not genuinely knowable: the actual future indoor temperature/humidity (depends on future weather, heating decisions, and occupancy) or the actual future outdoor weather - in a real system this would require an external weather forecast, which comes with its own error, rather than the ground-truth values used here from the test set. So the \u201cconditional\u201d XGBoost model, which used the real realised future readings, is not a genuine forecast - it is closer to a hindsight estimate that overstates what could actually be achieved live. Interestingly (see Q3), that hindsight information did not even help in this case, but the leakage risk is real regardless, and it is worth flagging any time exogenous variables are used without also forecasting those variables themselves."
));

children.push(h2("Q6. Considering accuracy, interpretability, uncertainty, computational cost, and ease of deployment, which model would you recommend, and why?"));
children.push(p(
  "SARIMAX(1,0,6)(1,1,1,24) is the recommendation here. It is the most accurate model across both evaluations by a clear margin, it provides native confidence intervals (with the below-zero caveat noted in Section 6), the seasonal structure and clean residual diagnostics make it reasonably interpretable, and it deploys with a stable, well-documented library without needing a complicated feature pipeline. Its real weakness is computational cost - the grid search and each rolling refit (\u224850 seconds per fit) add up, though that is entirely manageable for forecasting a single household once a day. If the goal changed to forecasting many households at once, or households with barely any history, the recommendation would likely shift: a single XGBoost model can be trained once and reused across many series, and a zero-shot foundation model becomes genuinely useful exactly where there is no per-household history yet to fit anything else on."
));

// 11. Limitations
children.push(h1("11. Limitations and Future Work"));
children.push(bullet("The SARIMAX grid search ran on a reduced 45-day window with a capped iteration limit, just to keep runtime manageable; 90 of the 147 combinations did not fully converge as a result (the winning model was unaffected, but a slower, fuller search might turn up something marginally better)."));
children.push(bullet("An exogenous SARIMAX variant was not tested; combining the seasonal ARIMA structure with a few selected covariates would be a natural next step."));
children.push(bullet("The XGBoost model is a single global model covering all 24 forecast horizons at once (the \u201cdirect\u201d approach) rather than 24 separate specialised models; a per-horizon ensemble might do better closer to the forecast origin."));
children.push(bullet("Chronos was only evaluated on the single 24-hour window, not the full rolling check the other models got, and only the \u201csmall\u201d model size was tried, not \u201cbase\u201d/\u201clarge\u201d or a fine-tuned version."));
children.push(bullet("This is all based on one house over one 4.5-month stretch, so findings such as the limited value of weather features might not hold up in a different climate, season, or household."));
children.push(p(
  "Some obvious next steps: an exogenous SARIMAX, a per-horizon or recursive XGBoost setup, a larger or fine-tuned Chronos model, a bounded/log-transformed model to fix the negative-interval issue, and validation across more than one household.",
  { spacing: { after: 140, line: 264 } }
));

// 12. Conclusion
children.push(h1("12. Conclusion and Recommendation"));
children.push(p(
  "Across benchmarks, SARIMAX, XGBoost, and Chronos, SARIMAX(1,0,6)(1,1,1,24) came out as the most accurate model for forecasting 24 hours of household appliance energy use, beating the best benchmark by \u224816% RMSE and outperforming a considerably more complex XGBoost setup. The zero-shot Chronos model, while requiring no training at all, only kept pace with the simplest benchmarks and seems better suited to genuine cold-start scenarios than to a data-rich single series like this one. For this specific task - one well-established household with several clean months of history behind it - SARIMAX is the model recommended here: the most accurate option on the evidence gathered, reasonably interpretable, and straightforward enough to deploy without a complicated pipeline."
));

// References
children.push(h1("References"));
const refs = [
  "Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Arango, S. P., Kapoor, S., Zschiegner, J., Maddix, D. C., Mahoney, M. W., Torkkola, K., Wilson, A. G., Bohlke-Schneider, M. & Wang, Y. (2024). Chronos: Learning the Language of Time Series. arXiv:2403.07815.",
  "Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung, G. M. (2015). Time Series Analysis: Forecasting and Control (5th ed.). Wiley.",
  "Candanedo, L. M., Feldheim, V. & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. Energy and Buildings, 140, 81\u201397.",
  "Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785\u2013794.",
  "Hyndman, R. J. & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.",
];
refs.forEach((r) => children.push(new Paragraph({
  spacing: { after: 100, line: 250 },
  indent: { left: 360, hanging: 360 },
  children: [new TextRun({ text: r, size: 19 })],
})));

// ---------- document ----------
const doc = new Document({
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 860, bottom: 860, left: 1170, right: 1170 },
      },
    },
    children,
  }],
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 21 } },
    },
    heading1: { run: { size: 26, bold: true, color: "1A365D" } },
    heading2: { run: { size: 22, bold: true, color: "2B6CB0" } },
  },
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(`${__dirname}/appliances_energy_forecast_report.docx`, buf);
  console.log("Report written.");
});
