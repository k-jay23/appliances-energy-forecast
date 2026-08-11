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
  "This report presents a case study in time-series forecasting: modelling household appliance electricity demand and producing a 24-hour-ahead forecast using a range of methods of increasing complexity. Four families of model are built and compared under identical evaluation conditions: naive statistical benchmarks, a seasonal autoregressive model (SARIMAX), a feature-based gradient-boosted tree model (XGBoost), and a pretrained time-series foundation model (Chronos). The aim is not only to find the most accurate model, but to critically examine whether additional model complexity is actually justified by the data."
));
children.push(p(
  "The dataset is the UCI \u201cAppliances Energy Prediction\u201d dataset (Candanedo, Feldheim & Deramaix, 2017): 10-minute readings of appliance energy use, together with indoor temperature/humidity from nine rooms and outdoor weather, collected over 4.5 months in a low-energy house in Belgium. The report follows the assignment structure: data preparation and exploratory analysis, stationarity testing, problem definition, benchmark modelling, SARIMAX, feature engineering and machine learning, a foundation model comparison, consolidated evaluation, critical discussion, limitations, and a final recommendation."
));

// 2. Data & Preprocessing
children.push(h1("2. Data and Preprocessing"));
children.push(p(
  "The raw dataset contains 19,735 readings at 10-minute resolution (11 Jan \u2013 27 May 2016) across 28 columns: the target variable Appliances (Wh), a secondary lights load, nine indoor temperature/humidity sensor pairs (T1\u2013T9, RH_1\u2013RH_9), six outdoor weather variables (T_out, RH_out, Press_mm_hg, Windspeed, Visibility, Tdewpoint), and two synthetic random-noise columns (rv1, rv2) that the original authors deliberately included as a control to test whether feature-selection methods correctly ignore uninformative variables. rv1/rv2 are excluded from every model built in this report, as they carry no real information by construction."
));
children.push(p(
  "The data required no missing-value treatment and had no irregular time gaps. For modelling, the series was resampled from 10-minute to hourly resolution (3,290 rows): the two energy columns (Appliances, lights) were summed, since Wh is an additive quantity, while all other sensor columns were averaged, since they are instantaneous readings."
));
children.push(h2("Exploratory findings"));
children.push(p(
  "Hourly appliance use follows a clear daily rhythm \u2014 low overnight, rising from around 7am, and peaking near 18:00 \u2014 with a milder weekly pattern (Mondays, Fridays and Saturdays somewhat higher than mid-week). An STL decomposition (24-hour period) confirms a moderate seasonal component (seasonal strength \u2248 0.32 on a 0\u20131 scale); the residual is noisy and spiky, reflecting how bursty individual appliance usage is rather than a clean repeating signal."
));
children.push(...figure("02_daily_profile.png", 400, 200, "Figure 1. Average hourly appliance energy use by hour of day (\u00b11 std. dev.)."));

// 3. Stationarity
children.push(h1("3. Stationarity Analysis"));
children.push(p(
  "An Augmented Dickey-Fuller (ADF) test on the raw hourly series rejects the presence of a unit root decisively (ADF statistic = \u22129.13, p \u2248 3.1\u00d710\u207b\u00b9\u2075), meaning the series is already stationary in the classical sense: it is bounded and mean-reverting rather than drifting. First-order and 24-hour seasonal differencing were also tested and remain stationary, so no differencing is strictly required to satisfy the ADF criterion."
));
children.push(p(
  "This is an important nuance: the ADF test checks only for a unit root, not for seasonality. The ACF/PACF of the raw series show significant correlation at lag 24 (and its multiples, 48 and 72), and after 24-hour seasonal differencing there is a sharp, significant negative spike at lag 24 in both the ACF and PACF \u2014 the textbook signature of a seasonal AR(1)/MA(1) structure at s=24. This evidence directly informed the seasonal order used for SARIMAX in Section 6."
));
children.push(...figure("07_acf_pacf_seasonal_diff.png", 440, 264, "Figure 2. ACF and PACF after 24-hour seasonal differencing, showing the significant lag-24 spike."));

// 4. Problem definition
children.push(h1("4. Forecasting Problem Definition"));
children.push(p([bold("Target variable: "), new TextRun("Appliances \u2014 hourly appliance energy use (Wh).")]));
children.push(p([bold("Forecast horizon: "), new TextRun("24 hours ahead.")]));
children.push(p([bold("Train/test split: "), new TextRun("the final 14 days (336 hours) are held out as a test set never seen during training; the preceding 2,954 hours form the training set.")]));
children.push(p([bold("Evaluation design: "), new TextRun(
  "every model is scored two ways \u2014 (i) a single 24-hour-ahead forecast made immediately at the end of training (the literal requirement in the assignment brief), and (ii) a more robust 14-day rolling evaluation, in which the 24-hour forecast is repeated across all 14 held-out days (refitting/re-forecasting with an expanding history each time) and the resulting errors averaged. The rolling check was added because a single day can be an unrepresentative sample \u2014 this was confirmed directly in Section 5, where the benchmark ranking changed materially between the two evaluation regimes."
)]));
children.push(p([bold("Metrics: "), new TextRun("RMSE (primary), MAE, and MAPE.")]));

// 5. Benchmarks
children.push(h1("5. Benchmark Models"));
children.push(p(
  "Five standard benchmark forecasts were implemented: Mean (overall training average), Naive (last observed value repeated), Daily Seasonal Naive (same hour yesterday), Weekly Seasonal Naive (same hour, same day last week), and Drift (naive plus a linear trend extrapolated across the training set)."
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
  "On the single-day test, Weekly Seasonal Naive appears strongest, while Naive and Drift collapse \u2014 both anchor on the final training value, which happened to be an evening peak, so they predict \u201chigh forever.\u201d The rolling 14-day check tells a fairer story: Mean and Weekly Seasonal Naive are roughly tied as the strongest simple benchmarks, Daily Seasonal Naive under-performs both despite the clear average daily shape seen in Figure 1, and Naive/Drift remain far worse. This indicates the series has a real average daily/weekly rhythm but no persistent short-term momentum \u2014 consistent with the stationarity finding in Section 3 \u2014 and that the exact timing of usage on any given day is too noisy for \u201ccopy yesterday exactly\u201d to be reliable.",
  { spacing: { after: 140, line: 264 } }
));

// 6. SARIMAX
children.push(h1("6. SARIMAX Model"));
children.push(p(
  "A SARIMA(p,d,q)(P,D,Q,24) model was fitted using statsmodels. Following the assignment specification, the non-seasonal orders were grid-searched exhaustively over p\u2208[0,6], d\u2208[0,2], q\u2208[0,6] (147 combinations), selecting by AIC. The seasonal order was fixed at (1,1,1,24) rather than also grid-searched: the Section 3 diagnostics (a clean seasonal AR/MA signature at lag 24) justified this choice, and grid-searching seasonal terms as well would have multiplied an already large search space beyond what was computationally tractable on the available single-core hardware (individual high-order fits took up to \u224840 seconds)."
));
children.push(p(
  "To keep the full 147-model search tractable, it was run on the most recent 45 days of the training set rather than the full 2,954-hour history; the winning order was then refit on the complete training set for the actual forecast. The best model by AIC was SARIMA(1,0,6)(1,1,1,24) (AIC = 14872.0 on the search subset), which converged cleanly. It is worth noting as a limitation that 90 of the 147 grid combinations did not fully converge within the reduced iteration cap (50) used to keep the search fast \u2014 though the selected winner was not among them."
));
children.push(p(
  "Refit on the full training set, residual diagnostics show no significant remaining autocorrelation (Figure 3, left), indicating the seasonal/autoregressive structure adequately captures the daily cycle and short-run dependence. The residual distribution (Figure 3, right) is right-skewed (skew = 1.84) with heavy tails (kurtosis = 8.16) and a near-zero mean (\u22122.5 Wh), consistent with the bursty, spiky nature of appliance usage \u2014 the model tracks the general level well but under-predicts occasional large spikes."
));
children.push(...figure("10_sarimax_residual_diagnostics.png", 540, 180, "Figure 3. SARIMAX residual ACF (left) and residual distribution (right)."));
children.push(dataTable(
  ["Evaluation", "RMSE", "MAE", "MAPE %"],
  [["Single 24h forecast", "159.6", "112.7", "19.5"], ["14-day rolling average", "328.1", "214.3", "33.2"]],
  [3960, 1800, 1800, 1800]
));
children.push(p(
  "SARIMAX clearly improves on the strongest benchmark on both evaluation regimes \u2014 a \u224816% RMSE reduction on the robust rolling check (328.1 vs. 391.2) and a much larger margin on the single-day test. One genuine limitation: the 95% forecast interval dips below zero at points, which is physically impossible for energy use \u2014 a known weakness of Gaussian-based SARIMAX intervals applied to skewed, non-negative data.",
  { spacing: { after: 140, line: 264 } }
));

// 7. Feature-based ML
children.push(h1("7. Feature-Based Machine Learning Model: XGBoost"));
children.push(p(
  "A feature-based model was built using a \u201cdirect multi-horizon\u201d design: one training row per (origin, hours-ahead) pair, covering hours-ahead = 1\u201324, so a single model learns to forecast any point in the next day. Features are restricted to what is genuinely knowable at the forecast origin: cyclically-encoded hour-of-day and day-of-week of the target time, a weekend flag, lagged Appliances values (1h, 24h and 168h relative to the origin), rolling mean/std (24h and 168h), and lag values measured relative to the specific target hour (24h and 168h before it \u2014 always valid within a 24-hour horizon)."
));
children.push(p(
  "Two feature sets were compared to directly test the effect of exogenous covariates. The \u201ctrue-forecast\u201d set (17 features) uses only the features above. The \u201cconditional\u201d set (41 features) additionally includes the actual indoor (T1\u2013T9, RH_1\u2013RH_9) and outdoor weather readings at the target time \u2014 real historical values that would not actually be known in advance in a live deployment (see Section 10, Question 5). Both were trained as XGBoost regressors (400 trees, max depth 5, learning rate 0.05) on 66,288 rows drawn only from the training period."
));
children.push(dataTable(
  ["Model variant", "RMSE (single)", "MAE (single)", "RMSE (rolling)", "MAE (rolling)"],
  [["True forecast (17 features)", "238.9", "172.9", "373.0", "240.9"], ["Conditional (41 features)", "231.3", "187.7", "379.3", "284.5"]],
  [3160, 1550, 1550, 1550, 1550]
));
children.push(p(
  "Feature importance (Figure 4) shows the true-forecast model relies overwhelmingly on cyclic day-of-week/hour-of-day encodings and the 24h/168h lag and rolling-mean features \u2014 seasonal and recency structure dominate. Strikingly, the conditional model, despite having access to real future weather and indoor sensor readings, did not outperform the honest model on the robust rolling check (379.3 vs. 373.0 RMSE) \u2014 these variables appear to add noise rather than genuine signal for this specific target, echoing the original dataset paper\u2019s own finding that weather contributes relatively little to predicting appliance-specific (as opposed to heating) energy use. Neither XGBoost variant outperformed SARIMAX on the fair rolling comparison (373\u2013379 vs. 328), despite substantially greater model and pipeline complexity \u2014 discussed further in Section 10."
));
children.push(...figure("13_xgb_true_forecast_feature_importance.png", 400, 200, "Figure 4. Top 15 feature importances, XGBoost (true-forecast variant)."));

// 8. Foundation model
children.push(h1("8. Foundation Model: Chronos"));
children.push(p(
  "Chronos (amazon/chronos-t5-small; Ansari et al., 2024) is a pretrained transformer for zero-shot time-series forecasting: it requires no training on the target series, only a window of recent history (\u201ccontext\u201d). The model was given the last 336 hours (14 days) of training data as context and asked for a 24-hour probabilistic forecast (100 sampled trajectories; median taken as the point forecast, 5th\u201395th percentile as a 90% interval)."
));
children.push(p(
  "Chronos required a separate execution environment (Google Colab) since the sandbox used for the rest of this project blocks the HuggingFace network domain that hosts its pretrained weights; only the single 24-hour evaluation was therefore run, not the 14-day rolling check used for the other models."
));
children.push(dataTable(["Evaluation", "RMSE", "MAE", "MAPE %"], [["Single 24h forecast", "305.2", "215.1", "26.4"]], [3960, 1800, 1800, 1800]));
children.push(...figure("14_chronos_forecast.png", 440, 198, "Figure 5. Chronos 24-hour forecast (median and 90% interval) vs. actual."));
children.push(p(
  "As Figure 5 shows, Chronos produces a smooth, damped forecast that tracks the broad overnight-low/evening-rise shape but substantially under-predicts the magnitude of both sharp peaks \u2014 actual usage exceeds even the top of its own 90% interval late in the day. This is a sensible failure mode for a general-purpose, zero-shot model that has never seen this specific house\u2019s idiosyncratic behaviour; it defaults to a cautious, averaged prediction. Its accuracy lands close to the Mean and Weekly Seasonal Naive benchmarks and clearly behind both SARIMAX and XGBoost.",
  { spacing: { after: 140, line: 264 } }
));

// 9. Comparison
children.push(h1("9. Consolidated Model Comparison"));
children.push(p("Table 3 and Figure 6 bring every model onto the same footing."));
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
  "SARIMAX is the clear winner on both evaluation regimes. XGBoost, despite far greater complexity, does not surpass it here; the foundation model, despite requiring zero training, is competitive only with the simplest benchmarks."
));

// 10. Discussion
children.push(h1("10. Critical Discussion"));

children.push(h2("Q1. Which benchmark model is strongest, and what does this tell you about the structure of appliance energy use?"));
children.push(p(
  "On the single-day test Weekly Seasonal Naive appeared strongest, but that result depended on an unrepresentative final training hour, which caused Naive and Drift to collapse. The fairer rolling check shows Mean and Weekly Seasonal Naive are roughly tied as strongest, with Daily Seasonal Naive clearly worse. This shows appliance use has a real average daily/weekly rhythm, but the exact timing of usage on any given day is too noisy for \u201ccopy yesterday exactly\u201d to work reliably \u2014 and the very poor showing of Naive/Drift confirms the series has no persistent short-term momentum (Section 3)."
));

children.push(h2("Q2. Does SARIMAX improve on the strongest benchmark? Are daily seasonality, autocorrelation, and exogenous variables adequately captured?"));
children.push(p(
  "Yes, clearly: a \u224816% RMSE reduction on the rolling check (328.1 vs. 391.2) and a much larger margin on the single-day test. The residual ACF shows no significant leftover autocorrelation, so the seasonal AR/MA structure adequately captures daily seasonality and short-run dependence. No exogenous variables were included in this SARIMAX (that was deliberately reserved for the feature-based model); given how little weather/indoor sensor data helped XGBoost (Q3), it is plausible an exogenous SARIMAX would add limited further benefit, though this was not directly tested."
));

children.push(h2("Q3. Does XGBoost improve when lag, rolling-window, time-of-day, and sensor/weather variables are added? Which feature groups are most useful?"));
children.push(p(
  "Time-of-day and day-of-week encodings, plus the 24h/168h lag and rolling-mean features, dominate feature importance. Adding real future weather/indoor sensor readings (the conditional variant) did not improve the robust rolling score (379.3 vs. 373.0 \u2014 slightly worse), despite using information the honest model did not have. This indicates weather/indoor readings carry little independent predictive value here beyond time-of-day and lag structure. Neither XGBoost variant beat SARIMAX on the rolling evaluation, suggesting SARIMAX\u2019s seasonal ARIMA structure suits this particular series better than a generic tree-based regressor, and/or that \u2248115 days of training history is limited for a single global model covering all 24 forecast horizons at once."
));

children.push(h2("Q4. Does the foundation model outperform the simpler benchmark, SARIMAX, and feature-based models? Is any improvement large enough to justify the extra complexity?"));
children.push(p(
  "No. Chronos (RMSE 305.2 on the single 24h test) is worse than SARIMAX (159.6) and both XGBoost variants (231\u2013239), and only roughly matches the Mean benchmark (300.0). It visibly missed the sharp evening peak in the test window, defaulting to a damped, cautious trajectory \u2014 expected behaviour for a model with zero exposure to this specific house\u2019s history. For a single, well-established series with 4+ months of dedicated data, the extra complexity (and, in this project, the added friction of a separate execution environment) is not justified on accuracy grounds. A zero-shot foundation model would be more attractive in a cold-start scenario \u2014 a brand-new home with no history to fit a per-home SARIMAX or XGBoost model on yet \u2014 where instant forecasting without training is worth more than peak accuracy."
));

children.push(h2("Q5. Which variables are genuinely known at the forecast origin? Is using future weather/indoor values from the test set a true or conditional forecast?"));
children.push(p(
  "Genuinely knowable at the forecast origin: all past/lagged Appliances values, and purely calendar-based features (hour-of-day, day-of-week), since calendars are known in advance. Not genuinely knowable: actual future indoor temperature/humidity (depends on future weather, heating decisions and occupancy) or actual future outdoor weather \u2014 in production these would require an external weather forecast, itself imperfect, rather than ground-truth values. The \u201cconditional\u201d XGBoost model, which used real realised future readings from the test set, is therefore not a true forecast but a conditional/hindsight estimate that overstates what is actually achievable in production. Notably (Q3), this hindsight information did not even help here \u2014 but the leakage risk is real and must be flagged whenever exogenous variables are used without first forecasting those variables themselves."
));

children.push(h2("Q6. Considering accuracy, interpretability, uncertainty, computational cost, and ease of deployment, which model would you recommend, and why?"));
children.push(p(
  "SARIMAX(1,0,6)(1,1,1,24) is recommended for this use case. It is the most accurate model tested on both evaluation regimes; it provides native confidence intervals (with the below-zero caveat noted in Section 6); its seasonal ARIMA structure and clean residual diagnostics are reasonably interpretable; and it deploys with a stable, well-documented library and no complex feature pipeline. Its main weakness is computational cost \u2014 the full grid search and each rolling refit (\u224850 seconds per fit) are non-trivial, though entirely manageable for a single household forecasting once a day. If the goal instead were forecasting many households simultaneously, or households with little history, the recommendation would likely shift: a single XGBoost model amortises training cost across many series, and a zero-shot foundation model becomes attractive precisely where no per-household history yet exists to fit SARIMAX or XGBoost on."
));

// 11. Limitations
children.push(h1("11. Limitations and Future Work"));
children.push(bullet("The SARIMAX grid search used a reduced 45-day window and a capped iteration limit for computational tractability; 90 of 147 combinations did not fully converge (the selected winner was unaffected, but a fuller search could plausibly find a better model)."));
children.push(bullet("No exogenous SARIMAX variant was tested; combining SARIMAX\u2019s seasonal structure with selected covariates is a natural extension."));
children.push(bullet("The XGBoost model is a single global model covering all 24 forecast horizons at once (a \u201cdirect\u201d approach); a per-horizon ensemble or a recursive strategy might improve near-term accuracy."));
children.push(bullet("Chronos was evaluated only on a single 24-hour window, not the 14-day rolling check used elsewhere, and only the \u201csmall\u201d model variant was tried, not \u201cbase\u201d/\u201clarge\u201d or a fine-tuned version."));
children.push(bullet("The analysis covers a single house over one 4.5-month period; findings (e.g. the limited value of weather features) may not generalise to other homes, climates, or seasons with heavier heating/cooling load."));
children.push(p(
  "Future work could test exogenous SARIMAX, a per-horizon or recursive XGBoost strategy, a fine-tuned or larger Chronos model, a bounded/log-transformed model to avoid physically-impossible negative forecast intervals, and validation across multiple households.",
  { spacing: { after: 140, line: 264 } }
));

// 12. Conclusion
children.push(h1("12. Conclusion and Recommendation"));
children.push(p(
  "Across benchmark, statistical, machine-learning, and foundation-model approaches, SARIMAX(1,0,6)(1,1,1,24) produced the most accurate 24-hour-ahead forecasts of household appliance energy use, improving on the strongest benchmark by \u224816% RMSE and outperforming a substantially more complex feature-based XGBoost model. The zero-shot Chronos foundation model, while requiring no training, was only competitive with the simplest benchmarks and is best suited to cold-start scenarios rather than a single, data-rich series like this one. For this specific forecasting task \u2014 a single well-established household with several months of clean history \u2014 SARIMAX is the recommended model, balancing strong accuracy with manageable computational cost and straightforward deployment."
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
