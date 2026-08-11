"""
sarimax_grid_search.py (resumable version)
--------------------------------------------
Same grid search as before, but designed to be run multiple times in a row:
each run reads what's already in the results CSV, skips combinations already
done, and works through the remaining ones for up to TIME_BUDGET_SECONDS
before exiting cleanly. Run it repeatedly until it prints "ALL DONE".
"""

import warnings
import itertools
import time
import os
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_utils import load_hourly, train_test_split, TARGET

METRICS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = METRICS_DIR / "part4_sarimax_grid_search.csv"

SEASONAL_ORDER = (1, 1, 1, 24)
SEARCH_WINDOW_DAYS = 45
P_RANGE = range(0, 7)
D_RANGE = range(0, 3)
Q_RANGE = range(0, 7)
TIME_BUDGET_SECONDS = float(os.environ.get("TIME_BUDGET_SECONDS", 170))


def get_done_combos():
    if not OUT_CSV.exists():
        return set()
    df = pd.read_csv(OUT_CSV)
    return set(zip(df["p"], df["d"], df["q"]))


def run_grid_search():
    df = load_hourly()
    train, test = train_test_split(df)
    y_search = train[TARGET].iloc[-SEARCH_WINDOW_DAYS * 24:]

    all_combos = list(itertools.product(P_RANGE, D_RANGE, Q_RANGE))
    done = get_done_combos()
    remaining = [c for c in all_combos if c not in done]

    if not OUT_CSV.exists():
        with open(OUT_CSV, "w") as f:
            f.write("p,d,q,P,D,Q,s,aic,converged,fit_seconds\n")

    print(f"Total={len(all_combos)}  Already done={len(done)}  Remaining={len(remaining)}", flush=True)

    t_start = time.time()
    n_done_this_run = 0
    for (p, d, q) in remaining:
        if time.time() - t_start > TIME_BUDGET_SECONDS:
            print(f"Time budget reached, stopping this batch. "
                  f"Done this run: {n_done_this_run}. Remaining overall: "
                  f"{len(remaining) - n_done_this_run}", flush=True)
            return

        t0 = time.time()
        try:
            model = SARIMAX(
                y_search, order=(p, d, q), seasonal_order=SEASONAL_ORDER,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            res = model.fit(disp=False, method="lbfgs", maxiter=50)
            aic = res.aic
            converged = not res.mle_retvals.get("warnflag", 0)
        except Exception:
            aic = np.inf
            converged = False
        elapsed = time.time() - t0

        with open(OUT_CSV, "a") as f:
            f.write(f"{p},{d},{q},{SEASONAL_ORDER[0]},{SEASONAL_ORDER[1]},"
                    f"{SEASONAL_ORDER[2]},{SEASONAL_ORDER[3]},{aic},{converged},{elapsed:.2f}\n")

        n_done_this_run += 1
        print(f"order=({p},{d},{q}) AIC={aic:.1f} fit={elapsed:.1f}s "
              f"[{len(done) + n_done_this_run}/{len(all_combos)} overall]", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    run_grid_search()
