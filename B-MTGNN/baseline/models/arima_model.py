"""
arima_model.py
──────────────
Baseline 1: ARIMA(1,1,1) — per-node univariate forecasting.

Public API
----------
run_arima_baseline(ctx) -> dict
    ctx : DataContext  (see run_baselines.py)
    Returns a result dict:
        {
            "val":  dict(rse, mae, mse, rmse, corr),
            "test": dict(rse, mae, mse, rmse, corr),
        }
    or {"val": None, "test": None} when statsmodels is unavailable.
"""

import numpy as np

try:
    from statsmodels.tsa.arima.model import ARIMA
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False


# ── internal helper ────────────────────────────────────────────────────────────

def _predict(starts, raw, h, out, N):
    """
    Run ARIMA(1,1,1) independently per node over each window defined by `starts`.

    Parameters
    ----------
    starts : list[int]   window start indices
    raw    : np.ndarray  [T, N, F]  level-domain data
    h      : int         forecast horizon offset (1-based lag)
    out    : int         number of steps to forecast
    N      : int         number of nodes

    Returns
    -------
    preds : np.ndarray  [n_windows, out, N]  level-domain forecasts
    """
    preds = np.zeros((len(starts), out, N))
    for wi, start_idx in enumerate(starts):
        end = start_idx - h + 1   # exclusive upper bound of history
        for node in range(N):
            history = raw[:end, node, 0]   # level-domain univariate series
            try:
                model = ARIMA(history, order=(1, 1, 1))
                res   = model.fit()
                preds[wi, :, node] = res.forecast(steps=out)
            except Exception:
                # fallback: persistence (repeat last observed value)
                preds[wi, :, node] = np.repeat(history[-1], out)
    return preds


# ── public API ─────────────────────────────────────────────────────────────────

def run_arima_baseline(ctx):
    """
    Train (fit) and evaluate ARIMA(1,1,1) on validation and test splits.

    Parameters
    ----------
    ctx : DataContext  namedtuple from run_baselines.py

    Returns
    -------
    result : dict  {"val": metrics_dict, "test": metrics_dict}
             metrics_dict keys: rse, mae, mse, rmse, corr
             Returns None values when statsmodels is not installed.
    """
    if not _has_statsmodels:
        print("[ARIMA] statsmodels not found — skipping.")
        return {"val": None, "test": None}

    print("\n[ARIMA] Running ARIMA(1,1,1)...")

    ar_val  = _predict(ctx.valid_starts, ctx.raw, ctx.h, ctx.out, ctx.N)
    ar_test = _predict(ctx.test_starts,  ctx.raw, ctx.h, ctx.out, ctx.N)

    rv = ctx.evaluate(ar_val,  ctx.Y_val_lvl)
    rt = ctx.evaluate(ar_test, ctx.Y_test_lvl)

    ctx.report("ARIMA", "Validation", rv)
    ctx.report("ARIMA", "Testing",    rt)

    # Expose ar_val / ar_test so ARIMA+Transformer can reuse them
    ctx.cache["ar_val"]  = ar_val
    ctx.cache["ar_test"] = ar_test

    return {
        "val":  rv,
        "test": rt,
    }
