"""
var_model.py
────────────
Baseline 2: VAR(1) — multivariate linear autoregression.

Uses all N nodes simultaneously (feature 0) for joint forecasting.
Falls back to last-value persistence per node on model failure.

Public API
----------
run_var_baseline(ctx) -> dict
"""

import numpy as np

try:
    from statsmodels.tsa.vector_ar.var_model import VAR
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False


# ── internal helper ────────────────────────────────────────────────────────────

def _predict(starts, raw, h, out, N):
    """
    Fit a VAR(1) model jointly over all N nodes for each window.

    Parameters
    ----------
    starts : list[int]   window start indices
    raw    : np.ndarray  [T, N, F]  level-domain data
    h      : int         forecast horizon offset
    out    : int         steps to forecast
    N      : int         number of nodes

    Returns
    -------
    preds : np.ndarray  [n_windows, out, N]
    """
    preds = np.zeros((len(starts), out, N))
    for wi, start_idx in enumerate(starts):
        end     = start_idx - h + 1
        history = raw[:end, :, 0]   # [t, N] level-domain, feature 0 only
        try:
            model = VAR(history)
            res   = model.fit(maxlags=1)
            preds[wi, :, :] = res.forecast(history[-res.k_ar:], steps=out)
        except Exception:
            # fallback: persistence per node
            for node in range(N):
                preds[wi, :, node] = np.repeat(raw[end - 1, node, 0], out)
    return preds


# ── public API ─────────────────────────────────────────────────────────────────

def run_var_baseline(ctx):
    """
    Fit and evaluate VAR(1) on validation and test splits.

    Parameters
    ----------
    ctx : DataContext  namedtuple from run_baselines.py

    Returns
    -------
    result : dict  {"val": metrics_dict, "test": metrics_dict}
    """
    if not _has_statsmodels:
        print("[VAR] statsmodels not found — skipping.")
        return {"val": None, "test": None}

    print("\n[VAR] Running VAR(1)...")

    var_val  = _predict(ctx.valid_starts, ctx.raw, ctx.h, ctx.out, ctx.N)
    var_test = _predict(ctx.test_starts,  ctx.raw, ctx.h, ctx.out, ctx.N)

    rv = ctx.evaluate(var_val,  ctx.Y_val_lvl)
    rt = ctx.evaluate(var_test, ctx.Y_test_lvl)

    ctx.report("VAR  ", "Validation", rv)
    ctx.report("VAR  ", "Testing",    rt)

    return {
        "val":  rv,
        "test": rt,
    }
