import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import math
import random
from types import SimpleNamespace
import numpy as np
import torch

# ARIMA ignore ConvergenceWarning
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Model modules
_baseline_dir = os.path.dirname(os.path.abspath(__file__))
if _baseline_dir not in sys.path:
    sys.path.insert(0, _baseline_dir)

from models import (
    run_arima_baseline,
    run_var_baseline,
    run_lstm_baseline,
    run_mtgnn_baseline,
    run_arima_transformer_baseline,
)

# 0. Device / Seed / Paths
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

baseline_dir   = os.path.dirname(os.path.abspath(__file__))
bmtgnn_dir     = os.path.dirname(baseline_dir)
workspace_root = os.path.dirname(bmtgnn_dir)

data_path = os.path.join(
    workspace_root, 'data_preparation/data/5_bmtgnn_input/bmtgnn_data.npy'
)
if not os.path.exists(data_path):
    alt = os.path.join(bmtgnn_dir, 'data_preparation/data/5_bmtgnn_input/bmtgnn_data.npy')
    if os.path.exists(alt):
        data_path = alt
print(f"Data path: {data_path}")

# 1. Load raw data  [N, T, F]  →  transpose  [T, N, F]
raw = np.load(data_path).transpose(1, 0, 2)   # [T, N, F]
T, N, F = raw.shape
print(f"Loaded data: T={T}, N={N}, F={F}")
horizon = 1    # args.horizon

while True:
    try:
        user_input = input("\ninput forecast month (3, 6, 9, 12, 24, 36 / default: 3): ").strip()
        if user_input == "":
            months = 3
            break
        months = int(user_input)
        if months in [3, 6, 9, 12, 24, 36]:
            break
    except ValueError:
        print("not a number.")

seq_out_len = months
seq_in_len = seq_out_len

out = seq_out_len
P   = seq_in_len
h   = horizon
num_eval = 7

# 2. Normalization — mirrors DataLoaderS._normalized(mode=2)
#    To prevent Data Leakage, we calculate max scale ONLY up to train_end.
train_end = T - 2 * num_eval
scale = np.ones((N, F))
dat   = np.zeros_like(raw)
for i in range(N):
    for j in range(F):
        mx = np.max(np.abs(raw[:train_end, i, j]))
        if mx == 0:
            mx = 1.0
        scale[i, j]   = mx
        dat[:, i, j]  = raw[:, i, j] / mx

# 3. Window splitting — mirrors DataLoaderS._split
test_starts  = list(range(T - out - num_eval + 1, T - out + 1))
valid_starts = list(range(T - out - 2 * num_eval + 1, T - out - num_eval + 1))
train_starts = list(range(P + h - 1, T - out - 2 * num_eval + 1))


def make_windows(starts):
    """
    Build (X_norm, Y_level) arrays from window start indices.

    Returns-
    X : np.ndarray  [n_wins, P, N, F]   normalised history
    Y : np.ndarray  [n_wins, out, N]    level-domain target (feature 0)
    """
    n = len(starts)
    X = np.zeros((n, P, N, F))
    Y = np.zeros((n, out, N))
    for i, start_idx in enumerate(starts):
        end   = start_idx - h + 1
        start = end - P
        X[i]  = dat[start:end, :, :]
        Y[i]  = raw[start_idx: start_idx + out, :, 0]
    return X, Y


X_train_norm, Y_train_lvl = make_windows(train_starts)
X_val_norm,   Y_val_lvl   = make_windows(valid_starts)
X_test_norm,  Y_test_lvl  = make_windows(test_starts)

print(f"Train windows: {len(train_starts)}, "
      f"Val windows: {len(valid_starts)}, "
      f"Test windows: {len(test_starts)}")

# 4. Metric helpers
def evaluate_predictions(preds, actual, train_raw=None):
    """
    Compute all 12 metrics matching train_test.py evaluate_direct().

    Parameters-
    preds, actual : np.ndarray  [n_windows, out, N]  — level domain
    train_raw     : np.ndarray  [n_train, N, F] or None  — for MASE / Hist-RSE

    Returns-
    dict with keys:
        rse, r2, mase, smape, mape, mse, mae, rmse, rae, hist_rse, global_rse, corr
    """
    eps = 1e-8
    node_rse, node_r2, node_mase = [], [], []
    node_smape, node_mape = [], []
    node_mse, node_mae, node_rmse = [], [], []
    node_rae, node_hist_rse, node_corr = [], [], []

    # For Hist-RSE: train std per node (feature 0)
    if train_raw is not None:
        train_std = np.std(train_raw[:, :, 0], axis=0)   # [N]
    else:
        train_std = None

    # For MASE: seasonal naive MAE (h=out lag) per node (feature 0)
    h_mase = out
    if train_raw is not None and train_raw.shape[0] > h_mase:
        naive_mae_per_node = np.mean(
            np.abs(train_raw[h_mase:, :, 0] - train_raw[:-h_mase, :, 0]), axis=0
        )  # [N]
    else:
        naive_mae_per_node = None

    for n in range(N):
        pred_n = preds[:, :, n].flatten()
        act_n  = actual[:, :, n].flatten()

        # CORE METRICS
        mse_n  = np.mean((pred_n - act_n) ** 2)
        mae_n  = np.mean(np.abs(pred_n - act_n))
        rmse_n = math.sqrt(mse_n)

        # RSE(window) — node-level relative squared error
        ss_err = np.sum((act_n - pred_n) ** 2)
        ss_tot = np.sum((act_n - np.mean(act_n)) ** 2)
        rse_n  = math.sqrt(ss_err / (ss_tot + eps))

        # R2-Score
        r2_n   = 1.0 - ss_err / (ss_tot + eps)

        # RAE — relative absolute error
        sum_abs_err = np.sum(np.abs(act_n - pred_n))
        sum_abs_dev = np.sum(np.abs(act_n - np.mean(act_n)))
        rae_n  = sum_abs_err / (sum_abs_dev + eps)

        # SMAPE
        smape_n = float(np.mean(
            np.abs(pred_n - act_n) / (np.abs(act_n) + np.abs(pred_n) + eps)
        ))

        # MAPE
        mape_n = float(np.mean(np.abs((act_n - pred_n) / (np.abs(act_n) + eps))))

        # MASE — scale by training naive MAE (h=12)
        if naive_mae_per_node is not None and naive_mae_per_node[n] > eps:
            mase_n = mae_n / naive_mae_per_node[n]
        else:
            mase_n = float('nan')

        # Hist-RSE — RMSE / training std
        if train_std is not None and train_std[n] > eps:
            hist_rse_n = rmse_n / train_std[n]
        else:
            hist_rse_n = float('nan')

        # Corr (Pearson)
        corr_n = (
            float(np.corrcoef(pred_n, act_n)[0, 1])
            if np.std(pred_n) > 0 and np.std(act_n) > 0
            else 0.0
        )

        node_rse.append(rse_n)
        node_r2.append(r2_n)
        node_mase.append(mase_n)
        node_smape.append(smape_n)
        node_mape.append(mape_n)
        node_mse.append(mse_n)
        node_mae.append(mae_n)
        node_rmse.append(rmse_n)
        node_rae.append(rae_n)
        node_hist_rse.append(hist_rse_n)
        node_corr.append(corr_n)

    # Global-RSE (full array, not per-node)
    sum_sq_err  = np.sum((preds - actual) ** 2)
    mean_all    = np.mean(actual, axis=0, keepdims=True)  # Match train_test.py (axis=0)
    sum_sq_dev  = np.sum((actual - mean_all) ** 2)
    global_rse  = math.sqrt(sum_sq_err) / (math.sqrt(sum_sq_dev) + eps)

    def _nanmean(lst):
        vals = [v for v in lst if not (isinstance(v, float) and math.isnan(v))]
        return float(np.mean(vals)) if vals else float('nan')

    return {
        'rse':        float(np.mean(node_rse)),
        'r2':         float(np.mean(node_r2)),
        'mase':       _nanmean(node_mase),
        'smape':      float(np.mean(node_smape)),
        'mape':       float(np.mean(node_mape)),
        'mse':        float(np.mean(node_mse)),
        'mae':        float(np.mean(node_mae)),
        'rmse':       float(np.mean(node_rmse)),
        'rae':        float(np.mean(node_rae)),
        'hist_rse':   _nanmean(node_hist_rse),
        'global_rse': global_rse,
        'corr':       float(np.mean(node_corr)),
    }


def print_and_collect(name, split, metrics):
    """
    Print all 12 metrics in train_test.py format and return the metrics dict.
    """
    print(
        f"[{name}] {split}  "
        f"RSE(window): {metrics['rse']:.4f} | "
        f"R2-Score: {metrics['r2']:.4f} | "
        f"MASE: {metrics['mase']:.4f} | "
        f"SMAPE: {metrics['smape']:.4f} | "
        f"MAPE: {metrics['mape']:.4f} | "
        f"MSE: {metrics['mse']:.4f} | "
        f"MAE: {metrics['mae']:.4f} | "
        f"RMSE: {metrics['rmse']:.4f} | "
        f"RAE: {metrics['rae']:.4f} | "
        f"Hist-RSE: {metrics['hist_rse']:.4f} | "
        f"Global-RSE: {metrics['global_rse']:.4f} | "
        f"Corr: {metrics['corr']:.4f}"
    )
    return metrics


# 5. Build DataContext — passed to every model module
# train_raw: [n_train, N, F] level-domain — needed for MASE and Hist-RSE
train_raw = raw[: train_starts[-1] + out + 1, :, :]  # generous upper bound

# Wrap evaluate with train_raw pre-bound
def _eval_with_train(preds, actual):
    return evaluate_predictions(preds, actual, train_raw=train_raw)

ctx = SimpleNamespace(
    # raw data & splits
    raw           = raw,
    dat           = dat,
    scale         = scale,
    train_raw     = train_raw,
    # dimensions
    T=T, N=N, F=F,
    P=P, out=out, h=h,
    # window indices
    train_starts  = train_starts,
    valid_starts  = valid_starts,
    test_starts   = test_starts,
    # pre-built window arrays
    X_train_norm  = X_train_norm,
    Y_train_lvl   = Y_train_lvl,
    X_val_norm    = X_val_norm,
    Y_val_lvl     = Y_val_lvl,
    X_test_norm   = X_test_norm,
    Y_test_lvl    = Y_test_lvl,
    # device & paths
    device        = device,
    bmtgnn_dir    = bmtgnn_dir,
    # metric helpers (bound as callables so modules need no extra imports)
    evaluate      = _eval_with_train,
    report        = print_and_collect,
    # shared result cache (e.g. ARIMA predictions reused by ARIMA+Transformer)
    cache         = {},
)

# 6. Select models to run
print("\nSelect models to run (comma separated) or press Enter for All:")
print("1: ARIMA")
print("2: VAR")
print("3: LSTM")
print("4: MTGNN")
print("5: ARIMA+Transformer")
print("All: Press Enter")

user_models = input("Selection: ").strip()
run_all = (user_models == "")

if not run_all:
    selected_indices = [idx.strip() for idx in user_models.split(',')]
else:
    selected_indices = ["1", "2", "3", "4", "5"]

results = {}

if "1" in selected_indices:
    results["ARIMA"] = run_arima_baseline(ctx)
if "2" in selected_indices:
    results["VAR"] = run_var_baseline(ctx)
if "3" in selected_indices:
    results["LSTM"] = run_lstm_baseline(ctx)
if "4" in selected_indices:
    results["MTGNN"] = run_mtgnn_baseline(ctx)
if "5" in selected_indices:
    if "1" not in selected_indices and "ARIMA" not in results:
        print("\n[Warning] ARIMA+Transformer needs ARIMA to run first. Running ARIMA automatically...")
        results["ARIMA"] = run_arima_baseline(ctx)
    results["ARIMA+Transformer"] = run_arima_transformer_baseline(ctx)

# 7. Save results
out_path = os.path.join(bmtgnn_dir, f'baseline/result/baseline_results_{months}mo_.txt')
csv_path = os.path.join(bmtgnn_dir, f'baseline/result/baseline_results_{months}mo_.csv')
os.makedirs(os.path.dirname(out_path), exist_ok=True)

METRIC_KEYS   = ['rse', 'r2', 'mase', 'smape', 'mape', 'mse', 'mae', 'rmse', 'rae', 'hist_rse', 'global_rse', 'corr']
METRIC_LABELS = ['RSE(window)', 'R2-Score', 'MASE', 'SMAPE', 'MAPE', 'MSE', 'MAE', 'RMSE', 'RAE', 'Hist-RSE', 'Global-RSE', 'Corr']

with open(out_path, 'w') as f:
    col_w   = 12   # width per metric column
    model_w = 20
    split_w = 12
    sep_len = model_w + split_w + len(METRIC_KEYS) * (col_w + 2) + 2
    sep_line = "-" * sep_len + "\n"

    header_labels = "".join(f"{lbl:>{col_w}}  " for lbl in METRIC_LABELS)
    header = f"{'Model':<{model_w}}  {'Split':<{split_w}}  {header_labels}\n"

    f.write("Baseline Comparison Results (val/test)\n")
    f.write("=" * sep_len + "\n")
    f.write(header)
    f.write(sep_line)

    for model_name, splits in results.items():
        for split_key, m in splits.items():
            if m is None:
                f.write(f"{model_name:<{model_w}}  {split_key:<{split_w}}  {'N/A':>{col_w}}\n")
            else:
                vals = "".join(f"{m[k]:>{col_w}.4f}  " for k in METRIC_KEYS)
                f.write(f"{model_name:<{model_w}}  {split_key:<{split_w}}  {vals}\n")

    f.write("=" * sep_len + "\n")

with open(csv_path, 'w') as f_csv:
    csv_header = "Model,Split," + ",".join(METRIC_LABELS) + "\n"
    f_csv.write(csv_header)
    for model_name, splits in results.items():
        for split_key, m in splits.items():
            if m is None:
                vals = ",".join(["N/A"] * len(METRIC_KEYS))
                f_csv.write(f"{model_name},{split_key},{vals}\n")
            else:
                vals = ",".join(f"{m[k]:.4f}" for k in METRIC_KEYS)
                f_csv.write(f"{model_name},{split_key},{vals}\n")

print(f"\nSaved results: {out_path} and {csv_path}")
