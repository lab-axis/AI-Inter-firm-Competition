"""
arima_transformer_model.py
──────────────────────────
Baseline 5: ARIMA + Transformer Hybrid.

Decomposition strategy
----------------------
  Final prediction = ARIMA linear forecast  +  Transformer residual forecast

Step-by-step
~~~~~~~~~~~~
1. ARIMA(1,1,1) forecasts the linear component (level domain).
2. Training residuals = Y_true_level − ARIMA_pred_level  (normalised before training).
3. A Transformer encoder learns to predict those residuals from the history window.
4. At inference: ARIMA forecast + de-normalised Transformer residual.

Architecture — ResidualTransformer
-----------------------------------
- Input projection : [B, P, N*F] → [B, P, d_model]
- Sinusoidal Positional Encoding
- Transformer Encoder  (nhead=4, dim_feedforward=4×d_model, num_layers=2)
- MLP Decoder (last encoder token → [B, out*N])

Training details
~~~~~~~~~~~~~~~~
- Huber Loss  (robust to ARIMA outlier residuals)
- AdamW  with weight_decay=1e-4
- Cosine Annealing LR scheduler
- Gradient clipping  (max_norm=1.0)

Public API
----------
run_arima_transformer_baseline(ctx) -> dict
"""

import math
import numpy as np
import torch
import torch.nn as nn

try:
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False


# ── ARIMA helper (reuses ctx.cache if available) ───────────────────────────────

def _arima_predict(starts, raw, h, out, N):
    """Fit ARIMA(1,1,1) per node per window. Returns [n_windows, out, N]."""
    preds = np.zeros((len(starts), out, N))
    for wi, start_idx in enumerate(starts):
        end = start_idx - h + 1
        for node in range(N):
            history = raw[:end, node, 0]
            try:
                res = _ARIMA(history, order=(1, 1, 1)).fit()
                preds[wi, :, node] = res.forecast(steps=out)
            except Exception:
                preds[wi, :, node] = np.repeat(history[-1], out)
    return preds


# Positional Encoding

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).

    Parameters
    ----------
    d_model : int   embedding dimension
    max_len : int   maximum sequence length
    dropout : float dropout probability
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model]"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ResidualTransformer model

class ResidualTransformer(nn.Module):
    """
    Transformer encoder that predicts ARIMA residuals.

    Parameters
    ----------
    in_dim    : int  number of input features per node
    n_nodes   : int  number of graph nodes
    seq_len   : int  history window length
    out_len   : int  forecast horizon
    d_model   : int  Transformer embedding dimension (default 64)
    nhead     : int  number of attention heads (default 4)
    num_layers: int  number of encoder layers (default 2)
    dropout   : float dropout probability (default 0.1)
    """

    def __init__(
        self,
        in_dim: int,
        n_nodes: int,
        seq_len: int,
        out_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.out_len = out_len

        # Project univariate [F] input → d_model
        self.input_proj = nn.Linear(in_dim, d_model)

        self.pos_enc = PositionalEncoding(
            d_model, max_len=seq_len + out_len, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP decoder: last encoder hidden state → [out_len]
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, out_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, P, N, F]  normalised history

        Returns
        -------
        Tensor  [B, out_len, N]  normalised residual predictions
        """
        b, seq, nodes, feats = x.shape
        # Treat nodes as batch dimension: [B*N, P, F]
        x_uni  = x.permute(0, 2, 1, 3).reshape(b * nodes, seq, feats)
        
        x_proj = self.pos_enc(self.input_proj(x_uni))      # [B*N, P, d_model]
        enc    = self.encoder(x_proj)                      # [B*N, P, d_model]
        last   = enc[:, -1, :]                             # [B*N, d_model]
        out_h  = self.decoder(last)                        # [B*N, out_len]
        
        # Restore shape: [B, N, out_len] -> [B, out_len, N]
        return out_h.reshape(b, nodes, self.out_len).permute(0, 2, 1)


# Training
def _train(
    model,
    X_train_norm,
    resid_train_norm,
    X_val_norm,
    resid_val_norm,
    device,
    epochs: int = 150,
    batch:  int = 8,
):
    """
    Train ResidualTransformer on normalised ARIMA residuals with Early Stopping.
    """
    X_t = torch.from_numpy(X_train_norm).float()
    R_t = torch.from_numpy(resid_train_norm).float()
    X_v = torch.from_numpy(X_val_norm).float()
    R_v = torch.from_numpy(resid_val_norm).float()

    optimizer  = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion  = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    best_val_loss = float('inf')
    best_state = None
    es_counter = 0
    es_patience = 30

    import copy

    for epoch in range(1, epochs + 1):
        model.train()
        perm  = torch.randperm(len(X_t))
        eloss = 0.0
        nb    = 0
        for i in range(0, len(X_t), batch):
            idx = perm[i : i + batch]
            bx  = X_t[idx].to(device)   # [b, P, N, F]
            by  = R_t[idx].to(device)   # [b, out, N]
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            eloss += loss.item()
            nb    += 1
            
        model.eval()
        with torch.no_grad():
            pred_v = model(X_v.to(device))
            by_v = R_v.to(device)
            val_loss = criterion(pred_v, by_v).item()
            
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  AT Epoch {epoch}/{epochs} | Train Loss: {eloss / nb:.6f} | Val Loss: {val_loss:.6f}")
            
        if es_counter >= es_patience:
            print(f"  AT Early stopping at epoch {epoch}")
            break
            
    if best_state is not None:
        model.load_state_dict(best_state)


# Evaluation
def _evaluate(model, X_np, ar_preds, scale, device, out, N):
    """
    Combine ARIMA forecast and Transformer residual forecast (level domain).

    Parameters
    ----------
    model    : ResidualTransformer
    X_np     : np.ndarray  [n_windows, P, N, F]  normalised history
    ar_preds : np.ndarray  [n_windows, out, N]   ARIMA level-domain forecast
    scale    : np.ndarray  [N, F]
    device   : torch.device
    out      : int  forecast horizon
    N        : int  number of nodes

    Returns
    -------
    preds : np.ndarray  [n_windows, out, N]  level domain
    """
    model.eval()
    preds = np.zeros((len(X_np), out, N))
    for wi in range(len(X_np)):
        bx = torch.from_numpy(X_np[wi : wi + 1]).float().to(device)
        with torch.no_grad():
            resid_norm = model(bx).squeeze(0).cpu().numpy()   # [out, N] normalised
        resid_lvl  = resid_norm * scale[:, 0]                # de-normalise → level
        preds[wi]  = ar_preds[wi] + resid_lvl               # ARIMA + residual
    return preds


# Public API
def run_arima_transformer_baseline(ctx):
    """
    Build, train, and evaluate the ARIMA+Transformer hybrid baseline.

    Reuses cached ARIMA predictions from ctx.cache["ar_val"] /
    ctx.cache["ar_test"] if run_arima_baseline was executed first.

    Parameters
    ----------
    ctx : DataContext  namedtuple from run_baselines.py

    Returns
    -------
    result : dict  {"val": metrics_dict, "test": metrics_dict}
    """
    if not _has_statsmodels:
        print("[ARIMA+Transformer] statsmodels not found — skipping.")
        return {"val": None, "test": None}

    # Step 1: ARIMA predictions
    print("\n[ARIMA+Transformer] Step 1 — ARIMA(1,1,1) predictions...")

    ar_train = _arima_predict(ctx.train_starts, ctx.raw, ctx.h, ctx.out, ctx.N)
    
    ar_val = ctx.cache.get("ar_val")
    if ar_val is None:
        ar_val = _arima_predict(ctx.valid_starts, ctx.raw, ctx.h, ctx.out, ctx.N)
        
    ar_test = ctx.cache.get("ar_test")
    if ar_test is None:
        ar_test = _arima_predict(ctx.test_starts, ctx.raw, ctx.h, ctx.out, ctx.N)

    # Step 2: build normalised residuals
    # residual = Y_true_level − ARIMA_level, then normalise per-node
    resid_train_lvl  = ctx.Y_train_lvl - ar_train         # [n_train, out, N]
    resid_train_norm = resid_train_lvl / ctx.scale[:, 0]  # broadcast [N] → normalised

    resid_val_lvl = ctx.Y_val_lvl - ar_val
    resid_val_norm = resid_val_lvl / ctx.scale[:, 0]

    # Step 3: train Transformer on residuals
    print("[ARIMA+Transformer] Step 2 — training Transformer on ARIMA residuals...")

    model = ResidualTransformer(
        in_dim=ctx.F, n_nodes=ctx.N, seq_len=ctx.P, out_len=ctx.out,
        d_model=64, nhead=4, num_layers=2, dropout=0.1,
    ).to(ctx.device)

    _train(model, ctx.X_train_norm, resid_train_norm, ctx.X_val_norm, resid_val_norm, ctx.device, epochs=150, batch=8)

    # Step 4: evaluate
    print("[ARIMA+Transformer] Step 3 — evaluating...")

    at_val  = _evaluate(model, ctx.X_val_norm,  ar_val,  ctx.scale, ctx.device, ctx.out, ctx.N)
    at_test = _evaluate(model, ctx.X_test_norm, ar_test, ctx.scale, ctx.device, ctx.out, ctx.N)

    rv = ctx.evaluate(at_val,  ctx.Y_val_lvl)
    rt = ctx.evaluate(at_test, ctx.Y_test_lvl)

    ctx.report("ARIMA+T", "Validation", rv)
    ctx.report("ARIMA+T", "Testing",    rt)

    return {
        "val":  rv,
        "test": rt,
    }
