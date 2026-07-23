"""
mtgnn_model.py
──────────────
Baseline 4: Deterministic MTGNN (no Bayesian MC dropout).

Uses the gtnet architecture from B-MTGNN/net.py.
Input tensor is permuted from [B, P, N, F] → [B, F, N, P] to match gtnet convention.

Public API
----------
run_mtgnn_baseline(ctx) -> dict
"""

import sys
import numpy as np
import torch
import torch.nn as nn


# ── Training ───────────────────────────────────────────────────────────────────

def _train(model, X_train_norm, Y_train_lvl, X_val_norm, Y_val_lvl, scale, device, epochs=150, batch=8):
    """
    Train the MTGNN model in normalised space with Early Stopping.
    """
    scale_node0 = torch.from_numpy(scale[:, 0]).float().to(device)  # [N]

    X_t = torch.from_numpy(X_train_norm).float()
    Y_t = torch.from_numpy(Y_train_lvl).float()
    X_v = torch.from_numpy(X_val_norm).float()
    Y_v = torch.from_numpy(Y_val_lvl).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    criterion = nn.MSELoss()

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
            idx  = perm[i : i + batch]
            bx   = X_t[idx].to(device)                              # [b, P, N, F]
            by   = Y_t[idx].to(device) / scale_node0.unsqueeze(0).unsqueeze(0)
            bx_t = bx.permute(0, 3, 2, 1).float()                   # [b, F, N, P]
            optimizer.zero_grad()
            mu, sigma = model(bx_t)
            pred = torch.squeeze(mu, 3)                             # [b, out, N]
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            eloss += loss.item()
            nb    += 1
            
        model.eval()
        with torch.no_grad():
            bx_v_t = X_v.to(device).permute(0, 3, 2, 1).float()
            by_v = Y_v.to(device) / scale_node0.unsqueeze(0).unsqueeze(0)
            mu_v, sigma_v = model(bx_v_t)
            pred_v = torch.squeeze(mu_v, 3)
            val_loss = criterion(pred_v, by_v).item()
            
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"  MTGNN Epoch {epoch}/{epochs} | Train Loss: {eloss / nb:.6f} | Val Loss: {val_loss:.6f}")
            
        if es_counter >= es_patience:
            print(f"  MTGNN Early stopping at epoch {epoch}")
            break
            
    if best_state is not None:
        model.load_state_dict(best_state)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _evaluate(model, X_np, scale, device, out, N):
    """
    Run inference and return level-domain predictions.

    Parameters
    ----------
    model  : gtnet instance
    X_np   : np.ndarray  [n_windows, P, N, F]  normalised
    scale  : np.ndarray  [N, F]
    device : torch.device
    out    : int  forecast horizon
    N      : int  number of nodes

    Returns
    -------
    preds : np.ndarray  [n_windows, out, N]  level domain
    """
    model.eval()
    preds = np.zeros((len(X_np), out, N))
    for wi in range(len(X_np)):
        bx   = torch.from_numpy(X_np[wi : wi + 1]).float().to(device)
        bx_t = bx.permute(0, 3, 2, 1).float()                       # [1, F, N, P]
        with torch.no_grad():
            mu, sigma = model(bx_t)
            p = torch.squeeze(mu, 3).squeeze(0).cpu().numpy()           # [out, N]
        preds[wi] = p * scale[:, 0]   # de-scale → level domain
    return preds


# ── Public API ─────────────────────────────────────────────────────────────────

def run_mtgnn_baseline(ctx):
    """
    Build, train, and evaluate the deterministic MTGNN baseline.

    Dynamically imports gtnet from the B-MTGNN root (bmtgnn_dir),
    so this module must be called after sys.path has been configured
    *or* ctx.bmtgnn_dir is set.

    Parameters
    ----------
    ctx : DataContext  namedtuple from run_baselines.py
          Required extra field: ctx.bmtgnn_dir (str)

    Returns
    -------
    result : dict  {"val": metrics_dict, "test": metrics_dict}
    """
    print("\n[MTGNN] Training deterministic MTGNN...")

    # Dynamic import: gtnet lives in the B-MTGNN root directory
    if ctx.bmtgnn_dir not in sys.path:
        sys.path.insert(0, ctx.bmtgnn_dir)
    from net import gtnet  # noqa: E402  (intentional deferred import)

    model = gtnet(
        gcn_true=True, buildA_true=True, gcn_depth=2,
        num_nodes=ctx.N, device=ctx.device, predefined_A=None,
        dropout=0.3, subgraph_size=20, node_dim=40,
        dilation_exponential=2, conv_channels=16,
        residual_channels=16, skip_channels=32, end_channels=64,
        seq_length=ctx.P, in_dim=ctx.F, out_dim=ctx.out,
        layers=3, propalpha=0.05, tanhalpha=3, subtract_last=True,
    ).to(ctx.device)

    _train(
        model,
        ctx.X_train_norm, ctx.Y_train_lvl,
        ctx.X_val_norm, ctx.Y_val_lvl,
        ctx.scale, ctx.device,
        epochs=150, batch=8,
    )

    mtgnn_val  = _evaluate(model, ctx.X_val_norm,  ctx.scale, ctx.device, ctx.out, ctx.N)
    mtgnn_test = _evaluate(model, ctx.X_test_norm, ctx.scale, ctx.device, ctx.out, ctx.N)

    rv = ctx.evaluate(mtgnn_val,  ctx.Y_val_lvl)
    rt = ctx.evaluate(mtgnn_test, ctx.Y_test_lvl)

    ctx.report("MTGNN", "Validation", rv)
    ctx.report("MTGNN", "Testing",    rt)

    return {
        "val":  rv,
        "test": rt,
    }
