import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import math
import time
import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import random
from util import DataLoaderS
from random import randrange
from matplotlib import pyplot as plt
from safetensors.torch import save_file, load_file
from safetensors import safe_open
import pandas as pd
import json
import optuna

script_dir = os.path.dirname(os.path.abspath(__file__))
bayesian_base_dir = os.path.join(script_dir, "Bayesian")

def load_model(Data):
    with safe_open(args.o_save, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    arch_json = metadata.get("arch")
    if arch_json:
        arch = json.loads(arch_json)
        # Safe fallback for subtract_last if missing in older checkpoint architectures
        sub_last = arch.get("subtract_last", True)
        model = gtnet(
            arch["gcn_true"],
            arch["buildA_true"],
            arch["gcn_depth"],
            arch["num_nodes"],
            device,
            Data.adj,
            dropout=arch["dropout"],
            subgraph_size=arch["subgraph_size"],
            node_dim=arch["node_dim"],
            dilation_exponential=arch["dilation_exponential"],
            conv_channels=arch["conv_channels"],
            residual_channels=arch["residual_channels"],
            skip_channels=arch["skip_channels"],
            end_channels=arch["end_channels"],
            seq_length=arch["seq_length"],
            in_dim=arch["in_dim"],
            out_dim=arch["out_dim"],
            layers=arch["layers"],
            propalpha=arch["propalpha"],
            tanhalpha=arch["tanhalpha"],
            layer_norm_affline=arch.get("layer_norm_affline", False),
            subtract_last=sub_last, weight_regularizer=arch.get("weight_regularizer", 1e-6), dropout_regularizer=arch.get("dropout_regularizer", 1e-5)
        ).to(device)

        ret = model.load_state_dict(state_dict, strict=False)
        if hasattr(ret, "missing_keys") and ret.missing_keys:
            print("[load_state_dict] missing keys:", ret.missing_keys)
        if hasattr(ret, "unexpected_keys") and ret.unexpected_keys:
            print("[load_state_dict] unexpected keys:", ret.unexpected_keys)
    else:
        print("[warn] No arch metadata in checkpoint. Falling back to partial load (strict=False).")
        msd = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in msd and v.shape == msd[k].shape}
        msd.update(filtered)
        model.load_state_dict(msd, strict=False)
        print(f"[warn] Loaded {len(filtered)} / {len(msd)} tensors by shape-matching.")
    return model


def plot_data(data, title):
    x = range(1, len(data)+1)
    plt.plot(x, data, 'b-', label='Actual')
    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title.upper(), y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Month", fontsize=15)
    plt.xticks(rotation='vertical', fontsize=13) 
    plt.yticks(fontsize=13)
    plt.show()


def consistent_name(name):
    if name == 'CAPTCHA' or name == 'DNSSEC' or name == 'RRAM':
        return name

    if not name.isupper():
        words = name.split(' ')
        result = ''
        for i, word in enumerate(words):
            if len(word) <= 2: 
                result += word
            else:
                result += word[0].upper() + word[1:]
            
            if i < len(words) - 1:
                result += ' '
        return result

    words = name.split(' ')
    result = ''
    for i, word in enumerate(words):
        if len(word) <= 3 or '/' in word or word == 'MITM' or word == 'SIEM':
            result += word
        else:
            result += word[0] + (word[1:].lower())
        
        if i < len(words) - 1:
            result += ' '
    return result


def save_metrics_1d(predict, test, title, type):
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    root_sum_squared = math.sqrt(sum_squared_diff)

    sum_absolute_diff = torch.sum(torch.abs(test - predict))

    test_s = test
    mean_all = torch.mean(test_s)
    diff_r = test_s - mean_all
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r)

    eps = 1e-8
    rrse = root_sum_squared / (root_sum_squared_r + eps)

    sum_absolute_r = torch.sum(torch.abs(diff_r))
    rae = sum_absolute_diff / (sum_absolute_r + eps)
    rae = rae.item()

    mse = torch.mean(torch.pow(test - predict, 2)).item()
    mae = torch.mean(torch.abs(test - predict)).item()
    rmse = math.sqrt(mse)

    title = title.replace('/', '_')
    out_dir = os.path.join(bayesian_base_dir, f"{args.version}/{type}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{title.upper()}_{type}.txt'), "w") as f:
        f.write('rse:' + str(rrse) + '\n')
        f.write('rae:' + str(rae) + '\n')
        f.write('mse:' + str(mse) + '\n')
        f.write('mae:' + str(mae) + '\n')
        f.write('rmse:' + str(rmse) + '\n')


def generate_month_labels(start: str, end: str):
    start_date = pd.to_datetime(start, format='%b-%y', errors='coerce')
    if pd.isna(start_date):
        start_date = pd.to_datetime(start, format='%b-%Y')
    end_date = pd.to_datetime(end, format='%b-%y', errors='coerce')
    if pd.isna(end_date):
        end_date = pd.to_datetime(end, format='%b-%Y')

    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    return [d.strftime('%b-%y') for d in dates]


def plot_predicted_actual(predicted, actual, title, type, variance, confidence_95, M):
    M2 = []
    p = []

    for index, value in enumerate(M):
        if 'Dec' in value or 'Mar' in value or 'Jun' in value or 'Sep' in value:
            M2.append(value)
            p.append(index + 1)

    pred_np = predicted.detach().cpu().numpy() if torch.is_tensor(predicted) else predicted
    act_np = actual.detach().cpu().numpy() if torch.is_tensor(actual) else actual
    conf_np = confidence_95.detach().cpu().numpy() if torch.is_tensor(confidence_95) else confidence_95

    x = range(1, len(pred_np) + 1)
    plt.plot(x, act_np, 'b-', label='Actual')
    plt.plot(x, pred_np, '--', color='purple', label='Predicted')
    plt.fill_between(x, pred_np - conf_np, pred_np + conf_np, alpha=0.5, color='pink', label='95% Confidence')
    
    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title.upper(), y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Month", fontsize=15)
    plt.xticks(p, M2, rotation=45, fontsize=10)
    plt.yticks(fontsize=13)
    
    out_dir = os.path.join(bayesian_base_dir, f"{args.version}/{type}")
    os.makedirs(out_dir, exist_ok=True)
    title_clean = title.replace('/', '_')
    plt.savefig(os.path.join(out_dir, f'{title_clean.upper()}_{type}.png'), bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f'{title_clean.upper()}_{type}.pdf'), bbox_inches="tight", format='pdf')
    plt.close()


def s_mape(yTrue, yPred):
    mape = 0
    for i in range(len(yTrue)):
        denominator = (abs(yTrue[i]) + abs(yPred[i]))
        if denominator == 0:
            denominator = 1e-8
        mape += abs(yTrue[i] - yPred[i]) / denominator
    mape /= len(yTrue)
    return mape


def mape(yTrue, yPred):
    eps = 1e-8
    yTrue = np.array(yTrue)
    yPred = np.array(yPred)
    return np.mean(np.abs((yTrue - yPred) / (np.abs(yTrue) + eps)))


def compute_mase(predict_np, true_np, train_rawdat, node_idx, feature_idx=0, h=12):
    train_series = train_rawdat[:, node_idx, feature_idx] 
    if len(train_series) <= h:
        return np.nan
    mae_naive_h = np.mean(np.abs(train_series[h:] - train_series[:-h]))
    eps = 1e-8

    pred_flat = predict_np[:, :, node_idx].flatten()
    true_flat = true_np[:, :, node_idx].flatten()
    mae_forecast = np.mean(np.abs(true_flat - pred_flat))
    return mae_forecast / (mae_naive_h + eps)


def evaluate_direct(data, X_, Y_, tf, model, evaluateL2, evaluateL1, batch_size, is_plot, type_name="Testing", z=1.96):
    model.eval()
    model.mc_dropout = True  # Enable proper test-time MC Dropout ensembling
    predict = None
    test = None
    variance = None
    confidence_95 = None
    num_runs = 10
    
    for X, Y in data.get_batches(X_, Y_, batch_size, False):
        X = X.permute(0, 3, 2, 1).float().to(device)
        Y = Y.to(device)
        
        mus = []
        sigmas = []
        with torch.no_grad():
            for _ in range(num_runs):
                mu, sigma = model(X)
                mu = torch.squeeze(mu, 3) 
                sigma = torch.squeeze(sigma, 3)
                if len(mu.shape) == 2:
                    mu = mu.unsqueeze(0)
                    sigma = sigma.unsqueeze(0)
                mus.append(mu)
                sigmas.append(sigma)
                
        mus = torch.stack(mus)
        sigmas = torch.stack(sigmas)
        
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        mean_pred = torch.mean(mus, dim=0)
        epistemic_var = torch.var(mus, dim=0)
        aleatoric_var = torch.mean(sigmas**2, dim=0)
        total_var = epistemic_var + aleatoric_var
        std_pred = torch.sqrt(total_var)
        conf_int = z * std_pred
        
        Y_target = Y[:, :, :, 0]
        
        target_scale = data.scale[:, 0]
        scale = target_scale.expand(Y_target.size(0), Y_target.size(1), data.m).to(Y_target.device)
        
        mean_pred = mean_pred * scale
        Y_target = Y_target * scale
        var_pred = total_var * scale
        conf_int = conf_int * scale
        
        if predict is None:
            predict = mean_pred
            test = Y_target
            variance = var_pred
            confidence_95 = conf_int
        else:
            predict = torch.cat((predict, mean_pred))
            test = torch.cat((test, Y_target))
            variance = torch.cat((variance, var_pred))
            confidence_95 = torch.cat((confidence_95, conf_int))
            
    num_nodes = data.m
    eps = 1e-8
    
    # Check if we are running differenced data
    is_diff = "diff" in args.data.lower()
    
    if is_diff:
        # Load original raw level data for inverse differencing reference
        orig_data = np.load('data_preparation/data/5_bmtgnn_input/bmtgnn_data.npy')
        
        # De-scale predictions and targets back to raw differences
        predict_diff_raw = predict.data.cpu().numpy()
        target_diff_raw = test.data.cpu().numpy()
        
        n_time = orig_data.shape[1]
        num_windows = predict_diff_raw.shape[0]
        seq_out_len = data.out_len
        
        reconstructed_preds = np.zeros((num_windows, seq_out_len, num_nodes))
        actual_levels = np.zeros((num_windows, seq_out_len, num_nodes))
        
        for w in range(num_windows):
            base_idx = n_time - 12 + w - 1
            base_level = orig_data[:, base_idx, 0] # [30]
            
            current_pred_level = base_level.copy()
            for step in range(seq_out_len):
                current_pred_level = current_pred_level + predict_diff_raw[w, step, :]
                reconstructed_preds[w, step, :] = current_pred_level
                actual_levels[w, step, :] = orig_data[:, base_idx + 1 + step, 0]
                
        # Override predict_np & Ytest_np with reconstructed physical levels
        predict_np = reconstructed_preds
        Ytest_np = actual_levels
    else:
        predict_np = predict.data.cpu().numpy()
        Ytest_np = test.data.cpu().numpy()

    node_rse_list = []
    node_rae_list = []
    node_mase_list = []
    node_mse_list = []
    node_mae_list = []
    node_rmse_list = []
    node_r2_list = []

    # Use actual training start indices to determine the exact end of training data
    n_train = data.train_starts[-1] + data.out_len + 1
    # Use original raw level training data for MASE calculation
    if is_diff:
        orig_data = np.load('data_preparation/data/5_bmtgnn_input/bmtgnn_data.npy')
        train_rawdat = orig_data[:n_train, :, :]
    else:
        train_rawdat = data.rawdat[:n_train, :, :]

    for node in range(num_nodes):
        pred_node = predict_np[:, :, node].flatten()
        true_node = Ytest_np[:, :, node].flatten()

        sum_sq_err = np.sum((true_node - pred_node) ** 2)
        mean_true = np.mean(true_node)
        sum_sq_dev = np.sum((true_node - mean_true) ** 2)
        node_rse = np.sqrt(sum_sq_err / (sum_sq_dev + eps))
        node_rse_list.append(node_rse)

        # Standard R2 Score calculation: 1 - (RSS / TSS)
        node_r2 = 1 - (sum_sq_err / (sum_sq_dev + eps))
        node_r2_list.append(node_r2)

        sum_abs_err = np.sum(np.abs(true_node - pred_node))
        sum_abs_dev = np.sum(np.abs(true_node - mean_true))
        node_rae = sum_abs_err / (sum_abs_dev + eps)
        node_rae_list.append(node_rae)

        mase_val = compute_mase(predict_np, Ytest_np, train_rawdat, node,
                                feature_idx=0, h=predict_np.shape[1])
        node_mase_list.append(mase_val)

        mse_val = np.mean((true_node - pred_node) ** 2)
        mae_val = np.mean(np.abs(true_node - pred_node))
        rmse_val = math.sqrt(mse_val)
        
        node_mse_list.append(mse_val)
        node_mae_list.append(mae_val)
        node_rmse_list.append(rmse_val)

    avg_node_rse = np.mean(node_rse_list)
    avg_node_r2 = np.mean(node_r2_list)
    avg_node_rae = np.mean(node_rae_list)
    avg_node_mase = np.mean(node_mase_list)
    avg_node_mse = np.mean(node_mse_list)
    avg_node_mae = np.mean(node_mae_list)
    avg_node_rmse = np.mean(node_rmse_list)

    train_std = np.std(train_rawdat[:, :, 0], axis=0)
    node_hist_rse_list = []
    for node in range(num_nodes):
        rmse = np.sqrt(np.mean((Ytest_np[:, :, node].flatten() -
                                predict_np[:, :, node].flatten()) ** 2))
        node_hist_rse_list.append(rmse / (train_std[node] + eps))
    avg_hist_rse = np.mean(node_hist_rse_list)

    # Global RSE matching Level Domain
    sum_squared_diff = np.sum((Ytest_np - predict_np) ** 2)
    sum_absolute_diff = np.sum(np.abs(Ytest_np - predict_np))
    root_sum_squared = math.sqrt(sum_squared_diff)
    mean_all = np.mean(Ytest_np, axis=0)
    diff_r = Ytest_np - mean_all[np.newaxis, :, :]
    sum_squared_r = np.sum(diff_r ** 2)
    global_rrse = root_sum_squared / (math.sqrt(sum_squared_r) + eps)

    rrse = avg_node_rse
    rae = avg_node_rae

    sigma_p = predict_np.std(axis=(0, 1))
    sigma_g = Ytest_np.std(axis=(0, 1))
    mean_p = predict_np.mean(axis=(0, 1))
    mean_g = Ytest_np.mean(axis=(0, 1))
    corr_list = []
    for node in range(num_nodes):
        if sigma_g[node] != 0 and sigma_p[node] != 0:
            c = ((predict_np[:, :, node] - mean_p[node]) *
                 (Ytest_np[:, :, node] - mean_g[node])).mean() / (sigma_p[node] * sigma_g[node])
            corr_list.append(c)
    correlation = np.mean(corr_list) if corr_list else 0.0

    smape_list = [s_mape(Ytest_np[:, :, node].flatten(),
                         predict_np[:, :, node].flatten())
                 for node in range(num_nodes)]
    smape = np.mean(smape_list)

    mape_list = [mape(Ytest_np[:, :, node].flatten(),
                      predict_np[:, :, node].flatten())
                 for node in range(num_nodes)]
    avg_mape = np.mean(mape_list)

    # Calculate PICP and MPIW globally
    lb = predict_np - confidence_95.data.cpu().numpy()
    ub = predict_np + confidence_95.data.cpu().numpy()
    picp = np.mean((Ytest_np >= lb) & (Ytest_np <= ub))
    mpiw = np.mean(ub - lb)

    print(f"[{type_name}] "
          f"RSE(window): {avg_node_rse:.4f} | "
          f"R2-Score: {avg_node_r2:.4f} | "
          f"MASE: {avg_node_mase:.4f} | "
          f"SMAPE: {smape:.4f} | "
          f"MAPE: {avg_mape:.4f} | "
          f"MSE: {avg_node_mse:.4f} | "
          f"MAE: {avg_node_mae:.4f} | "
          f"RMSE: {avg_node_rmse:.4f} | "
          f"RAE: {avg_node_rae:.4f} | "
          f"Hist-RSE: {avg_hist_rse:.4f} | "
          f"Global-RSE: {global_rrse:.4f} | "
          f"Corr: {correlation:.4f} | "
          f"PICP: {picp:.4f} | "
          f"MPIW: {mpiw:.4f}")

    if is_plot:
        # Determine plotting directories
        out_dir = os.path.join(bayesian_base_dir, f"{args.version}/{type_name}")
        os.makedirs(out_dir, exist_ok=True)
        
        # Consistent trajectory plotting: Use the last sliding window of size `data.out_len`
        w_last = predict_np.shape[0] - 1
        dates_fmt = [pd.to_datetime(d).strftime('%b-%y') for d in tf[-1]]
        
        for node in range(num_nodes):
            nod_name = data.col[node].replace('-ALL', '').replace('ALL', '').strip()
            nod_name = consistent_name(nod_name)
            
            pred_node = predict_np[w_last, :, node] # [6] or [3]
            act_node = Ytest_np[w_last, :, node] # [6] or [3]
            conf_node = confidence_95[w_last, :, node].cpu().numpy()  # 95% confidence interval for this node

            # Save 1D metric
            save_metrics_1d(torch.from_numpy(pred_node), torch.from_numpy(act_node), nod_name, type_name)
            
            plt.figure(figsize=(10, 5))
            x_axis = range(1, data.out_len + 1)
            
            plt.plot(x_axis, act_node, 'b-', linewidth=2, label='Actual AFCI Level')
            plt.plot(x_axis, pred_node, '--', color='purple', linewidth=2, label='Predicted AFCI Level')
            
            # Fan Chart Gradient for Uncertainty
            std_node = conf_node / 1.96
            conf_95 = std_node * 1.96
            conf_80 = std_node * 1.28
            conf_68 = std_node * 1.00
            
            plt.fill_between(x_axis, pred_node - conf_95, pred_node + conf_95, alpha=0.15, color='hotpink', label='95% Confidence', edgecolor='none')
            plt.fill_between(x_axis, pred_node - conf_80, pred_node + conf_80, alpha=0.20, color='hotpink', label='80% Confidence', edgecolor='none')
            plt.fill_between(x_axis, pred_node - conf_68, pred_node + conf_68, alpha=0.25, color='hotpink', label='68% Confidence', edgecolor='none')
            
            plt.legend(loc="best", prop={'size': 11})
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.title(f"{nod_name.upper()} - {data.out_len}M Horizon Forecasting", y=1.03, fontsize=15)
            plt.ylabel("AFCI Value", fontsize=12)
            plt.xlabel("Forecasting Horizon (Month)", fontsize=12)
            plt.xticks(x_axis, dates_fmt, rotation=30, fontsize=10)
            plt.yticks(fontsize=11)
            
            title_clean = nod_name.replace('/', '_').replace(' ', '_')
            plt.savefig(os.path.join(out_dir, f'{title_clean.upper()}_{type_name}.png'), bbox_inches="tight")
            plt.savefig(os.path.join(out_dir, f'{title_clean.upper()}_{type_name}.pdf'), bbox_inches="tight", format='pdf')
            plt.close()
    model.mc_dropout = False  # Reset flag to avoid side effects
    return rrse, rae, correlation, smape, avg_node_mase, avg_node_mse, avg_node_mae, avg_node_rmse, avg_node_r2, avg_mape, avg_hist_rse, global_rrse, picp, mpiw


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (p.grad * scale).to(p)
                p.add_(e_w)  # climb to local maximum in perturbation direction
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # restore original parameters
        self.base_optimizer.step()  # step on original parameters in gradient direction from peak
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super(SAM, self).load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def gaussian_nll_loss(mu, target, sigma):
    var = sigma ** 2
    loss = 0.5 * torch.log(var + 1e-6) + 0.5 * ((target - mu) ** 2) / (var + 1e-6)
    return loss.mean()

def train(data, X, Y, tf, model, criterion, optimizer, batch_size):
    model.train()
    total_loss = 0
    iter = 0

    for X, Y in data.get_batches(X, Y, batch_size, True):
        X = X.to(device)
        Y = Y.to(device)

        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]

            id = torch.tensor(id).to(device)
            tx = X.permute(0, 3, 2, 1).float().to(device)
            ty = Y[:, :, :, 0].float().to(device) 

            if isinstance(optimizer, SAM):
                # 1. First step: ascent to local maximum of loss sharpness
                optimizer.zero_grad()
                mu, sigma = model(tx)
                mu = torch.squeeze(mu, 3)
                sigma = torch.squeeze(sigma, 3)
                loss = gaussian_nll_loss(mu, ty, sigma) + model.reg_loss
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # 2. Second step: compute loss at perturbed weights, backprop, and step
                mu, sigma = model(tx)
                mu = torch.squeeze(mu, 3)
                sigma = torch.squeeze(sigma, 3)
                loss2 = gaussian_nll_loss(mu, ty, sigma) + model.reg_loss
                loss2.backward()
                
                if args.clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.second_step(zero_grad=True)
                loss_val = loss.item()
            else:
                # Standard Adam Optimization
                optimizer.zero_grad()
                mu, sigma = model(tx)           
                mu = torch.squeeze(mu, 3)
                sigma = torch.squeeze(sigma, 3)
                loss = gaussian_nll_loss(mu, ty, sigma) + model.reg_loss
                loss.backward()
                
                if args.clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                loss_val = loss.item()

            total_loss += loss_val

        if iter % 1 == 0:
            print('iter:{:3d} | loss: {:.6f}'.format(iter, loss_val))
        iter += 1
    return total_loss / iter


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def current_device_of_model(model):
    return next(model.parameters()).device


def resolve_device(dev_str: str) -> torch.device:
    dev_str = (dev_str or "").lower().strip()
    if dev_str in ("", "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev_str == "cpu":
        return torch.device("cpu")
    if dev_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[warn] CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        if dev_str == "cuda":
            return torch.device("cuda:0")
        parts = dev_str.split(":")
        if len(parts) == 2:
            try:
                idx = int(parts[1])
            except ValueError:
                idx = 0
        else:
            idx = 0
        count = torch.cuda.device_count()
        if idx < 0 or idx >= count:
            if count > 0:
                return torch.device("cuda:0")
            else:
                return torch.device("cpu")
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


def main(experiment):
    set_random_seed(fixed_seed)
    # gcn_depth: 2 dominant in top-20 (13/20), avg(2)=1.203 vs avg(1)=1.232
    gcn_depths    = [1, 2]
    # lr: 0.0003 best avg (1.199), 0.0008 strong in top-20 (13/20); 0.001 & 0.0002 removed
    lrs           = [0.0003, 0.0005, 0.0008]
    # conv: 8 & 12 essentially tied (~1.212 avg); 16 removed (1.316); 10 removed (bug)
    convs         = [8, 12]
    # res: 8 clearly best avg (1.199); 12/16 ok; 24 removed (1.307); 10 removed (bug)
    ress          = [8, 12, 16]
    # skip: 16 best avg (1.199), 24 next (1.215), 32 ok (1.220); 20 & 48 removed
    skips         = [16, 24, 32]
    # end: 32 best avg (1.203), 64 & 48 ok; 96 removed (1.315)
    ends          = [32, 48, 64]
    # k (subgraph_size): 5 dominant in top-20 (13/20); 9 best mean (1.215); expanded to 15, 20 for adaptive graph from scratch
    ks            = [5, 7, 9, 15, 20]
    # dropout: 0.6 dominant in top-20 (13/20); 0.4 best mean (1.208); all kept
    dropouts      = [0.4, 0.5, 0.6]
    # dilation_ex: 1 overwhelmingly dominant in top-20 (18/20) ??weight search toward 1
    dilation_exs  = [1, 1, 2]   # doubled weight on 1 via duplication
    # node_dim: expanded search space up to 64 for purely adaptive graph learning
    node_dims     = [10, 20, 30, 40, 64]
    # prop_alpha: 0.1 best avg (1.201), 0.2 close (1.204); 0.001/0.01 removed
    prop_alphas   = [0.05, 0.1, 0.2]
    # tanh_alpha: 0.1 best avg (1.212); 0.05 removed (avg 1.300, 2 terrible samples)
    tanh_alphas   = [0.1, 1, 3]
    # layers: 2 & 3 essentially tied; both kept
    layers_list   = [2, 3]

    best_val = 10000000
    best_rse = 10000000
    best_rae = 10000000
    best_corr = -10000000
    best_smape = 10000000
    
    best_test_rse = 10000000
    best_test_corr = -10000000

    best_hp = []
    if os.path.exists(args.hp_path):
        try:
            with open(args.hp_path, "r") as f:
                content = f.read().strip()
                if content and content != '[]':
                    best_hp = eval(content)
                    print(f"Loaded best hyperparameters from {args.hp_path}: {best_hp}")
        except Exception as e:
            print(f"Warning: Could not load {args.hp_path}: {e}. Starting with empty hp.")

    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for attr in ['data', 'graph_file', 'nodes_file']:
        val = getattr(args, attr)
        if val and not os.path.isabs(val):
            if not os.path.exists(val):
                abs_val = os.path.join(workspace_root, val)
                if os.path.exists(abs_val):
                    setattr(args, attr, abs_val)
                    print(f"{attr}: {abs_val}")

    Data = DataLoaderS(
        args.data, 
        args.train_ratio, 
        args.valid_ratio, 
        device, 
        args.horizon, 
        args.seq_in_len, 
        args.graph_file,
        args.normalize, 
        args.seq_out_len,
        nodes_file=args.nodes_file
    )
    args.in_dim = Data.f
    args.num_nodes = Data.m

    # Dynamically calculate ConcreteDropout regularizers based on dataset size N
    # N = number of training windows * number of nodes
    try:
        train_windows = len(Data.train[0])
    except:
        train_windows = 1000 # fallback
    N = train_windows * args.num_nodes
    args.dropout_regularizer = 2.0 / max(N, 1)
    args.weight_regularizer = (1e-2) / max(N, 1)
    print(f"[Dynamic Setting] N = {N} (windows: {train_windows}, nodes: {args.num_nodes})")
    print(f" -> dropout_regularizer = {args.dropout_regularizer:.2e}")
    print(f" -> weight_regularizer = {args.weight_regularizer:.2e}")

    os.makedirs(os.path.join(bayesian_base_dir, f"{args.version}/Validation"), exist_ok=True)
    os.makedirs(os.path.join(bayesian_base_dir, f"{args.version}/Testing"), exist_ok=True)

    evaluateL2 = nn.MSELoss(reduction='mean').to(device) 
    evaluateL1 = nn.L1Loss(reduction='mean').to(device) 

    study = optuna.create_study(direction="minimize")
    for q in range(args.search_iters):
        trial = study.ask()
        gcn_depth = trial.suggest_categorical("gcn_depth", gcn_depths)
        lr = trial.suggest_categorical("lr", lrs)
        conv = trial.suggest_categorical("conv", convs)
        res = trial.suggest_categorical("res", ress)
        skip = trial.suggest_categorical("skip", skips)
        end = trial.suggest_categorical("end", ends)
        layer = trial.suggest_categorical("layer", layers_list)
        k = trial.suggest_categorical("k", ks)
        dropout = trial.suggest_categorical("dropout", dropouts)
        dilation_ex = trial.suggest_categorical("dilation_ex", dilation_exs)
        node_dim = trial.suggest_categorical("node_dim", node_dims)
        prop_alpha = trial.suggest_categorical("prop_alpha", prop_alphas)
        tanh_alpha = trial.suggest_categorical("tanh_alpha", tanh_alphas)
        # Exploitation phase: subtract_last=True is fixed (always best performing)
        subtract_last = True
        
        iter_best_sum = 0.0
        iter_best_rse = 0.0
        iter_best_rae = 0.0
        iter_best_corr = 0.0
        iter_best_smape = 0.0
        iter_best_test_rse = 0.0
        iter_best_test_corr = 0.0
        iter_best_epoch = 0
        iter_has_valid_model = False
        
        iter_best_mse = 0.0
        iter_best_mae = 0.0
        iter_best_rmse = 0.0
        iter_best_r2 = 0.0
        iter_best_mase = 0.0
        iter_best_mape = 0.0
        iter_best_hist_rse = 0.0
        iter_best_global_rse = 0.0
        iter_test_mse = 0.0
        iter_test_mae = 0.0
        iter_test_rmse = 0.0
        iter_test_r2 = 0.0
        iter_test_mase = 0.0
        iter_test_mape = 0.0
        iter_test_hist_rse = 0.0
        iter_test_global_rse = 0.0
        iter_best_test_mse = 0.0
        iter_best_test_mae = 0.0
        iter_best_test_rmse = 0.0
        iter_best_test_r2 = 0.0
        iter_best_test_mase = 0.0
        iter_best_test_smape = 0.0
        iter_best_test_mape = 0.0
        iter_best_test_rae = 0.0
        iter_best_test_hist_rse = 0.0
        iter_best_test_global_rse = 0.0

        sum_loss = 0.0
        val_loss = 0.0
        val_rae = 0.0
        val_corr = 0.0
        val_smape = 0.0
        epoch = 0

        print('train X:', Data.train[0].shape)
        print('train Y:', Data.train[1].shape)
        print('valid X:', Data.valid[0].shape)
        print('valid Y:', Data.valid[1].shape)
        print('test X:', Data.test[0].shape)
        print('test Y:', Data.test[1].shape)

        model = gtnet(args.gcn_true, args.buildA_true, gcn_depth, args.num_nodes,
                    device, Data.adj, dropout=dropout, subgraph_size=k,
                    node_dim=node_dim, dilation_exponential=dilation_ex,
                    conv_channels=conv, residual_channels=res,
                    skip_channels=skip, end_channels=end,
                    seq_length=args.seq_in_len,
                    in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha, layer_norm_affline=False,
                    subtract_last=subtract_last, weight_regularizer=args.weight_regularizer, dropout_regularizer=args.dropout_regularizer).to(device)
        
        print(args)
        print('The receptive field size is', model.receptive_field)
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of model parameters is', nParams, flush=True)

        # Force Smooth L1 Loss (Huber Loss) to effectively minimize both MAE and MSE
        criterion = nn.SmoothL1Loss(reduction='mean').to(device)

        # Standard PyTorch Adam optimizer & Plateau learning rate decay scheduler
        if getattr(args, 'use_sam', True):
            optimizer = SAM(model.parameters(), torch.optim.Adam, rho=0.05, lr=lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
            
        # patience=15: LR halves if val_rse doesn't improve for 15 epochs (increased for adaptive graph from scratch)
        scheduler_opt = optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(scheduler_opt, mode='min', factor=0.5, patience=15, threshold=1e-4)
        es_patience = 30  # Early stop if no improvement for 30 epochs
        
        es_counter = 0 
        best_iter_val_loss = 10000000.0
        
        try:
            print('\n\nbegin training')
            for epoch in range(1, args.epochs + 1):
                print('Experiment:', (experiment+1))
                print('Iter:', q)
                print('epoch:', epoch)
                print('hp=', [gcn_depth, lr, conv, res, skip, end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch, subtract_last])
                print('best sum=', best_val)
                print('best rrse=', best_rse)
                print('best rrae=', best_rae)
                print('best corr=', best_corr)
                print('best smape=', best_smape)       
                print('best hps=', best_hp)
                print('best test rse=', best_test_rse)
                print('best test corr=', best_test_corr)
                
                es_counter += 1

                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], Data.train[2], model, criterion, optimizer, args.batch_size)
                
                # Evaluate direct metrics
                val_loss, val_rae, val_corr, val_smape, val_mase, val_mse, val_mae, val_rmse, val_r2, val_mape, val_hist_rse, val_global_rse, val_picp, val_mpiw = evaluate_direct(
                                                 Data, Data.valid[0], Data.valid[1], Data.valid[2], model, evaluateL2, evaluateL1,
                                                 args.batch_size, False, type_name='Validation')
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid mase {:5.4f} | valid rae {:5.4f} | valid corr {:5.4f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_mase, val_rae, val_corr), flush=True)
                
                # Decay learning rate when validation RSE plateaus
                scheduler.step(val_loss)
                
                # Bayesian Composite Objective
                # NLL (Negative Log-Likelihood) is adopted as a Strictly Proper Scoring Rule.
                # It perfectly penalizes point prediction errors while preventing improperly calibrated uncertainty bounds.
                sum_loss = val_loss

                # Tracking metrics at the iteration level (Burn-in period: 20 epochs to prevent selecting underfitted early models)
                if epoch > 20 and (not math.isnan(val_corr)) and (iter_best_sum == 0.0 or sum_loss < iter_best_sum):
                    iter_best_sum = sum_loss
                    iter_best_rse = val_loss
                    iter_best_rae = val_rae
                    iter_best_corr = val_corr
                    iter_best_smape = val_smape
                    iter_best_epoch = epoch
                    iter_has_valid_model = True
                    iter_best_mse = val_mse
                    iter_best_mae = val_mae
                    iter_best_rmse = val_rmse
                    iter_best_r2 = val_r2
                    iter_best_mase = val_mase
                    iter_best_mape = val_mape
                    iter_best_hist_rse = val_hist_rse
                    iter_best_global_rse = val_global_rse
                    
                    iter_test_acc, iter_test_rae, iter_test_corr, iter_test_smape, iter_test_mase, iter_test_mse, iter_test_mae, iter_test_rmse, iter_test_r2, iter_test_mape, iter_test_hist_rse, iter_test_global_rse, iter_test_picp, iter_test_mpiw = evaluate_direct(
                        Data, Data.test[0], Data.test[1], Data.test[2], model, evaluateL2, evaluateL1,
                        args.batch_size, False, type_name='Testing'
                    )
                    iter_best_test_rse = iter_test_acc
                    iter_best_test_corr = iter_test_corr
                    iter_best_test_mse = iter_test_mse
                    iter_best_test_mae = iter_test_mae
                    iter_best_test_rmse = iter_test_rmse
                    iter_best_test_r2 = iter_test_r2
                    iter_best_test_mase = iter_test_mase
                    iter_best_test_smape = iter_test_smape
                    iter_best_test_mape = iter_test_mape
                    iter_best_test_rae = iter_test_rae
                    iter_best_test_hist_rse = iter_test_hist_rse
                    iter_best_test_global_rse = iter_test_global_rse

                # If this is the absolute best validation model based on sum_loss, save weights and parameters
                if epoch > 20 and (not math.isnan(val_corr)) and sum_loss < best_val:
                    arch_meta = {
                        "gcn_true": args.gcn_true,
                        "buildA_true": args.buildA_true,
                        "gcn_depth": gcn_depth,
                        "num_nodes": args.num_nodes,
                        "dropout": dropout,
                        "subgraph_size": k,
                        "node_dim": node_dim,
                        "dilation_exponential": dilation_ex,
                        "conv_channels": conv,
                        "residual_channels": res,
                        "skip_channels": skip,
                        "end_channels": end,
                        "seq_length": args.seq_in_len,
                        "in_dim": args.in_dim,
                        "out_dim": args.seq_out_len,
                        "layers": layer,
                        "propalpha": prop_alpha,
                        "tanhalpha": tanh_alpha,
                        "layer_norm_affline": False,
                        "subtract_last": subtract_last
                    }
                    save_file(model.state_dict(), args.o_save, metadata={"arch": json.dumps(arch_meta)})
                    print(f"Saved safetensors file to {args.o_save}, size = {os.path.getsize(args.o_save)} bytes")

                    best_val = sum_loss
                    best_rse = val_loss
                    best_rae = val_rae
                    best_corr = val_corr
                    best_smape = val_smape

                    best_hp = [
                        gcn_depth, lr, conv, res, 
                        skip, end, k, dropout, 
                        dilation_ex, node_dim, prop_alpha, tanh_alpha, 
                        layer, epoch, subtract_last
                    ]
                    
                    with open(args.hp_path, "w") as f:
                        f.write(str(best_hp))
                    print(f"New best HP found and saved to {args.hp_path}: {best_hp}")
                    
                    test_acc, test_rae, test_corr, test_smape, test_mase, test_mse, test_mae, test_rmse, test_r2, test_mape, test_hist_rse, test_global_rse, test_picp, test_mpiw = evaluate_direct(
                        Data, Data.test[0], Data.test[1], Data.test[2], model, evaluateL2, evaluateL1,
                        args.batch_size, False, type_name='Testing'
                    ) 
                    print('*' * 100)
                    print("test rse {:5.4f} | test mase {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(
                        test_acc, test_mase, test_rae, test_corr), flush=True)
                    print('*' * 100)
                    best_test_rse = test_acc
                    best_test_corr = test_corr
                
                # Active Early Stopping Check
                if sum_loss < best_iter_val_loss:
                    best_iter_val_loss = sum_loss
                    es_counter = 0
                
                # Active Early Stopping
                if epoch >= 10 and es_counter >= es_patience:
                    print(f"[Early Stopping] No improvement for {es_patience} epochs (Counter: {es_counter}). Stopping at epoch {epoch}.")
                    break


        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        log_file_path = os.path.join(bayesian_base_dir, f"{args.version}/search_results.txt")
        csv_file_path = os.path.join(bayesian_base_dir, f"{args.version}/search_results.csv")
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_exists = os.path.exists(log_file_path)
        csv_exists = os.path.exists(csv_file_path)
        
        if not iter_has_valid_model:
            iter_best_sum = sum_loss
            iter_best_rse = val_loss
            iter_best_rae = val_rae
            iter_best_corr = val_corr
            iter_best_smape = val_smape
            iter_best_epoch = epoch
            iter_best_mse = 0.0
            iter_best_mae = 0.0
            iter_best_rmse = 0.0
            iter_best_r2 = 0.0
            iter_best_mase = 0.0
            iter_best_mape = 0.0
            iter_best_hist_rse = 0.0
            iter_best_global_rse = 0.0
            try:
                iter_test_acc, iter_test_rae, iter_test_corr, iter_test_smape, iter_test_mase, iter_test_mse, iter_test_mae, iter_test_rmse, iter_test_r2, iter_test_mape, iter_test_hist_rse, iter_test_global_rse, iter_test_picp, iter_test_mpiw = evaluate_direct(
                    Data, Data.test[0], Data.test[1], Data.test[2], model, evaluateL2, evaluateL1,
                    args.batch_size, False, type_name='Testing'
                )
                iter_best_test_rse = iter_test_acc
                iter_best_test_corr = iter_test_corr
                iter_best_test_mse = iter_test_mse
                iter_best_test_mae = iter_test_mae
                iter_best_test_rmse = iter_test_rmse
                iter_best_test_r2 = iter_test_r2
                iter_best_test_mase = iter_test_mase
                iter_best_test_smape = iter_test_smape
                iter_best_test_mape = iter_test_mape
                iter_best_test_rae = iter_test_rae
                iter_best_test_hist_rse = iter_test_hist_rse
                iter_best_test_global_rse = iter_test_global_rse
            except Exception as e:
                print(f"[warn] Failed to evaluate test window for fallback: {e}")
                iter_best_test_rse = 0.0
                iter_best_test_corr = 0.0
                iter_best_test_mse = 0.0
                iter_best_test_mae = 0.0
                iter_best_test_rmse = 0.0
                iter_best_test_r2 = 0.0
                iter_best_test_mase = 0.0
                iter_best_test_smape = 0.0
                iter_best_test_mape = 0.0
                iter_best_test_rae = 0.0
                iter_best_test_hist_rse = 0.0
                iter_best_test_global_rse = 0.0
        else:
            iter_best_mse = val_mse
            iter_best_mae = val_mae
            iter_best_rmse = val_rmse
            iter_best_r2 = val_r2
            iter_best_mase = val_mase
            iter_best_mape = val_mape
            iter_best_hist_rse = val_hist_rse
            iter_best_global_rse = val_global_rse
            iter_best_test_mse = iter_test_mse
            iter_best_test_mae = iter_test_mae
            iter_best_test_rmse = iter_test_rmse
            iter_best_test_r2 = iter_test_r2
            iter_best_test_mase = iter_test_mase
            iter_best_test_smape = iter_test_smape
            iter_best_test_mape = iter_test_mape
            iter_best_test_rae = iter_test_rae
            iter_best_test_hist_rse = iter_test_hist_rse
            iter_best_test_global_rse = iter_test_global_rse
        
        hps_str = f"[{gcn_depth}, {lr}, {conv}, {res}, {skip}, {end}, {k}, {dropout}, {dilation_ex}, {node_dim}, {prop_alpha}, {tanh_alpha}, {layer}, {subtract_last}]"
        
        header_cols = [
            "Iter", "hps", "sum", "rrse", "rrae", "corr", "smape",
            "valid rse", "valid r2", "valid mase", "valid smape", "valid mape", "valid mse", "valid mae", "valid rmse", "valid rae", "valid hist_rse", "valid global_rse", "valid corr",
            "test rse", "test r2", "test mase", "test smape", "test mape", "test mse", "test mae", "test rmse", "test rae", "test hist_rse", "test global_rse", "test corr",
            "best_epoch"
        ]
        
        vals = [
            q, hps_str, iter_best_sum, iter_best_rse, iter_best_rae, iter_best_corr, iter_best_smape,
            iter_best_rse, iter_best_r2, iter_best_mase, iter_best_smape, iter_best_mape, iter_best_mse, iter_best_mae, iter_best_rmse, iter_best_rae, iter_best_hist_rse, iter_best_global_rse, iter_best_corr,
            iter_best_test_rse, iter_best_test_r2, iter_best_test_mase, iter_best_test_smape, iter_best_test_mape, iter_best_test_mse, iter_best_test_mae, iter_best_test_rmse, iter_best_test_rae, iter_best_test_hist_rse, iter_best_test_global_rse, iter_best_test_corr,
            iter_best_epoch
        ]
        
        txt_str = "\t".join([f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals]) + "\n"
        csv_str = ",".join([f'"{v}"' if isinstance(v, str) and ',' in v else (f"{v:.6f}" if isinstance(v, float) else str(v)) for v in vals]) + "\n"
        
        with open(log_file_path, "a") as f_log:
            if not file_exists: f_log.write("\t".join(header_cols) + "\n")
            f_log.write(txt_str)
            
        with open(csv_file_path, "a") as f_csv:
            if not csv_exists: f_csv.write(",".join(header_cols) + "\n")
            f_csv.write(csv_str)
        print(f"Logged iteration {q} results to {log_file_path} and {csv_file_path}")
        study.tell(trial, iter_best_sum)

    if args.search_iters == 0:
        try:
            print(f"Attempting to load best saved model from {args.o_save}...")
            model = load_model(Data)
            model = model.to(device)
            print("Successfully loaded model and weights!")
        except Exception as e:
            print(f"Direct load failed: {e}. Trying hyperparameter fallback...")
            if not best_hp:
                print("Error: No hyperparameters in hp.txt and search_iters=0. Cannot fallback.")
                return 0,0,0,0,0,0,0,0,0,0,0,0,0,0
            else:
                h = best_hp
                model = gtnet(args.gcn_true, args.buildA_true, h[0], args.num_nodes,
                              device, Data.adj,
                              dropout=h[7], subgraph_size=h[6], node_dim=h[9],
                              dilation_exponential=h[8],
                              conv_channels=h[2], residual_channels=h[3],
                              skip_channels=h[4], end_channels=h[5],
                              seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                              layers=h[12], propalpha=h[10], tanhalpha=h[11],
                              subtract_last=h[14], weight_regularizer=args.weight_regularizer, dropout_regularizer=args.dropout_regularizer).to(device)
                try:
                    with safe_open(args.o_save, framework="pt", device="cpu") as f:
                        state_dict = {k: f.get_tensor(k) for k in f.keys()}
                        model.load_state_dict(state_dict, strict=False)
                    print(f"Model weights loaded from {args.o_save} using strict=False fallback.")
                except Exception as ex:
                    print(f"Weights not found in {args.o_save} ({ex}). Using uninitialized model.")
    else:
        with open(args.hp_path, "w") as f:
            f.write(str(best_hp))
        print(f"Best hyperparameters saved to {args.hp_path}")
        
        model = load_model(Data)
        model = model.to(device)

    vtest_acc, vtest_rae, vtest_corr, vtest_smape, vtest_mase, vtest_mse, vtest_mae, vtest_rmse, vtest_r2, vtest_mape, vtest_hist_rse, vtest_global_rse, vtest_picp, vtest_mpiw = evaluate_direct(
        Data, Data.valid[0], Data.valid[1], Data.valid[2], model, evaluateL2, evaluateL1,
        args.batch_size, True, type_name='Validation')

    test_acc, test_rae, test_corr, test_smape, test_mase, test_mse, test_mae, test_rmse, test_r2, test_mape, test_hist_rse, test_global_rse, test_picp, test_mpiw = evaluate_direct(
        Data, Data.test[0], Data.test[1], Data.test[2], model, evaluateL2, evaluateL1,
        args.batch_size, True, type_name='Testing')
    print('*' * 100)
    print("final test rse {:5.4f} | test mase {:5.4f} | test mse {:5.4f} | test mae {:5.4f} | test rmse {:5.4f} | test corr {:5.4f}".format(
        test_acc, test_mase, test_mse, test_mae, test_rmse, test_corr))
    print('*' * 100)
    return vtest_acc, vtest_rae, vtest_corr, vtest_mase, vtest_mse, vtest_mae, vtest_rmse, test_acc, test_rae, test_corr, test_mase, test_mse, test_mae, test_rmse


plt.rcParams['savefig.dpi'] = 1200

from config import get_args
args = get_args()

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

args.seq_out_len = months
args.seq_in_len = args.seq_out_len
args.train_ratio = 0.70 # Dummy value; will be dynamically adjusted train_starts in util.py
args.valid_ratio = 0.15 # Dummy value; will be dynamically adjusted valid_starts in util.py

# Dynamically resolve run version directories
if args.version == "auto":
    base_dir = bayesian_base_dir
    os.makedirs(base_dir, exist_ok=True)
    existing_runs = []
    for d in os.listdir(base_dir):
        if d.startswith("run_"):
            try:
                idx = int(d.split("_")[1])
                existing_runs.append(idx)
            except:
                pass
    if args.search_iters == 0:
        if not existing_runs:
            raise FileNotFoundError("No existing run_X directories found. Please train a model first.")
        latest_idx = max(existing_runs)
        #args.version = f"run_{latest_idx}"
        args.version = "12mo"
        print(f"[Load Mode] search_iters=0. Automatically selected latest run: {args.version}")
    else:
        next_idx = max(existing_runs) + 1 if existing_runs else 0
        args.version = f"run_{next_idx}"

args.o_save = os.path.join(bayesian_base_dir, f"{args.version}/o_model.safetensors")
args.save = os.path.join(bayesian_base_dir, f"{args.version}/model.safetensors")
args.hp_path = os.path.join(bayesian_base_dir, f"{args.version}/hp.txt")

print(f"Run resolved to version: {args.version}")
print(f"Model outputs will be saved to: {os.path.join(bayesian_base_dir, args.version)}")

if args.train_ratio + args.valid_ratio > 1.0:
    raise ValueError(f"train_ratio + valid_ratio must be <= 1.0 "
                    f"(got {args.train_ratio + args.valid_ratio})")

fixed_seed = 123
device = resolve_device(args.device)
torch.set_num_threads(3)

if __name__ == "__main__":
    vacc = []
    vmase = []
    vmse = []
    vmae = []
    vrmse = []
    acc = []
    mase = []
    amse = []
    amae = []
    armse = []
    for i in range(1):
        (val_acc, val_rae, val_corr, val_mase, val_mse, val_mae, val_rmse,
         test_acc, test_rae, test_corr, test_mase, test_mse, test_mae, test_rmse) = main(i)
        vacc.append(val_acc)
        vmase.append(val_mase)
        vmse.append(val_mse)
        vmae.append(val_mae)
        vrmse.append(val_rmse)
        acc.append(test_acc)
        mase.append(test_mase)
        amse.append(test_mse)
        amae.append(test_mae)
        armse.append(test_rmse)

