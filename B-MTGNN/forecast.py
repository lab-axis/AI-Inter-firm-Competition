import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive plotting backend.
import numpy as np
import torch
torch.set_num_threads(1)  # Limit CPU threading for the Windows execution path.
import csv
from collections import defaultdict
from matplotlib import pyplot
from safetensors import safe_open
import json
from net import gtnet
import pandas as pd
import math
from util import DataLoaderS
from run_profile import load_profile

pyplot.rcParams['savefig.dpi'] = 1200
colours = [
    "RoyalBlue", "Crimson", "DarkOrange", "MediumPurple",
    "MediumVioletRed", "DodgerBlue", "Indigo", "coral",
    "hotpink", "DarkMagenta", "SteelBlue", "brown",
    "MediumAquamarine", "SlateBlue", "SeaGreen", "MediumSpringGreen",
    "DarkOliveGreen", "Teal", "OliveDrab", "MediumSeaGreen",
    "DeepSkyBlue", "MediumSlateBlue", "MediumTurquoise", "FireBrick",
    "DarkCyan", "violet", "MediumOrchid", "DarkSalmon", "DarkRed"
]


def load_model(Data):
    with safe_open(args.o_save, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    arch_json = metadata.get("arch")
    if arch_json:
        arch = json.loads(arch_json)
        if not arch.get('buildA_true', False):
            raise ValueError('This checkpoint requires an external graph and is not supported by this release.')
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
            subtract_last=arch.get("subtract_last", True),
            weight_regularizer=arch.get("weight_regularizer", 1e-6),
            dropout_regularizer=arch.get("dropout_regularizer", 1e-5),
            use_revin=arch.get('use_revin', True),
            use_decomp=arch.get('use_decomp', True),
            use_concrete_dropout=arch.get('use_concrete_dropout', True),
            use_ar_branch=arch.get('use_ar_branch', True),
            use_afci_feedback=arch.get('use_afci_feedback', True),
        ).to(device)

        model.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError('Checkpoint is missing architecture metadata; refusing a partial model load.')
    # Return arch alongside the model so callers can use seq_length, out_dim, etc.
    return model, arch


def exponential_smoothing(series, alpha):
    result = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result


# Clip negative values for the optional display path.
def zero_negative_curves(data, forecast, attack, firms):
    for node in [attack] + list(firms):
        data[:, index[node]] = data[:, index[node]].clamp(min=0)
        forecast[:, index[node]] = forecast[:, index[node]].clamp(min=0)
    return data, forecast


def extend_months_and_format(dt_index, months):
    """
    dt_index : pandas.DatetimeIndex (monthly intervals)
    months   : number of months to forecast
    return   : list of string years (yyyy) covering the existing data + forecasted horizon
    """
    if not isinstance(dt_index, pd.DatetimeIndex):
        raise TypeError("dt_index must be a pandas DatetimeIndex")

    new_dates = pd.date_range(
        start=dt_index[-1] + pd.offsets.MonthBegin(1),
        periods=months,
        freq='MS'
    )
    full_index = dt_index.union(new_dates)
    return full_index.strftime("%Y").tolist()


def distance_to_next_january(dt_index):
    """
    dt_index : pandas.DatetimeIndex
    return   : number of months (int) from the first month in data until the next January
    """
    if not isinstance(dt_index, pd.DatetimeIndex):
        raise TypeError("dt_index must be a pandas DatetimeIndex")

    month = dt_index[0].month
    count = 0
    while True:
        month = month % 12 + 1
        count += 1
        if month == 1:
            break
    return count


# Plot the focal firm and the supplied comparison firms.
def plot_forecast(data, forecast, confidence, attack, firms, timeindex, index, col):
    data, forecast = zero_negative_curves(data, forecast, attack, firms)
    n = len(colours)

    pyplot.style.use("seaborn-v0_8-dark")
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])

    # Plot the focal node (attack/company)
    counter = 0
    d_attack = torch.cat((data[:, index[attack]], forecast[0:1, index[attack]]), dim=0)
    f_attack = forecast[:, index[attack]]
    c_attack = confidence[:, index[attack]]
    a = attack

    ax.plot(range(len(d_attack)), d_attack, '-', color=colours[counter % n], label=a, linewidth=2)
    ax.plot(range(len(d_attack) - 1, (len(d_attack) + len(f_attack)) - 1), f_attack, '-', color=colours[counter % n], linewidth=2)
    
    std_attack = c_attack / 1.96
    ax.fill_between(range(len(d_attack) - 1, (len(d_attack) + len(f_attack)) - 1), f_attack - (std_attack * 1.96), f_attack + (std_attack * 1.96), color=colours[counter % n], alpha=0.1, edgecolor='none')
    ax.fill_between(range(len(d_attack) - 1, (len(d_attack) + len(f_attack)) - 1), f_attack - (std_attack * 1.28), f_attack + (std_attack * 1.28), color=colours[counter % n], alpha=0.15, edgecolor='none')
    ax.fill_between(range(len(d_attack) - 1, (len(d_attack) + len(f_attack)) - 1), f_attack - (std_attack * 1.00), f_attack + (std_attack * 1.00), color=colours[counter % n], alpha=0.20, edgecolor='none')
    counter += 1

    # Plot each comparison firm.
    for s in firms:
        d = torch.cat((data[:, index[s]], forecast[0:1, index[s]]), dim=0)
        f = forecast[:, index[s]]
        c = confidence[:, index[s]]
        
        ax.plot(range(len(d)), d, '-', color=colours[counter % n], label=s, linewidth=1)
        ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f, '-', color=colours[counter % n], linewidth=1)
        
        std_c = c / 1.96
        ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), f - (std_c * 1.96), f + (std_c * 1.96), color=colours[counter % n], alpha=0.15, edgecolor='none')
        ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), f - (std_c * 1.28), f + (std_c * 1.28), color=colours[counter % n], alpha=0.20, edgecolor='none')
        ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), f - (std_c * 1.00), f + (std_c * 1.00), color=colours[counter % n], alpha=0.25, edgecolor='none')
        
        
        counter += 1

    # X-axis: show yearly ticks aligned to January
    x = extend_months_and_format(timeindex, months=len(forecast))
    offset = distance_to_next_january(timeindex)
    xticks = range(len(x))
    ax.set_xticks(xticks[offset::12])
    ax.set_xticklabels(x[offset::12])

    ax.set_ylabel("AFCI", fontsize=15)
    pyplot.yticks(fontsize=13)
    ax.legend(loc="upper left", prop={'size': 10}, bbox_to_anchor=(1, 1.03))
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90, fontsize=13)
    pyplot.title(a.upper(), y=1.03, fontsize=18)

    fig = pyplot.gcf()
    fig.set_size_inches(10, 7)

    pyplot.savefig(args.images_dir + a.replace('/', '_') + '.png', bbox_inches="tight")
    pyplot.savefig(args.images_dir + a.replace('/', '_') + ".pdf", bbox_inches="tight", format='pdf')
    pyplot.close()


# Saves the numerical forecast, past data, confidence interval, and variance for each node
def save_data(data, forecast, confidence, variance, col):
    for i in range(data.shape[1]):
        name = col[i]
        with open(args.file_dir + name.replace('/', '_') + '.txt', 'w') as ff:
            ff.write('Data: ' + str(data[:, i].tolist()) + '\n')
            ff.write('Forecast: ' + str(forecast[:, i].tolist()) + '\n')
            ff.write('95% Confidence: ' + str(confidence[:, i].tolist()) + '\n')
            ff.write('Variance: ' + str(variance[:, i].tolist()) + '\n')


# Save monthly differences between the focal firm and the comparison firms.
def save_gap(forecast, attack, firms, index):
    last_date = timeindex[-1]
    new_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=forecast.shape[0],
        freq='MS'
    )
    future_months = new_dates.strftime("%Y-%m").tolist()

    with open(args.gap_dir + attack.replace('/', '_') + '_gap.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Firm'] + future_months)
        table = []
        a = forecast[:, index[attack]].tolist()
        for s in firms:
            f = forecast[:, index[s]].tolist()
            gap = [x - y for x, y in zip(a, f)]
            table.append([s] + gap)
        sorted_table = sorted(table, key=lambda row: sum(row[1:]))
        for row in sorted_table:
            writer.writerow(row)


def comparison_pairs(firms):
    """Report all ordered firm pairs; this does not construct a model adjacency."""
    return {firm: [other for other in firms if other != firm] for firm in firms}


if __name__ == '__main__':
    from config import get_args
    import os
    import argparse

    # num_runs controls Monte Carlo forward passes, not training repetitions.
    # run selects a named checkpoint directory under Bayesian/.
    run_parser = argparse.ArgumentParser(add_help=False)
    run_parser.add_argument("--num_runs", type=int, default=50)
    run_parser.add_argument("--run", type=str, default="12mo")
    run_parser.add_argument("--plots", action="store_true", help="Also render optional diagnostic figures")
    run_parser.add_argument("--output-dir", help="Separate directory for newly generated forecasts")
    run_args, remaining_argv = run_parser.parse_known_args()

    num_runs = run_args.num_runs
    if num_runs < 2:
        raise ValueError('--num_runs must be at least 2 for the sample variance.')
    selected_run = run_args.run if run_args.run is not None else num_runs

    args = get_args(remaining_argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bayesian_base_dir = os.path.join(script_dir, "Bayesian")
    run_dir = os.path.join(bayesian_base_dir, f"{selected_run}")

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"[Error] The specified run directory does not exist: {run_dir}\n")

    model_path = os.path.join(run_dir, "o_model.safetensors")
    if not os.path.isfile(model_path) or os.path.getsize(model_path) == 0:
        raise FileNotFoundError(f"[Error] Valid model file does not exist: {model_path}\n")

    # Restore per-run normalization and partition settings before loading input.
    profile = load_profile(run_dir, args.data, args.nodes_file)
    args.split_policy = profile['split_policy']
    args.test_reserve = profile['test_reserve']
    args.valid_span = profile['valid_span']
    args.normalize = profile.get('normalize', 2)
    args.horizon = profile.get('horizon_offset', 1)
    if args.split_policy == 'legacy':
        print('[NOTICE] Historical compatibility profile; not a target-disjoint evaluation.')
    # Scaling windows must match the model horizon before constructing the loader.
    with safe_open(model_path, framework='pt', device='cpu') as handle:
        checkpoint_arch = json.loads((handle.metadata() or {}).get('arch', '{}'))
    if not checkpoint_arch.get('buildA_true', False):
        raise ValueError('A learned-graph checkpoint with architecture metadata is required.')
    args.seq_in_len = checkpoint_arch['seq_length']
    args.seq_out_len = checkpoint_arch['out_dim']
    output_dir = os.path.abspath(run_args.output_dir) if run_args.output_dir else os.path.join(run_dir, 'forecast')

    args.version    = f"run_{selected_run}"
    args.o_save     = model_path
    args.images_dir = os.path.join(output_dir, "plots", "")
    args.file_dir   = os.path.join(output_dir, "data", "")
    args.gap_dir    = os.path.join(output_dir, "gap", "")

    if run_args.plots:
        os.makedirs(args.images_dir, exist_ok=True)
    os.makedirs(args.file_dir,   exist_ok=True)
    os.makedirs(args.gap_dir,    exist_ok=True)

    print(f"Using model: {model_path}  ({os.path.getsize(model_path):,} bytes)")
    print(f"Results saved to: {output_dir}\n")

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

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
        nodes_file=args.nodes_file,
        split_policy=args.split_policy,
        test_reserve=args.test_reserve,
        valid_span=args.valid_span,
        num_eval=profile.get('num_eval', 7),
    )

    timeindex = pd.to_datetime(Data.timeindex, format="%Y-%m-%d")
    col = Data.col
    index = {c: i for i, c in enumerate(col)}

    graph = comparison_pairs(col)

    dat = Data.dat[:, :, 0]          # target feature (index 0) for 2D plotting
    scale = Data.scale.cpu().numpy()[:, 0]

    print('\ndata shape:', dat.shape)

    # Restore the model architecture and weights from checkpoint metadata.
    model, arch = load_model(Data)
    model = model.to(device)

    # Use the seq_length baked into the saved checkpoint, not the config default.
    P = arch["seq_length"]
    print(f'[Forecast] seq_length from model arch: {P}')
    X = torch.from_numpy(Data.dat[-P:, :, :])  # [P, N, d]
    X = X.unsqueeze(0)                          # [1, P, N, d]
    X = X.permute(0, 3, 2, 1).float().to(device)  # [1, d, N, P]

    # Run inference in a worker with an enlarged stack for the Windows execution path.
    import threading

    _exc_holder = [None]

    def _run_inference():
        try:
            # Bayesian estimation: run the model num_runs times and aggregate statistics
            model.eval()
            model.mc_dropout = True  # Enable MC Dropout during evaluation
            mu_outputs = []
            sigma_outputs = []

            for i in range(num_runs):
                print(f"run: {i + 1}/{num_runs}", flush=True)
                with torch.no_grad():
                    x_mu, x_sigma = model(X)
                    y_mu = x_mu[-1, :, :, -1].clone()
                    y_sigma = x_sigma[-1, :, :, -1].clone()
                mu_outputs.append(y_mu)
                sigma_outputs.append(y_sigma)

            mu_out = torch.stack(mu_outputs)
            sigma_out = torch.stack(sigma_outputs)

            Y = torch.mean(mu_out, dim=0)

            # Law of Total Variance
            epistemic_var = torch.var(mu_out, dim=0)
            aleatoric_var = torch.mean(sigma_out ** 2, dim=0)
            total_variance = epistemic_var + aleatoric_var

            total_std = torch.sqrt(total_variance)
            confidence_val = 1.96 * total_std  # 95% prediction interval

            Y_cpu          = Y.cpu()
            variance_cpu   = total_variance.cpu()
            confidence_cpu = confidence_val.cpu()

            # Inverse-scale to original value range
            # Convert scale to tensor to avoid NumPy 2.0 __array_wrap__ deprecation
            scale_t = torch.from_numpy(scale).float()
            dat_scaled     = dat * scale
            Y_scaled       = Y_cpu * scale_t
            variance_scaled = variance_cpu * (scale_t ** 2)
            confidence_scaled = confidence_cpu * scale_t

            print('output shape:', Y_scaled.shape)
            forecast_len = Y_scaled.shape[0]  # actual forecast horizon from the saved model

            dat_t = torch.from_numpy(dat_scaled)
            save_data(dat_t, Y_scaled, confidence_scaled, variance_scaled, col)

            # Global normalisation across all nodes (single shared max)
            all_data = torch.cat((dat_t, Y_scaled), dim=0)
            global_max = all_data.max().item()

            all_n        = all_data / global_max
            confidence_n = confidence_scaled / global_max

            # Exponential smoothing
            smoothed_dat        = torch.stack(exponential_smoothing(all_n, 0.1))
            smoothed_confidence = torch.stack(exponential_smoothing(confidence_n, 0.1))

            # Write gap tables for all comparison firms; optionally render plots.
            for attack, firms in graph.items():
                if run_args.plots:
                    plot_forecast(
                        smoothed_dat[:-forecast_len, ],
                        smoothed_dat[-forecast_len:, ],
                        smoothed_confidence,
                        attack,
                        firms,
                        timeindex,
                        index,
                        col
                    )
                save_gap(smoothed_dat[-forecast_len:, ], attack, firms, index)

        except Exception as e:
            _exc_holder[0] = e

    # Allocate a 64 MB stack for the inference worker.
    threading.stack_size(64 * 1024 * 1024)
    worker = threading.Thread(target=_run_inference)
    worker.start()
    worker.join()

    if _exc_holder[0] is not None:
        raise _exc_holder[0]

    print("Done.")
