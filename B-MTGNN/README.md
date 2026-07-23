# B-MTGNN — Bayesian Multivariate Time-series Graph Neural Network

A graph neural-network forecasting model (an MTGNN variant) that predicts the multivariate
time series of 30 AI/IT companies. The inter-company graph is **learned adaptively** during
training; the model adds RevIN normalization, series decomposition, Concrete-Dropout /
MC-Dropout Bayesian uncertainty, an optional SAM optimizer, and optional Optuna hyperparameter
search.

---

## Inputs

Produced by the `data_preparation/` pipeline (stage 5):

| File | Shape / content | Used for |
|---|---|---|
| `data/5_bmtgnn_input/bmtgnn_data.npy` | `[num_nodes=30, timesteps, features=17]` | Model input tensor (training & forecasting) |
| `data/5_bmtgnn_input/node_ids.csv` | node index → ticker | Node labels |
| `data/5_bmtgnn_input/adj_mat.csv` | 30×30 static adjacency | **Visualization only** — see note below |

> **Graph note.** The model **learns its own graph** (`buildA_true=True`; the predefined-graph
> blending in `net.py` is disabled). `adj_mat.csv` is **not** a model/training input. It is read
> only by `forecast.py` to group *which* focal-node + neighbour forecasts are drawn together
> (edge presence only, `weight > 0`); it never affects the forecast values. This is consistent
> with `data_preparation/README.md` §4.

---

## Prerequisites

- Python 3.11+ (Recommend 3.13), a CUDA GPU is recommended (`device: cuda:0` by default; set `--device cpu` to use CPU)
- `pip install -r B-MTGNN/requirements.txt`
- Stage-5 inputs available under `../data_preparation/data/5_bmtgnn_input/`

---

## Configuration

All settings live in `config.py` (a `@dataclass Config`). Resolution order is
**defaults → YAML (`--config file.yaml`) → CLI overrides (`--key value`)**.

Selected defaults:

| Key | Default | Meaning |
|---|---|---|
| `num_nodes` | 30 | Companies (graph nodes) |
| `in_dim` | 17 | Input features per node |
| `seq_in_len` | 36 | Input window (months) |
| `seq_out_len` | 6 | Forecast horizon (3 mo; 12 → 6 mo, 24 → 12 mo alternatives are commented) |
| `gcn_true` / `buildA_true` | True / True | Use GCN with an adaptively-learned graph |
| `epochs`, `lr`, `batch_size` | 150, 0.001, 8 | Training schedule |
| `train_ratio` / `valid_ratio` | 0.75 / 0.125 | Chronological split |
| `use_sam` | True | Sharpness-Aware Minimization optimizer |
| `weight_regularizer` / `dropout_regularizer` | 1e-6 / 1e-5 | Concrete-Dropout regularizers |
| `search_iters` | 0 | Optuna HPO iterations (0 = off) |

---

## Usage

Run from **inside the `B-MTGNN/` directory** (default paths resolve relative to it):

```bash
cd B-MTGNN

# Train (saves model to model/Bayesian/*.safetensors)
python train.py

# Override any config key on the CLI, or use a YAML file
python train.py --epochs 200 --seq_out_len 12 --device cpu
python train.py --config my_run.yaml

# Forecast + plots (Bayesian MC-Dropout, num_runs samples)
python forecast.py --num_runs 50 --run <run_id>
```

Training uses Concrete-Dropout with (optionally) the SAM optimizer, saves weights as
`safetensors`, and organizes runs under `model/Bayesian/`. `forecast.py` reloads a saved
checkpoint, performs `num_runs` MC-Dropout passes to produce mean/uncertainty bands, and writes
plots, per-series data, and gap tables under `model/Bayesian/forecast/`.

---

## Component map

| File | Role |
|---|---|
| `config.py` | Central `Config` dataclass; YAML + CLI argument resolution (`get_args`) |
| `net.py` | `gtnet` model: adaptive graph construction, GCN, dilated-inception TCN, RevIN, series decomposition, Concrete-Dropout |
| `layer.py` | Model building blocks (graph constructor, mix-hop propagation, dilated inception, etc.) |
| `trainer.py` | `Optim` optimizer wrapper (Adam / SAM) |
| `o_util.py`, `util.py` | Data loaders (`DataLoaderS`): load the tensor, chronological split, scaling |
| `train.py` | Training entry point |
| `forecast.py` | Forecasting + plotting entry point (Bayesian MC-Dropout) |
| `pt_plots.py`, `for_test.py`, `train_test.py` | Auxiliary plotting / experimental utilities |

---

## Outputs

- Trained weights: `model/Bayesian/*.safetensors`
- Forecast artifacts: `model/Bayesian/forecast/plots/`, `.../data/`, `.../gap/`

All `model/` and `data/` artifacts are git-ignored; regenerate them via training / forecasting.
