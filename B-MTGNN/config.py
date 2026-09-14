from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List
import yaml
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data_preparation' / 'data' / '5_bmtgnn_input'


def _str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    raise ValueError(f"Cannot interpret boolean value from: {v}")


@dataclass
class Config:
    # Dataset paths are resolved relative to the repository root.
    data: str = str(INPUT / 'bmtgnn_data.npy')
    log_interval: int = 2000
    o_save: str = str(ROOT / 'B-MTGNN/Bayesian/12mo/o_model.safetensors')
    save: str = str(ROOT / 'B-MTGNN/Bayesian/12mo/model.safetensors')
    optim: str = "adam"
    L1Loss: bool = False
    normalize: int = 2
    device: str = "cuda:0"

    # forecast
    data_file: str = str(INPUT / 'bmtgnn_data.npy')
    nodes_file: str = str(INPUT / 'node_ids.csv')
    graph_file: str = ""  # No external adjacency in the released framework.

    gcn_true: bool = True
    buildA_true: bool = True
    gcn_depth: int = 2
    num_nodes: int = 30
    dropout: float = 0.3
    subgraph_size: int = 8
    node_dim: int = 40
    dilation_exponential: int = 2
    conv_channels: int = 16
    residual_channels: int = 16
    skip_channels: int = 32
    end_channels: int = 64
    in_dim: int = 17
    seq_in_len: int = 12
    months: int = 12

    seq_out_len: int = 12  # Resolved from --months by get_args.
    horizon: int = 1
    layers: int = 3

    batch_size: int = 8
    lr: float = 0.001
    weight_decay: float = 0.00001
    clip: int = 10

    weight_regularizer: float = 1e-6
    dropout_regularizer: float = 1e-5

    propalpha: float = 0.05
    tanhalpha: float = 3

    epochs: int = 150
    num_split: int = 1
    step_size: int = 100

    search_iters: int = 0

    train_ratio: float = 0.75   # Compatibility argument; split_policy defines target windows.

    valid_ratio: float = 0.125  # Compatibility argument; split_policy defines target windows.

    hp_path: str = str(ROOT / 'B-MTGNN/Bayesian/12mo/hp.txt')

    # forecast
    images_dir: str = str(ROOT / 'B-MTGNN/Bayesian/12mo/forecast/plots') + '/'
    file_dir: str = str(ROOT / 'B-MTGNN/Bayesian/12mo/forecast/data') + '/'
    gap_dir: str = str(ROOT / 'B-MTGNN/Bayesian/12mo/forecast/gap') + '/'

    # run versioning configuration
    version: str = "12mo"
    use_sam: bool = True

    # Model components and fixed-hyperparameter training options.
    fixed_hp: bool = False            # pin hyperparameters from hp_path; search_iters becomes the repeat count
    seed: int = 123                   # base seed; with fixed_hp, repeat r uses seed + r
    use_revin: bool = True            # Reversible Instance Normalization
    use_decomp: bool = True           # trend/residual series decomposition
    use_concrete_dropout: bool = True # concrete dropout (learned rate) vs. fixed-rate dropout
    use_ar_branch: bool = True        # linear autoregressive branch on the trend
    use_afci_feedback: bool = True    # AFCI as an autoregressive input channel

    # The default withheld_matched policy keeps legacy test origins but separates target blocks.
    # withheld reserves test_reserve months for testing and valid_span for validation.
    # legacy is retained as a historical comparison option.
    split_policy: str = "withheld_matched"
    test_reserve: int = 24
    valid_span: int = 24

    # Negative seeds disable these explicit seed overrides.
    search_seed: int = -1      # one fixed seed for every Optuna trial
    sampler_seed: int = -1     # TPESampler(seed=...)
    burn_in: int = 20          # earliest epoch a checkpoint may be selected from
    dedupe_dilation: bool = False   # [1, 2] instead of [1, 1, 2]

    # Directory for per-evaluation prediction dumps. Empty disables it.
    calib_dump: str = ""


def _coerce_type(name: str, value: Any, cfg: Config):
    from typing import get_type_hints
    t = get_type_hints(cfg.__class__).get(name)
    if t is None or isinstance(value, t):
        return value
    try:
        if t is bool:
            return _str2bool(value)
        return t(value)
    except Exception:
        return value


def _merge_dicts(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in extra.items():
        if v is None:
            continue
        out[k] = v
    return out


def load_yaml(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config yaml must contain a mapping at top-level.")
    return data


def get_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    known, unknown = parser.parse_known_args(args=argv)

    cfg = Config()
    yaml_dict = load_yaml(known.config)

    cli_kv: Dict[str, Any] = {}
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok.lstrip("-")
        val: Any = True
        if "=" in key:
            key, val = key.split("=", 1)
        else:
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                val = unknown[i + 1]
                i += 1
        cli_kv[key] = val
        i += 1

    merged = asdict(cfg)
    for k, v in yaml_dict.items():
        if k in merged:
            merged[k] = v

    for k, v in cli_kv.items():
        if k in merged:
            merged[k] = _coerce_type(k, v, cfg)

    cfg_final = Config(**merged)
    if cfg_final.graph_file or not cfg_final.buildA_true:
        raise ValueError('This release uses a learned graph only; external adjacency is disabled.')
    if cfg_final.months not in (3, 6, 9, 12, 24, 36):
        raise ValueError('Unsupported forecast horizon.')
    cfg_final.seq_in_len = cfg_final.months
    cfg_final.seq_out_len = cfg_final.months
    if 'version' not in cli_kv and 'version' not in yaml_dict:
        cfg_final.version = f'{cfg_final.months}mo'
    return Namespace(**asdict(cfg_final))


if __name__ == "__main__":
    ns = get_args(sys.argv[1:])
    for k, v in vars(ns).items():
        print(f"{k}: {v!r}")
