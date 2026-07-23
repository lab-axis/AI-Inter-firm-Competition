from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List
import yaml
import sys


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
    # === defaults (AI Company Dataset resolved from Workspace Root) ===
    data: str = "../data_preparation/data/5_bmtgnn_input/bmtgnn_data.npy"
    log_interval: int = 2000
    o_save: str = "../B-MTGNN/model/Bayesian/o_model.safetensors"
    save: str = "../B-MTGNN/model/Bayesian/model.safetensors"
    optim: str = "adam"
    L1Loss: bool = False
    normalize: int = 2
    device: str = "cuda:0"

    # forecast
    data_file: str = "../data_preparation/data/5_bmtgnn_input/bmtgnn_data.npy"
    nodes_file: str = "../data_preparation/data/5_bmtgnn_input/node_ids.csv"
    graph_file: str = "../data_preparation/data/5_bmtgnn_input/adj_mat.csv"

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
    seq_in_len: int = 36

    seq_out_len: int = 6    # 3mo
    # seq_out_len: int = 12  # 6mo
    # seq_out_len: int = 24  # 12mo
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

    # fast: 50~100. 1.0~1.5H / best: 130~260. 1.5H~3.5H / real: 200. 3.0H / 1.2 IterPerMin
    search_iters: int = 0

    train_ratio: float = 0.75   # 3mo
    # train_ratio: float = 0.70  # 6mo
    # train_ratio: float = 0.60  # 12mo

    valid_ratio: float = 0.125  # 3mo
    # valid_ratio: float = 0.15  # 6mo
    # valid_ratio: float = 0.20  # 12mo

    hp_path: str = "../B-MTGNN/model/Bayesian/org_hp.txt"

    # forecast
    images_dir: str = "../B-MTGNN/model/Bayesian/forecast/plots/"
    file_dir: str = "../B-MTGNN/model/Bayesian/forecast/data/"
    gap_dir: str = "../B-MTGNN/model/Bayesian/forecast/gap/"

    # run versioning configuration
    version: str = "auto"
    use_sam: bool = True


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
    return Namespace(**asdict(cfg_final))


if __name__ == "__main__":
    ns = get_args(sys.argv[1:])
    for k, v in vars(ns).items():
        print(f"{k}: {v!r}")
