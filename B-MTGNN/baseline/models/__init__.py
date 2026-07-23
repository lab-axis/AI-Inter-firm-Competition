# baseline/models/__init__.py
# Each module exposes a single public function: run_<model>(ctx) -> dict
# ctx is a DataContext namedtuple supplied by run_baselines.py.
from .arima_model           import run_arima_baseline
from .var_model             import run_var_baseline
from .lstm_model            import run_lstm_baseline
from .mtgnn_model           import run_mtgnn_baseline
from .arima_transformer_model import run_arima_transformer_baseline

__all__ = [
    "run_arima_baseline",
    "run_var_baseline",
    "run_lstm_baseline",
    "run_mtgnn_baseline",
    "run_arima_transformer_baseline",
]
