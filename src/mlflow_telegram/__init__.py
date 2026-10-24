"""mlflow-telegram: watch MLflow training runs from Telegram.

An unofficial community project, not affiliated with MLflow or Databricks.

The pieces, each usable on its own:

- `Monitor` (training side): logs params, metrics and tags to MLflow and sends
  start / progress / evaluation / finish / crash messages to Telegram.
- `MlflowSource` (bot side): reads runs back through the `RunSource` protocol.
- `create_application` / `run_bot`: the Telegram bot (`bot` extra).
- `encode` / `decode`: the chart-link payload format (`docs/payload.md`).
- `build_page`: the static Mini App page that draws those links.
- `Config`: every setting, from `MLFLOW_TELEGRAM_*` environment variables.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["Monitor", "Formatter", "Config", "MlflowSource", "RunSource", "RunInfo", "Point",
           "encode", "decode", "build_page", "create_application", "run_bot", "__version__"]

_LAZY = {
    "Monitor": ("mlflow_telegram.monitor", "Monitor"),
    "Formatter": ("mlflow_telegram.monitor", "Formatter"),
    "Config": ("mlflow_telegram.config", "Config"),
    "MlflowSource": ("mlflow_telegram.sources.mlflow", "MlflowSource"),
    "RunSource": ("mlflow_telegram.sources", "RunSource"),
    "RunInfo": ("mlflow_telegram.sources", "RunInfo"),
    "Point": ("mlflow_telegram.sources", "Point"),
    "encode": ("mlflow_telegram.payload", "encode"),
    "decode": ("mlflow_telegram.payload", "decode"),
    "build_page": ("mlflow_telegram.page", "build_page"),
    "create_application": ("mlflow_telegram.bot.app", "create_application"),
    "run_bot": ("mlflow_telegram.bot.app", "run_bot"),
}


def __getattr__(name: str):
    # Imported on first use, so `from mlflow_telegram import Monitor` in a training
    # script never pulls in the bot's dependencies.
    if name in _LAZY:
        import importlib
        module, attribute = _LAZY[name]
        return getattr(importlib.import_module(module), attribute)
    raise AttributeError(f"module 'mlflow_telegram' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
