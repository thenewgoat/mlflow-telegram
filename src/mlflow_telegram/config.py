"""Settings, from the environment (and a `.env` file), in one place.

Every setting has an `MLFLOW_TELEGRAM_*` name. A few older names are accepted
as fallbacks so existing setups keep working (`TELEGRAM_BOT_TOKEN`,
`TELEGRAM_CHAT_ID`, `TELEGRAM_EVERY_MINUTES`, `TRAINBOT_*`, `MONITOR_ENV_FILE`).

==============================  ==========================================  =========
Setting                         Variables (first one set wins)              Default
==============================  ==========================================  =========
``token``                       MLFLOW_TELEGRAM_TOKEN, TELEGRAM_BOT_TOKEN   (none)
``chat_id``                     MLFLOW_TELEGRAM_CHAT_ID, TELEGRAM_CHAT_ID   (none)
``tracking_uri``                MLFLOW_TRACKING_URI                         (none)
``pages_url``                   MLFLOW_TELEGRAM_PAGES_URL,                  (none)
                                TRAINBOT_PAGES_URL
``max_url``                     MLFLOW_TELEGRAM_MAX_URL,                    4096
                                TRAINBOT_PAGES_MAX_URL
``state_path``                  MLFLOW_TELEGRAM_STATE, TRAINBOT_STATE       see below
``disk_path``                   MLFLOW_TELEGRAM_DISK_PATH,                  .
                                TRAINBOT_DISK_PATH
``disk_gb``                     MLFLOW_TELEGRAM_DISK_GB, TRAINBOT_DISK_GB   20
``every_minutes``               MLFLOW_TELEGRAM_EVERY_MINUTES,              30
                                TELEGRAM_EVERY_MINUTES
``experiment``                  MLFLOW_EXPERIMENT_NAME                      (script)
==============================  ==========================================  =========

The state file (mute, push interval, alerts sent, chart states) defaults to
``$XDG_STATE_HOME/mlflow-telegram/state.json`` (``~/.local/state/...``).

`.env`: ``MLFLOW_TELEGRAM_ENV_FILE`` (or ``MONITOR_ENV_FILE``) names it, else
``./.env`` is used when present. It never overrides a variable already set.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = ["Config", "load_env_file", "default_state_path", "setting"]

PREFIX = "MLFLOW_TELEGRAM_"


def setting(environ: Mapping[str, str], *names: str, default: str = "") -> str:
    """The first of `names` set (non-empty) in `environ`, else `default`."""
    for name in names:
        value = environ.get(name, "")
        if value:
            return value
    return default


def default_state_path(environ: Mapping[str, str] | None = None) -> str:
    environ = os.environ if environ is None else environ
    explicit = setting(environ, PREFIX + "STATE", "TRAINBOT_STATE")
    if explicit:
        return explicit
    base = environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local",
                                                         "state")
    return os.path.join(base, "mlflow-telegram", "state.json")


def load_env_file(path: str | None = None) -> str | None:
    """Loads a `.env` into `os.environ` without overriding anything already set.

    Returns the file loaded, or None (no file, or python-dotenv not installed)."""
    path = path or setting(os.environ, PREFIX + "ENV_FILE", "MONITOR_ENV_FILE") or ".env"
    if not os.path.isfile(path):
        return None
    try:
        from dotenv import load_dotenv
    except ImportError:
        print(f"[mlflow-telegram] {path} not loaded: python-dotenv is not installed",
              file=sys.stderr, flush=True)
        return None
    load_dotenv(path, override=False)
    return path


def _number(text: str, default: float) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    """Everything the monitor and the bot read from the environment."""

    token: str = ""
    chat_id: int | None = None
    tracking_uri: str = ""
    pages_url: str = ""
    max_url: int = 4096
    state_path: str = field(default_factory=default_state_path)
    disk_path: str = "."
    disk_gb: float = 20.0
    every_minutes: float = 30.0
    experiment: str = ""

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None,
                 env_file: str | None = None) -> Config:
        """Settings from `environ` (default: `os.environ`, after loading the `.env`)."""
        if environ is None:
            load_env_file(env_file)
            environ = os.environ
        chat = setting(environ, PREFIX + "CHAT_ID", "TELEGRAM_CHAT_ID")
        return cls(
            token=setting(environ, PREFIX + "TOKEN", "TELEGRAM_BOT_TOKEN"),
            chat_id=int(chat) if chat.lstrip("-").isdigit() else None,
            tracking_uri=environ.get("MLFLOW_TRACKING_URI", ""),
            pages_url=setting(environ, PREFIX + "PAGES_URL", "TRAINBOT_PAGES_URL").rstrip("/"),
            max_url=int(_number(setting(environ, PREFIX + "MAX_URL", "TRAINBOT_PAGES_MAX_URL"),
                                4096)),
            state_path=default_state_path(environ),
            disk_path=setting(environ, PREFIX + "DISK_PATH", "TRAINBOT_DISK_PATH", default="."),
            disk_gb=_number(setting(environ, PREFIX + "DISK_GB", "TRAINBOT_DISK_GB"), 20.0),
            every_minutes=_number(setting(environ, PREFIX + "EVERY_MINUTES",
                                          "TELEGRAM_EVERY_MINUTES"), 30.0),
            experiment=environ.get("MLFLOW_EXPERIMENT_NAME", ""),
        )

    @property
    def telegram_ready(self) -> bool:
        return bool(self.token and self.chat_id is not None)
