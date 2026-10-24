"""The bot's small amount of state, kept in one JSON file both sides read.

The bot writes it (mute, the push interval, which alerts went out, the pinned
status message); each `Monitor` reads `mute_until` and `every_minutes` from it
before a throttled push, so `/mute` and `/every` reach runs already going.
`MLFLOW_TELEGRAM_STATE` names the file (see `mlflow_telegram.config`).
"""

import json
import os
import time

_cache: dict = {}


def default_path() -> str:
    from mlflow_telegram.config import default_state_path
    return default_state_path()


def read(path: str | None) -> dict:
    """The state, or {} when there is none or it cannot be read.

    Re-read only when the file changes, so a training loop can ask every step."""
    if not path:
        return {}
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        return {}
    cached = _cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        with open(path) as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return {}
    _cache[path] = (mtime, value)
    return value


def write(path: str, value: dict) -> None:
    """Replaces the state atomically (a reader never sees half a file)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=1, sort_keys=True)
    os.replace(temporary, path)


def muted(value: dict, now: float | None = None) -> bool:
    return value.get("mute_until", 0) > (time.time() if now is None else now)
