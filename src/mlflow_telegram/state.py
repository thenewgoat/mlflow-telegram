"""The bot's small amount of state, kept in one JSON file both sides read.

The bot writes it (mute, the push interval, which alerts went out, the pinned
status message); each `Monitor` reads `mute_until` and `every_minutes` from it
before a throttled push, so `/mute` and `/every` reach runs already going.
`MLFLOW_TELEGRAM_STATE` names the file (see `mlflow_telegram.config`).
"""

import json
import os
import tempfile
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
    """Replaces the state atomically (a reader never sees half a file), readable
    by its owner only, through a fresh temporary file (never a followed link)."""
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, mode=0o700, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(handle, "w") as stream:      # mkstemp creates it 0600
            json.dump(value, stream, indent=1, sort_keys=True)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def muted(value: dict, now: float | None = None) -> bool:
    return value.get("mute_until", 0) > (time.time() if now is None else now)
