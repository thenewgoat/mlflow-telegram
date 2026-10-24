"""What the bot says about a run: progress, ETA, staleness, best evaluation.

Works on any `RunSource`; the conventions it reads (all optional) are the tags
`Monitor` writes (`monitor.pid`, `monitor.host`, `monitor.max_steps`,
`monitor.patience`, `monitor.nonfinite`, `best_step`), a `max_steps`/`patience`
param, or a logged `command` whose flags are read as params.
"""

from __future__ import annotations

import math
import os
import shlex
import socket
import statistics
import time
from dataclasses import dataclass, field

from mlflow_telegram.redact import redact_params
from mlflow_telegram.sources import Point, RunInfo, RunSource

__all__ = ["RunView", "view", "live", "recent", "system_history", "parse_command",
           "all_params", "higher_is_better", "stale_after", "seconds_per_step", "pid_alive",
           "best_of", "duration", "number", "bar", "TRAIN_KEYS", "Point"]

TRAIN_KEYS = ("training_mean", "train_loss", "loss", "training")
STALE_FLOOR = 600.0        # seconds without a metric before a run counts as stale
HIGHER_IS_BETTER = ("f1", "acc", "precision", "recall", "auc", "ok")


def higher_is_better(key: str) -> bool:
    return any(word in key.lower() for word in HIGHER_IS_BETTER)


def parse_command(command: str) -> dict:
    """`--max-steps 9000 --no-bf16` → {"max_steps": "9000", "no_bf16": "true"}.

    For runs that logged only their command line as a param."""
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    found, index = {}, 0
    while index < len(words):
        word = words[index]
        if word.startswith("--") and len(word) > 2:
            key = word[2:].replace("-", "_")
            if "=" in key:
                key, value = key.split("=", 1)
                found[key] = value
            elif index + 1 < len(words) and not words[index + 1].startswith("--"):
                found[key] = words[index + 1]
                index += 1
            else:
                found[key] = "true"
        index += 1
    return found


def all_params(run: RunInfo) -> dict:
    """The run's params, with the arguments of a logged `command` filled in under
    them, and anything secret-looking masked (`mlflow_telegram.redact`)."""
    params = dict(run.params)
    if "command" in params:
        for key, value in parse_command(params["command"]).items():
            params.setdefault(key, value)
    return redact_params(params)


def _integer(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


@dataclass
class RunView:
    """What the bot says about one run."""

    id: str
    name: str
    experiment: str
    status: str
    started: float                     # epoch seconds
    ended: float | None
    metrics: dict                      # latest value per key
    params: dict
    tags: dict
    step: int | None = None
    max_steps: int | None = None
    patience: int | None = None
    seconds_per_step: float | None = None
    last_seen: float | None = None
    stale_after: float = STALE_FLOOR
    pid_alive: bool | None = None      # None: no pid tag, or another host
    best: tuple | None = None          # (key, value, step)
    train_key: str | None = None
    histories: dict = field(default_factory=dict)

    @property
    def live(self) -> bool:
        return self.status == "RUNNING"

    @property
    def fraction(self) -> float | None:
        if self.step is None or not self.max_steps:
            return None
        return min(1.0, self.step / self.max_steps)

    @property
    def eta(self) -> float | None:
        if self.step is None or not self.max_steps or not self.seconds_per_step:
            return None
        return max(0.0, (self.max_steps - self.step) * self.seconds_per_step)

    def silent_for(self, now: float | None = None) -> float | None:
        if self.last_seen is None:
            return None
        return (time.time() if now is None else now) - self.last_seen

    def stale(self, now: float | None = None) -> bool:
        """Live but quiet for longer than its own reporting rhythm allows."""
        if not self.live:
            return False
        reference = self.last_seen if self.last_seen is not None else self.started
        return (time.time() if now is None else now) - reference > self.stale_after

    @property
    def dead(self) -> bool:
        """Live in MLflow, but its process (on this host) is gone."""
        return self.live and self.pid_alive is False

    @property
    def short(self) -> str:
        return self.id[:8]


def stale_after(timestamps: list[float]) -> float:
    """How long a run may be silent: ten minutes, or three of its usual gaps
    between reports when it reports more slowly than that."""
    if len(timestamps) < 3:
        return STALE_FLOOR
    recent = sorted(timestamps)[-11:]
    gaps = [b - a for a, b in zip(recent, recent[1:]) if b > a]
    if not gaps:
        return STALE_FLOOR
    return max(STALE_FLOOR, 3 * statistics.median(gaps))


def seconds_per_step(metrics: dict, points: list[Point]) -> float | None:
    """From a logged speed when there is one, else from the curve's timestamps."""
    if metrics.get("seconds_per_step"):
        return float(metrics["seconds_per_step"])
    if metrics.get("steps_per_s"):
        return 1.0 / float(metrics["steps_per_s"])
    recent = points[-10:]
    if len(recent) >= 2 and recent[-1].step > recent[0].step:
        return (recent[-1].timestamp - recent[0].timestamp) / (recent[-1].step - recent[0].step)
    return None


def pid_alive(tags: dict, started: float) -> bool | None:
    """Whether the process that started the run is still running.

    None when the run has no pid tag or ran on another host. A pid reused by a
    process that began after the run did counts as gone."""
    pid, host = _integer(tags.get("monitor.pid")), tags.get("monitor.host")
    if pid is None or host != socket.gethostname():
        return None
    try:
        import psutil
        process = psutil.Process(pid)
        return process.create_time() <= started + 60
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    except Exception:  # noqa: BLE001 -- NoSuchProcess and friends
        return False


def best_of(histories: dict) -> tuple | None:
    """(key, value, step) of the best evaluation: `eval/validation` first."""
    keys = [k for k in histories if k.startswith("eval/") and histories[k]]
    if not keys:
        return None
    key = "eval/validation" if "eval/validation" in keys else sorted(keys)[0]
    pick = max if higher_is_better(key) else min
    point = pick(histories[key], key=lambda p: p.value)
    return key, point.value, point.step


def live(source: RunSource) -> list[RunInfo]:
    return source.list_runs(status="RUNNING", limit=100)


def recent(source: RunSource, hours: float = 24.0) -> list[RunInfo]:
    """Runs that started within the last `hours`, plus any still running."""
    started = source.list_runs(since=time.time() - hours * 3600, limit=100)
    seen = {run.id for run in started}
    return started + [run for run in live(source) if run.id not in seen]


def view(source: RunSource, run: RunInfo, now: float | None = None) -> RunView:
    """Everything the bot says about `run`: progress, ETA, staleness, best."""
    params = all_params(run)
    tags, metrics = dict(run.tags), dict(run.metrics)
    histories = {key: source.metric_history(run.id, key) for key in metrics
                 if not key.startswith(("system/", "final/"))}
    train_key = next((k for k in TRAIN_KEYS if histories.get(k)), None)
    steps = [points[-1].step for points in histories.values() if points]
    stamps = [p.timestamp for points in histories.values() for p in points]
    primary = histories.get(train_key) or next(
        (points for points in histories.values() if points), [])
    return RunView(
        id=run.id, name=run.name, experiment=run.experiment, status=run.status,
        started=run.start_time, ended=run.end_time, metrics=metrics, params=params, tags=tags,
        step=max(steps) if steps else None,
        max_steps=_integer(tags.get("monitor.max_steps")) or _integer(
            params.get("max_steps") or params.get("steps")),
        patience=_integer(tags.get("monitor.patience")) or _integer(params.get("patience")),
        seconds_per_step=seconds_per_step(metrics, primary),
        last_seen=max(stamps) if stamps else None,
        stale_after=stale_after([p.timestamp for p in primary]),
        pid_alive=pid_alive(tags, run.start_time) if run.status == "RUNNING" else None,
        best=best_of(histories), train_key=train_key, histories=histories)


def system_history(source: RunSource, run: RunInfo) -> dict:
    """`system/...` metric key → [(timestamp ms, value)]."""
    return {key: [(p.timestamp * 1000.0, p.value) for p in source.metric_history(run.id, key)]
            for key in run.metrics if key.startswith("system/")}


# -- words for the bot -----------------------------------------------------

def duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "?"
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h{minutes:02d}m"
    return f"{hours // 24}d{hours % 24}h"


def number(value) -> str:
    if value is None:
        return "?"
    value = float(value)
    if value == int(value) and abs(value) < 1e9:
        return str(int(value))
    return f"{value:.4g}"


def bar(fraction: float | None, width: int = 10) -> str:
    if fraction is None:
        return ""
    filled = round(fraction * width)
    return "▰" * filled + "▱" * (width - filled)
