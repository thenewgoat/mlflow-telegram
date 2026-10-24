"""The watcher's rules: what about a run (or the machine) is worth a message.

Pure functions over `RunView`s, so they test without MLflow or Telegram. Each
alert has a `key` that stays the same while its condition holds; the bot sends
a key once, forgets it when the condition clears, and so warns again if the
same thing happens later.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from mlflow_telegram.runs import RunView, duration, number

__all__ = ["Alert", "Rule", "HostRule", "Died", "Diverging", "GpuIdle", "EarlyStopNear",
           "DiskLow", "DEFAULT_RULES", "evaluate", "for_run"]

DIVERGE_FACTOR = 3.0
GPU_IDLE_PERCENT = 10.0
GPU_IDLE_SECONDS = 300.0
GPU_KEY = "system/gpu_0_utilization_percentage"


@dataclass(frozen=True)
class Alert:
    key: str
    run_id: str | None
    text: str
    can_kill: bool = False


def died(view: RunView, now: float) -> Alert | None:
    """Its process is gone, or it has said nothing for longer than its rhythm."""
    if not view.live:
        return None
    title = f"{view.experiment} {view.name}"
    if view.dead:
        return Alert(f"{view.id}:died", view.id,
                     f"💀 {title} died: its process is gone, last metric "
                     f"{duration(view.silent_for(now))} ago at step {view.step}. "
                     f"MLflow still says RUNNING.", can_kill=True)
    if view.stale(now):
        silent = view.silent_for(now)
        since = (f"{duration(silent)} ago" if silent is not None
                 else f"never ({duration(now - view.started)} since start)")
        alive = " (its process is still there)" if view.pid_alive else ""
        return Alert(f"{view.id}:stalled", view.id,
                     f"🥶 {title} has gone quiet{alive}: last metric {since}, "
                     f"at step {view.step}; it usually reports within "
                     f"{duration(view.stale_after)}.", can_kill=not view.pid_alive)
    return None


def diverging(view: RunView) -> Alert | None:
    """A NaN/inf metric, or a training loss over three times its recent median."""
    if not view.live:
        return None
    title = f"{view.experiment} {view.name}"
    flagged = view.tags.get("monitor.nonfinite")
    if flagged:
        return Alert(f"{view.id}:nonfinite", view.id,
                     f"🔥 {title} logged a NaN/inf: {flagged}.")
    points = view.histories.get(view.train_key or "", [])
    if len(points) < 6:
        return None
    window = [p.value for p in points[-21:-1]]
    median = statistics.median(window)
    last = points[-1]
    if median > 0 and last.value > DIVERGE_FACTOR * median:
        return Alert(f"{view.id}:diverging", view.id,
                     f"📈 {title} may be diverging: {view.train_key} {number(last.value)} "
                     f"at step {last.step}, {last.value / median:.1f}× its recent median "
                     f"{number(median)}.")
    return None


def gpu_idle(view: RunView, gpu: list[tuple[float, float]], now: float) -> Alert | None:
    """GPU under 10% for the last five minutes of a live run's system metrics.

    `gpu` is [(timestamp ms, percent)]; a run without system metrics never alerts."""
    if not view.live or not gpu:
        return None
    window = [(t / 1000.0, v) for t, v in gpu if t / 1000.0 >= now - GPU_IDLE_SECONDS - 60]
    if len(window) < 3 or window[-1][0] - window[0][0] < GPU_IDLE_SECONDS * 0.8:
        return None
    if now - window[-1][0] > 180:
        return None  # sampling stopped; `died` covers that
    if all(v < GPU_IDLE_PERCENT for _, v in window):
        return Alert(f"{view.id}:gpu-idle", view.id,
                     f"😴 {view.experiment} {view.name}: GPU under {GPU_IDLE_PERCENT:.0f}% "
                     f"for {duration(window[-1][0] - window[0][0])} "
                     f"(latest {window[-1][1]:.0f}%).")
    return None


def early_stop_near(view: RunView) -> Alert | None:
    """One more evaluation without a new best and patience stops the run."""
    used = view.metrics.get("patience_used")
    if not view.live or not view.patience or used is None:
        return None
    if int(used) == view.patience - 1:
        best = (f"best {view.best[0]} {number(view.best[1])} at step {view.best[2]}"
                if view.best else "no best yet")
        return Alert(f"{view.id}:patience@{view.tags.get('best_step', '')}", view.id,
                     f"⏳ {view.experiment} {view.name}: one evaluation left of patience "
                     f"{view.patience} ({best}).")
    return None


def disk_low(free_bytes: float, threshold_gb: float, path: str) -> Alert | None:
    free_gb = free_bytes / 2**30
    if free_gb < threshold_gb:
        return Alert("host:disk-low", None,
                     f"💾 Disk low: {free_gb:.1f} GB free on {path} "
                     f"(threshold {threshold_gb:g} GB).")
    return None


def for_run(view: RunView, gpu: list | None = None, now: float | None = None) -> list[Alert]:
    now = time.time() if now is None else now
    found = [died(view, now), diverging(view), gpu_idle(view, gpu or [], now),
             early_stop_near(view)]
    return [alert for alert in found if alert is not None]


# -- pluggable rules ----------------------------------------------------------

class Rule:
    """One check on a live run. Subclass, implement `check`, and pass instances to
    `create_application(rules=...)` (the defaults are `DEFAULT_RULES`)."""

    #: whether the rule needs the run's GPU utilisation samples
    wants_gpu = False

    def check(self, view: RunView, now: float, gpu: list | None = None) -> Alert | None:
        raise NotImplementedError


class HostRule:
    """One check on the machine, not tied to a run."""

    def check(self, now: float) -> Alert | None:
        raise NotImplementedError


class Died(Rule):
    """The run's process is gone, or it has been silent longer than its rhythm."""

    def check(self, view, now, gpu=None):
        return died(view, now)


class Diverging(Rule):
    """A NaN/inf metric, or a training loss over 3× its recent median."""

    def check(self, view, now, gpu=None):
        return diverging(view)


class GpuIdle(Rule):
    """GPU under 10% for five minutes of a live run (needs system metrics)."""

    wants_gpu = True

    def check(self, view, now, gpu=None):
        return gpu_idle(view, gpu or [], now)


class EarlyStopNear(Rule):
    """One evaluation left before patience stops the run."""

    def check(self, view, now, gpu=None):
        return early_stop_near(view)


class DiskLow(HostRule):
    """Less than `threshold_gb` free on `path`."""

    def __init__(self, path: str = ".", threshold_gb: float = 20.0):
        self.path, self.threshold_gb = path, threshold_gb

    def check(self, now):
        import shutil
        try:
            free = shutil.disk_usage(self.path).free
        except OSError:
            return None
        return disk_low(free, self.threshold_gb, self.path)


DEFAULT_RULES: tuple[Rule, ...] = (Died(), Diverging(), GpuIdle(), EarlyStopNear())


def evaluate(source, views: list[RunView], rules=DEFAULT_RULES, host_rules=(),
             now: float | None = None) -> list[Alert]:
    """Every alert the rules raise for `views` (live runs) and the host."""
    now = time.time() if now is None else now
    found = []
    for view in views:
        gpu = None
        if any(rule.wants_gpu for rule in rules) and GPU_KEY in view.metrics:
            gpu = [(p.timestamp * 1000.0, p.value) for p in source.metric_history(view.id, GPU_KEY)]
        for rule in rules:
            alert = rule.check(view, now, gpu)
            if alert is not None:
                found.append(alert)
    for rule in host_rules:
        alert = rule.check(now)
        if alert is not None:
            found.append(alert)
    return found
