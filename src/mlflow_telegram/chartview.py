"""The bot's chart explorer: what a chart message shows, and the buttons that change it.

A chart message is a `ChartState` (runs, metrics, range, log, smoothing) kept in
the bot's state file under a short id, so every button fits Telegram's 64-byte
callback limit (`cv:<id>:<op>`) and still works after the bot restarts. A
button changes the state and the bot redraws the picture in the same message.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from mlflow_telegram.runs import TRAIN_KEYS
from mlflow_telegram.sources import RunSource

RANGES = (("all", "All"), ("50", "Last 50%"), ("20", "Last 20%"), ("best", "Since best"))
SMOOTHING = (0.0, 0.6, 0.9)
KEEP = 300            # chart states remembered (the oldest are forgotten)
MAX_RUNS = 6
MAX_KEYS = 24         # metric toggles offered


@dataclass
class ChartState:
    runs: list[str]                      # run ids, the first is the chart's own run
    keys: list[str]                      # metrics shown
    mode: str = "all"
    log: bool = False
    smooth: float = 0.0
    picker: str = ""                     # "", "metrics" or "runs"
    offered: list[str] = field(default_factory=list)     # metric keys the picker lists
    candidates: list[str] = field(default_factory=list)  # run ids the run picker lists
    made: float = field(default_factory=time.time)

    @classmethod
    def load(cls, value: dict) -> ChartState:
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in value.items() if k in known})


def new_id() -> str:
    return secrets.token_hex(4)


def remember(state: dict, chart_id: str, chart: ChartState) -> None:
    charts = state.setdefault("charts", {})
    charts[chart_id] = asdict(chart)
    if len(charts) > KEEP:
        for old in sorted(charts, key=lambda k: charts[k].get("made", 0))[:len(charts) - KEEP]:
            del charts[old]


def recall(state: dict, chart_id: str) -> ChartState | None:
    value = state.get("charts", {}).get(chart_id)
    return ChartState.load(value) if value else None


def default_keys(available: list[str]) -> list[str]:
    """The training curve and the main evaluation, when the run has them."""
    train = next((k for k in TRAIN_KEYS if k in available), None)
    evals = [k for k in available if k == "eval/validation"] or \
        [k for k in available if k.startswith("eval/")][:1]
    keys = [k for k in [train, *evals] if k]
    return keys or available[:1]


def ordered_keys(available) -> list[str]:
    """Metric keys in picker order: training, evaluation, the rest, system."""
    def rank(key):
        if key in TRAIN_KEYS:
            return 0, TRAIN_KEYS.index(key), key
        if key.startswith("eval/"):
            return 1, 0, key
        if key.startswith("system/"):
            return 3, 0, key
        return 2, 0, key
    return sorted((k for k in available if not k.startswith("final/")), key=rank)


def apply(chart: ChartState, op: str, arg: str = "") -> ChartState:
    """One button press. Unknown ops leave the chart as it was."""
    if op == "r" and arg in dict(RANGES):
        chart.mode = arg
    elif op == "l":
        chart.log = not chart.log
    elif op == "s":
        index = SMOOTHING.index(chart.smooth) if chart.smooth in SMOOTHING else 0
        chart.smooth = SMOOTHING[(index + 1) % len(SMOOTHING)]
    elif op == "m":
        chart.picker = "metrics"
    elif op == "o":
        chart.picker = "runs"
    elif op == "d":
        chart.picker = ""
    elif op == "k" and arg.isdigit() and int(arg) < len(chart.offered):
        key = chart.offered[int(arg)]
        if key in chart.keys:
            if len(chart.keys) > 1:
                chart.keys.remove(key)
        else:
            chart.keys.append(key)
    elif op == "a" and arg.isdigit() and int(arg) < len(chart.candidates):
        run_id = chart.candidates[int(arg)]
        if run_id in chart.runs:
            if run_id != chart.runs[0]:
                chart.runs.remove(run_id)
        elif len(chart.runs) < MAX_RUNS:
            chart.runs.append(run_id)
    return chart


def keyboard_rows(chart_id: str, chart: ChartState, names: dict[str, str]) -> list[list[tuple]]:
    """The buttons as rows of (label, callback data); `names` maps run id to name.

    Kept free of Telegram types so it tests on its own; the bot wraps them."""
    def data(op, arg=""):
        return f"cv:{chart_id}:{op}" + (f":{arg}" if arg != "" else "")

    if chart.picker == "metrics":
        toggles = [(("✅ " if key in chart.keys else "▫️ ") + key.removeprefix("system/"),
                    data("k", i)) for i, key in enumerate(chart.offered)]
        rows = [toggles[i:i + 2] for i in range(0, len(toggles), 2)]
        return rows + [[("✔️ Done", data("d"))]]
    if chart.picker == "runs":
        toggles = [(("✅ " if run_id in chart.runs else "▫️ ") + names.get(run_id, run_id[:8]),
                    data("a", i)) for i, run_id in enumerate(chart.candidates)]
        rows = [toggles[i:i + 2] for i in range(0, len(toggles), 2)]
        return rows + [[("✔️ Done", data("d"))]]
    ranges = [(("• " if chart.mode == mode else "") + label, data("r", mode))
              for mode, label in RANGES]
    smooth = "Smooth off" if not chart.smooth else f"Smooth {chart.smooth:g}"
    return [ranges,
            [("Log ✓" if chart.log else "Log", data("l")), (smooth, data("s")),
             ("📏 Metrics", data("m")), ("➕ Runs", data("o"))]]


def caption(chart: ChartState, names: dict[str, str]) -> str:
    runs = " vs ".join(names.get(r, r[:8]) for r in chart.runs)
    parts = [dict(RANGES)[chart.mode]]
    if chart.log:
        parts.append("log y")
    if chart.smooth:
        parts.append(f"EMA {chart.smooth:g}")
    if chart.picker == "metrics":
        parts.append("tap metrics to show or hide")
    elif chart.picker == "runs":
        parts.append("tap runs to overlay")
    return f"{runs}\n{', '.join(chart.keys)} · {' · '.join(parts)}"


# -- data ------------------------------------------------------------------

def to_steps(system: list[tuple[float, float]], clock: list[tuple[float, int]]):
    """System samples (timestamp, value) placed on the step axis by interpolating
    the run's own (timestamp, step) points, so they share the x axis with the losses."""
    if len(clock) < 2 or not system:
        return []
    times = np.array([t for t, _ in clock])
    steps = np.array([s for _, s in clock])
    inside = [(t, v) for t, v in system if times[0] <= t <= times[-1]]
    return [(float(np.interp(t, times, steps)), v) for t, v in inside]


def gather(source: RunSource, chart: ChartState) -> tuple[list[tuple[str, dict]], dict, int | None]:
    """[(run name, {key: [(step, value)]})] for the chart, run id → name, and the
    first run's best evaluation step (for "Since best")."""
    runs, names, best_step = [], {}, None
    for index, run_id in enumerate(chart.runs):
        run = source.get_run(run_id)
        name = run.name or run_id[:8]
        if name in names.values():
            name = f"{name} ({run_id[:4]})"
        names[run_id] = name
        series = {}
        clock = None
        for key in chart.keys:
            if key not in run.metrics:
                continue
            points = source.metric_history(run_id, key)
            if key.startswith("system/"):
                if clock is None:
                    train = next((k for k in TRAIN_KEYS if k in run.metrics), None)
                    clock = [(p.timestamp, p.step) for p in source.metric_history(run_id, train)] \
                        if train else []
                series[key] = to_steps([(p.timestamp, p.value) for p in points], clock)
            else:
                series[key] = [(p.step, p.value) for p in points]
        if index == 0 and "eval/validation" in run.metrics:
            history = source.metric_history(run_id, "eval/validation")
            if history:
                best_step = min(history, key=lambda p: p.value).step
        runs.append((name, series))
    return runs, names, best_step
