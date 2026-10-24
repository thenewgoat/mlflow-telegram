"""Curves as PNG bytes for Telegram: a run's panels, runs overlaid, system usage.

Static images (a phone shows them inline), so no hover; every chart has one y
axis, a legend whenever it has two or more series, the dataviz reference
palette in its fixed order, 2px lines and recessive grid and axes.
"""

import io
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
          "#e34948"]

# Training metrics worth a curve; the rest (learning rate, speed) are numbers on the card.
TRAIN_KEYS = ("training_mean", "train_loss", "loss", "training")
SKIP = ("learning_rate", "lr", "seconds_per_step", "steps_per_s", "patience_used",
        "elapsed")


def _axes(ax, title: str, xlabel: str = "step") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _log_scale(values: list[float]) -> bool:
    """A log axis for a positive curve spanning two orders of magnitude or more."""
    positive = [v for v in values if v > 0]
    return len(positive) == len(values) and max(positive) / min(positive) >= 100


def _plot(ax, label: str, points, color: str, show_best: bool = False,
          markers: bool | None = None) -> None:
    steps = [s for s, _ in points]
    values = [v for _, v in points]
    markers = len(points) <= 30 if markers is None else markers
    ax.plot(steps, values, color=color, linewidth=2, label=label,
            marker="o" if markers else None, markersize=4)
    if show_best and values:
        index = min(range(len(values)), key=values.__getitem__)
        ax.plot(steps[index], values[index], marker="o", markersize=9, color=color,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        right = index > len(values) / 2   # keep the label inside the plot
        ax.annotate(f"best {values[index]:.4g} @ {steps[index]}",
                    (steps[index], values[index]), textcoords="offset points",
                    xytext=(-8 if right else 8, -14), ha="right" if right else "left",
                    fontsize=8, color=INK)


def _png(figure, dpi: int = 140) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def _panels(series: dict) -> list[tuple[str, dict]]:
    """Training curves in one panel, each evaluation metric in its own (they
    rarely share a scale), at most four panels."""
    train = {k: v for k, v in series.items() if k in TRAIN_KEYS}
    evals = {k: v for k, v in series.items() if k.startswith("eval/")}
    panels = []
    if train:
        panels.append(("training", train))
    for key, points in evals.items():
        panels.append((key, {key: points}))
    if not panels:
        rest = {k: v for k, v in series.items() if k not in SKIP and not k.startswith("system/")}
        panels = [(key, {key: points}) for key, points in list(rest.items())[:2]]
    return panels[:4]


def run_panels(series: dict, title: str) -> bytes | None:
    """One run's curves: `series` maps metric key to [(step, value), ...]."""
    panels = _panels(series)
    if not panels:
        return None
    figure, axes = plt.subplots(len(panels), 1, figsize=(7, 2.6 * len(panels) + 0.4),
                                squeeze=False)
    figure.patch.set_facecolor(SURFACE)
    figure.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    for ax, (name, group) in zip(axes[:, 0], panels):
        _axes(ax, name)
        values = []
        for index, (key, points) in enumerate(group.items()):
            _plot(ax, key, points, SERIES[index % len(SERIES)],
                  show_best=key.startswith("eval/"))
            values += [v for _, v in points]
        if values and _log_scale(values):
            ax.set_yscale("log")
        if len(group) >= 2:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    figure.tight_layout()
    return _png(figure)


def compare(runs: dict, metric: str, title: str = "") -> bytes | None:
    """Several runs' `metric` overlaid: `runs` maps run label to [(step, value), ...].

    More runs than palette slots are cut, never given generated colours."""
    runs = {label: points for label, points in runs.items() if points}
    if not runs:
        return None
    figure, ax = plt.subplots(figsize=(7, 3.6))
    figure.patch.set_facecolor(SURFACE)
    _axes(ax, title or metric)
    values = []
    markers = all(len(points) <= 30 for points in runs.values())
    for index, (label, points) in enumerate(list(runs.items())[:len(SERIES)]):
        _plot(ax, label, points, SERIES[index], show_best=metric.startswith("eval/"),
              markers=markers)
        values += [v for _, v in points]
    if _log_scale(values):
        ax.set_yscale("log")
    if len(runs) >= 2:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    figure.tight_layout()
    return _png(figure)


def system(series: dict, title: str) -> bytes | None:
    """System metrics against minutes since the first sample: utilisation
    percentages in one panel, GPU memory in another."""
    percent = {k: v for k, v in series.items() if k.endswith("_percentage") and v}
    memory = {k: v for k, v in series.items()
              if "gpu" in k and k.endswith("memory_usage_megabytes") and v}
    panels = [(name, group) for name, group in (("utilisation %", percent),
                                                ("GPU memory MB", memory)) if group]
    if not panels:
        return None
    start = min(points[0][0] for _, group in panels for points in group.values())
    figure, axes = plt.subplots(len(panels), 1, figsize=(7, 2.6 * len(panels) + 0.4),
                                squeeze=False)
    figure.patch.set_facecolor(SURFACE)
    figure.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    for ax, (name, group) in zip(axes[:, 0], panels):
        _axes(ax, name, xlabel="minutes")
        for index, (key, points) in enumerate(list(group.items())[:len(SERIES)]):
            label = key.removeprefix("system/").removesuffix("_percentage")
            minutes = [((t - start) / 60_000, v) for t, v in points]
            _plot(ax, label, minutes, SERIES[index])
        if name.endswith("%"):
            ax.set_ylim(0, 100)
        if len(group) >= 2:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    figure.tight_layout()
    return _png(figure)


def finite(points) -> list:
    return [(s, v) for s, v in points if v is not None and math.isfinite(v)]


# -- the explorer: one chart the bot's buttons reshape ----------------------

DASHES = ["-", "--", ":", "-."]
LOSS_WORDS = ("loss", "training", "validation")


def ema(values: list[float], weight: float) -> list[float]:
    """TensorBoard-style exponential smoothing, debiased so the start is not dragged to 0."""
    if not weight:
        return list(values)
    out, last, count = [], 0.0, 0
    for value in values:
        count += 1
        last = weight * last + (1 - weight) * value
        out.append(last / (1 - weight ** count))
    return out


def panel_of(key: str) -> str:
    """Which panel a metric shares: every loss in one (same scale), every other
    metric in its own, so no chart needs a second y axis."""
    name = key.lower()
    if key.startswith("eval/") and any(w in name for w in ("f1", "acc", "precision", "recall")):
        return key
    if key.startswith("eval/") or any(word in name for word in LOSS_WORDS):
        return "loss"
    return key


def visible_range(steps: list[int], mode: str, best_step: int | None) -> tuple[float, float]:
    """The x range a range button shows: all, the last 50% or 20%, or from
    shortly before the best evaluation."""
    if not steps:
        return 0.0, 1.0
    low, high = min(steps), max(steps)
    span = max(high - low, 1)
    if mode == "50":
        return high - 0.5 * span, high
    if mode == "20":
        return high - 0.2 * span, high
    if mode == "best" and best_step is not None:
        return max(low, best_step - 0.15 * span), high
    return low, high


def robust_limits(values_by_step: list[tuple[float, float]], x0: float, x1: float,
                  log: bool, skip_head: bool) -> tuple[float, float, bool] | None:
    """y limits from the 2nd–98th percentile of what is visible, leaving out the
    first fifth of the range when it is the full curve (the fast early descent).
    The flag says whether visible points fall outside (clipped)."""
    import numpy as np

    visible = [(s, v) for s, v in values_by_step if x0 <= s <= x1 and math.isfinite(v)
               and (v > 0 or not log)]
    if not visible:
        return None
    tail = visible
    if skip_head:
        cut = x0 + 0.2 * (x1 - x0)
        tail = [(s, v) for s, v in visible if s >= cut]
        if len(tail) < 5:
            tail = visible
    values = np.array([v for _, v in tail])
    if log:
        values = np.log10(values)
    # Percentiles only for the full curve; a zoomed range shows everything in it.
    low, high = ((np.percentile(values, 2), np.percentile(values, 98)) if skip_head
                 else (values.min(), values.max()))
    if high <= low:
        spread = 0.1 if log else max(abs(high) * 0.05, 1e-9)
        low, high = low - spread, high + spread
    pad = 0.08 * (high - low)
    low, high = low - pad, high + pad
    if log:
        low, high = 10 ** low, 10 ** high
    clipped = any(v > high or v < low for _, v in visible)
    return float(low), float(high), clipped


def explore(runs: list[tuple[str, dict]], keys: list[str], mode: str = "all",
            log: bool = False, smooth: float = 0.0, best_step: int | None = None,
            title: str = "", dpi: int = 140) -> bytes | None:
    """The bot's interactive chart as a PNG.

    `runs` is [(label, {key: [(step, value), ...]})]. Losses share one panel,
    each other metric gets its own, all sharing the step axis (at most four
    panels). Colour follows the run when several are overlaid (line style then
    tells the metrics apart), else the metric."""
    keys = [k for k in keys if any(series.get(k) for _, series in runs)]
    if not keys:
        return None
    panels: dict[str, list[str]] = {}
    for key in keys:
        panels.setdefault(panel_of(key), []).append(key)
    names = list(panels)[:4]
    steps = [s for _, series in runs for k in keys for s, _ in series.get(k, [])]
    x0, x1 = visible_range(steps, mode, best_step)
    overlay = len(runs) > 1
    plt.rcParams.update({"font.size": 11})
    figure, axes = plt.subplots(len(names), 1, figsize=(7.2, 1.0 + 3.0 * len(names)),
                                squeeze=False, sharex=True)
    figure.patch.set_facecolor(SURFACE)
    if title:
        figure.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    notes = []
    for ax, name in zip(axes[:, 0], names):
        label_ = panels[name][0] if len(panels[name]) == 1 else "losses"
        _axes(ax, label_, xlabel="")
        ax.title.set_fontsize(12)
        ax.tick_params(labelsize=10)
        everything = []
        for r, (label, series) in enumerate(runs):
            for k, key in enumerate(panels[name]):
                points = [(s, v) for s, v in series.get(key, []) if math.isfinite(v)]
                if not points:
                    continue
                color = SERIES[(r if overlay else keys.index(key)) % len(SERIES)]
                dash = DASHES[k % len(DASHES)] if overlay else "-"
                xs = [s for s, _ in points]
                ys = [v for _, v in points]
                legend = (f"{label} · {key}" if overlay and len(panels[name]) > 1
                          else label if overlay else key)
                if smooth:
                    ax.plot(xs, ys, color=color, linewidth=1, alpha=0.25, linestyle=dash)
                    ys = ema(ys, smooth)
                ax.plot(xs, ys, color=color, linewidth=2, linestyle=dash, label=legend,
                        marker="o" if len(xs) <= 30 else None, markersize=3.5)
                if key.startswith("eval/") and not smooth:
                    pick = max if "f1" in key or "acc" in key else min
                    inside = [(s, v) for s, v in points if x0 <= s <= x1] or points
                    s, v = pick(inside, key=lambda p: p[1])
                    ax.plot(s, v, marker="o", markersize=8, color=color,
                            markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
                everything += list(zip(xs, ys))
        if log and all(v > 0 for _, v in everything):
            from matplotlib.ticker import FuncFormatter
            ax.set_yscale("log")
            plain = FuncFormatter(lambda value, _: f"{value:.4g}")
            ax.yaxis.set_major_formatter(plain)
            ax.yaxis.set_minor_formatter(plain)
        limits = robust_limits(everything, x0, x1, log and ax.get_yscale() == "log",
                               skip_head=mode == "all")
        if limits:
            ax.set_ylim(limits[0], limits[1])
            if ax.get_yscale() == "log" and limits[1] / limits[0] >= 10:
                from matplotlib.ticker import NullFormatter
                ax.yaxis.set_minor_formatter(NullFormatter())   # decades are labelled
            if limits[2]:
                ax.text(0.99, 0.97, "some values off-scale", transform=ax.transAxes,
                        ha="right", va="top", fontsize=9, color=INK_2)
        if len(ax.get_lines()) and (overlay or len(panels[name]) > 1):
            ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right",
                      bbox_to_anchor=(1, 0.9))
    for ax in axes[:, 0]:
        ax.set_xlim(x0, x1 + (x1 - x0) * 0.01)
    axes[-1, 0].set_xlabel("step", color=INK_2, fontsize=10)
    if len(panels) > 4:
        notes.append(f"{len(panels) - 4} more panels not shown")
    if notes:
        figure.text(0.01, 0.005, "; ".join(notes), fontsize=9, color=INK_2)
    figure.tight_layout()
    return _png(figure, dpi)
