"""Watching a training run from elsewhere: MLflow for the curves, Telegram for the news.

One `Monitor` per run. The training loop calls `progress` as it trains,
`evaluation` after each evaluation, and `finish` at the end; used as a context
manager it also reports a crash (or a Ctrl-C) with the end of the traceback,
then lets the exception through::

    from mlflow_telegram import Monitor

    with Monitor("train.py", name="lr-3e-4", config=vars(args), max_steps=10_000) as monitor:
        for step in range(1, 10_001):
            loss = train_step()
            monitor.progress(step, train_loss=loss, lr=scheduler.get_last_lr()[0])
            if step % 500 == 0:
                monitor.evaluation(step, best=improved, validation=val_loss,
                                   checkpoint="checkpoints/best.pt")

Both halves are off unless the environment turns them on (see
`mlflow_telegram.config`), so a test or a run nobody asked to watch touches
nothing: MLflow when `MLFLOW_TRACKING_URI` is set (the experiment is the
script's name unless `MLFLOW_EXPERIMENT_NAME` says otherwise), Telegram when a
bot token and chat id are set.

Start, finish and crash messages always go out. Progress and evaluation
messages go out at most once per `every_minutes` (default 30) each; the bot's
`/every` and `/mute`, kept in the shared state file, change that for runs
already going. Every message comes from a `Formatter`, which can be replaced.

What the bot needs to watch a run is recorded as tags when the run starts:
`monitor.pid` and `monitor.host` (so a run whose process is gone can be told
from a slow one), and `monitor.out`, `monitor.max_steps`, `monitor.patience`
when given. `config` is logged one param per key; with psutil installed MLflow
also samples CPU, memory, disk and (with pynvml) GPU as `system/...` metrics.
An evaluation marked `best` with a `checkpoint` tags `best_checkpoint`, and with
`patience` the evaluations since the best are logged as `patience_used`; a NaN
or infinite metric is tagged `monitor.nonfinite`. The `artifact` file is
attached to the run at the end.

Nothing here can stop a run: every MLflow and Telegram failure is printed and
dropped, and Telegram sends from a background thread so a slow network never
holds up a training step. Needs only the standard library (and mlflow, and
matplotlib for charts in evaluation messages, when used).
"""

from __future__ import annotations

import json
import math
import os
import socket
import sys
import time
import traceback

from mlflow_telegram import state as bot_state
from mlflow_telegram.config import Config, load_env_file  # noqa: F401 (re-exported)
from mlflow_telegram.notify import Telegram
from mlflow_telegram.redact import redact_params, redact_text

__all__ = ["Monitor", "Formatter", "Telegram", "load_env_file"]


def _number(value):
    """A finite float, or None for anything MLflow would reject."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _nonfinite(value) -> bool:
    """A number that is NaN or infinite (which `_number` drops)."""
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _format(metrics: dict) -> str:
    parts = []
    for key, value in metrics.items():
        number = _number(value)
        if number is None:
            parts.append(f"{key} {value}")
        elif number == int(number) and abs(number) < 1e9:
            parts.append(f"{key} {int(number)}")
        else:
            parts.append(f"{key} {number:.4g}")
    return ", ".join(parts)


def _duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def _system_metrics_available() -> bool:
    try:
        import psutil  # noqa: F401
    except ImportError:
        return False
    return True


class Formatter:
    """The Telegram messages a `Monitor` sends. Subclass and pass `formatter=` to
    change any of them; each returns the message text (HTML is not used)."""

    def start(self, title: str, host: str) -> str:
        return f"▶️ {title} started on {host}"

    def progress(self, title: str, step: int, elapsed: str, metrics: dict) -> str:
        return f"⏳ {title} step {step} ({elapsed}): {_format(metrics)}"

    def evaluation(self, title: str, step: int, best: bool, metrics: dict,
                   patience: tuple[int, int] | None) -> str:
        text = f"📊 {title} eval at step {step}{' ⭐ new best' if best else ''}: {_format(metrics)}"
        if patience:
            text += f" (patience {patience[0]}/{patience[1]})"
        return text

    def finish(self, title: str, elapsed: str, summary: dict) -> str:
        text = f"✅ {title} finished after {elapsed}"
        return text + (f": {_format(summary)}" if summary else "")

    def interrupted(self, title: str, where: str, elapsed: str) -> str:
        return f"⏹ {title} interrupted{where} after {elapsed}"

    def crashed(self, title: str, where: str, elapsed: str, error: BaseException,
                trace: str) -> str:
        text = f"💥 {title} crashed{where} after {elapsed}: {type(error).__name__}: {error}"
        return text + (f"\n\n{trace[-1500:]}" if trace else "")


class Monitor:
    """One training run, reported to whichever of MLflow and Telegram are on.

    `name` is what the run is called in both (a run directory's name is a good
    one); `experiment` is the MLflow experiment, by default `MLFLOW_EXPERIMENT_NAME`
    or else `script`. `params` and `config` are logged once at the start, each
    key its own param. `out`, `max_steps` and `patience` become tags the bot
    reads; `artifact` is a file attached to the run when it ends.
    """

    def __init__(self, script: str, name: str = "", params: dict | None = None,
                 experiment: str = "", every_minutes: float | None = None,
                 environ=None, *, config: dict | None = None, out: str | None = None,
                 max_steps: int | None = None, patience: int | None = None,
                 artifact: str | None = None, system_metrics: bool = True,
                 charts: bool = True, formatter: Formatter | None = None,
                 send_traceback: bool = True, redact: bool = True):
        real = environ is None
        settings = Config.from_env(None if real else environ)
        self.formatter = formatter or Formatter()
        self.send_traceback = send_traceback
        self.redact = redact
        self._secrets = [settings.token] if settings.token else []
        self.script = script
        self.name = name or time.strftime("%Y%m%d-%H%M%S")
        self.params = {**(config or {}), **(params or {})}
        self.out = out
        self.max_steps = max_steps
        self.patience = patience
        self.artifact = artifact
        self.charts = charts
        self.host = socket.gethostname()
        if every_minutes is None:
            every_minutes = settings.every_minutes
        self.every = every_minutes * 60.0
        # The bot's /mute and /every apply to runs reading the real environment;
        # a caller handing in its own `environ` sees them only if it names a state file.
        explicit = not real and any(environ.get(k) for k in ("MLFLOW_TELEGRAM_STATE",
                                                             "TRAINBOT_STATE"))
        self.state_path = settings.state_path if real or explicit else None
        self.started = None
        self.finished = False
        self.last_step = None
        self.since_best = 0
        self.history: dict[str, list[tuple[int, float]]] = {}
        self._flagged = False
        self._sent = {"progress": float("-inf"), "evaluation": float("-inf")}

        self.mlflow = None
        self.system_metrics = False
        uri = settings.tracking_uri
        if uri:
            try:
                import mlflow
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment(experiment or settings.experiment or script)
                self.mlflow = mlflow
                self.system_metrics = system_metrics and _system_metrics_available()
            except Exception as error:  # noqa: BLE001
                self._warn("mlflow is off", error)

        self.telegram = None
        if settings.telegram_ready:
            try:
                self.telegram = Telegram(settings.token, str(settings.chat_id))
            except ValueError as error:
                self._warn("telegram is off", error)

    @property
    def enabled(self) -> bool:
        return self.mlflow is not None or self.telegram is not None

    # -- the calls a training loop makes -----------------------------------

    def start(self) -> Monitor:
        self.started = time.monotonic()
        if self.mlflow is not None:
            try:
                if self.system_metrics:
                    self.mlflow.start_run(run_name=self.name, log_system_metrics=True)
                else:
                    self.mlflow.start_run(run_name=self.name)
                params = {key: (value if isinstance(value, (int, float, str, bool))
                                else json.dumps(value, default=str))
                          for key, value in self.params.items() if value is not None}
                if self.redact:   # secret-looking keys and --api-key style flags
                    params = redact_params(params)
                # MLflow caps a param value's length.
                self.mlflow.log_params({key: str(value)[:6000] for key, value in params.items()})
                tags = {"monitor.pid": os.getpid(), "monitor.host": self.host,
                        "monitor.script": self.script, "monitor.out": self.out,
                        "monitor.max_steps": self.max_steps, "monitor.patience": self.patience}
                for key, value in tags.items():
                    if value is not None:
                        self.mlflow.set_tag(key, str(value))
            except Exception as error:  # noqa: BLE001
                self._warn("mlflow start failed; mlflow is off for this run", error)
                self.mlflow = None
        self._say(self.formatter.start(self._title(), self.host))
        return self

    def progress(self, step: int, **metrics) -> None:
        """Training metrics: every call to MLflow, to Telegram once per interval."""
        self.last_step = step
        self._flag_nonfinite(step, metrics)
        self._log(metrics, step)
        if self._due("progress"):
            self._say(self.formatter.progress(self._title(), step, self._elapsed(), metrics))

    def evaluation(self, step: int, best: bool = False, checkpoint: str | None = None,
                   **metrics) -> None:
        """Evaluation metrics, logged under `eval/`; `best` marks a new best, whose
        `checkpoint` (a path) is tagged on the run."""
        self.last_step = step
        self._flag_nonfinite(step, metrics)
        logged = {f"eval/{key}": value for key, value in metrics.items()}
        if self.patience:
            self.since_best = 0 if best else self.since_best + 1
            logged["patience_used"] = self.since_best
        self._log(logged, step)
        if best and self.mlflow is not None:
            try:
                self.mlflow.set_tag("best_step", str(step))
                if checkpoint:
                    self.mlflow.set_tag("best_checkpoint", checkpoint)
            except Exception as error:  # noqa: BLE001
                self._warn("mlflow tag failed", error)
        if self._due("evaluation"):
            text = self.formatter.evaluation(
                self._title(), step, best, metrics,
                (self.since_best, self.patience) if self.patience else None)
            png = self._chart()
            if png is not None:
                self.telegram.send_photo(png, text)
            else:
                self._say(text)

    def finish(self, **summary) -> None:
        """The end of a run that ended normally; `summary` goes to both."""
        if self.finished:
            return
        self.finished = True
        if self.mlflow is not None:
            try:
                self._log({f"final/{key}": value for key, value in summary.items()}, None)
                self._attach()
                self.mlflow.end_run(status="FINISHED")
            except Exception as error:  # noqa: BLE001
                self._warn("mlflow finish failed", error)
        self._say(self.formatter.finish(self._title(), self._elapsed(), summary))
        self._close()

    def fail(self, error: BaseException) -> None:
        """The end of a run that raised: the error and the end of its traceback."""
        if self.finished:
            return
        self.finished = True
        interrupted = isinstance(error, KeyboardInterrupt)
        if self.mlflow is not None:
            try:
                self.mlflow.set_tag("error", self._clean(f"{type(error).__name__}: {error}")[:5000])
                self._attach()
                self.mlflow.end_run(status="KILLED" if interrupted else "FAILED")
            except Exception as problem:  # noqa: BLE001
                self._warn("mlflow end failed", problem)
        where = f" at step {self.last_step}" if self.last_step is not None else ""
        if interrupted:
            self._say(self.formatter.interrupted(self._title(), where, self._elapsed()))
        else:
            tail = ("".join(traceback.format_exception(type(error), error, error.__traceback__))
                    if self.send_traceback else "")
            self._say(self._clean(self.formatter.crashed(self._title(), where, self._elapsed(),
                                                         error, tail)))
        self._close()

    def __enter__(self) -> Monitor:
        return self.start()

    def __exit__(self, kind, error, trace) -> bool:
        if error is not None:
            if not isinstance(error, SystemExit) or error.code not in (0, None):
                self.fail(error)
            else:
                self.finish()
        else:
            self.finish()
        return False

    # -- internals ---------------------------------------------------------

    def _title(self) -> str:
        return f"{self.script} {self.name}"

    def _elapsed(self) -> str:
        return _duration(time.monotonic() - self.started) if self.started else "0m"

    def _due(self, kind: str) -> bool:
        if self.telegram is None:
            return False
        settings = bot_state.read(self.state_path)
        if settings.get("mute_until", 0) > time.time():
            return False
        every = _number(settings.get("every_minutes"))
        every = every * 60.0 if every else self.every
        now = time.monotonic()
        if now - self._sent[kind] < every:
            return False
        self._sent[kind] = now
        return True

    def _flag_nonfinite(self, step, metrics: dict) -> None:
        """Tags the first NaN or infinite metric, which MLflow cannot store but
        the bot's divergence alert wants to know about."""
        if self._flagged or self.mlflow is None:
            return
        bad = [key for key, value in metrics.items() if _nonfinite(value)]
        if bad:
            self._flagged = True
            try:
                self.mlflow.set_tag("monitor.nonfinite", f"{bad[0]}@{step}")
            except Exception as error:  # noqa: BLE001
                self._warn("mlflow tag failed", error)

    def _log(self, metrics: dict, step) -> None:
        values = {key: number for key, value in metrics.items()
                  if (number := _number(value)) is not None}
        if step is not None:
            for key, value in values.items():
                self.history.setdefault(key, []).append((step, value))
        if self.mlflow is None or not values:
            return
        try:
            self.mlflow.log_metrics(values, step=step)
        except Exception as error:  # noqa: BLE001
            self._warn("mlflow log failed", error)

    def _chart(self) -> bytes | None:
        """The run's curves so far as a PNG, or None (too few points, no matplotlib)."""
        if not self.charts:
            return None
        series = {key: points for key, points in self.history.items()
                  if len(points) >= 2 and key != "patience_used"}
        if not series:
            return None
        try:
            from mlflow_telegram import charts
            return charts.run_panels(series, title=self._title())
        except Exception as error:  # noqa: BLE001
            self._warn("chart failed", error)
            return None

    def _attach(self) -> None:
        if self.artifact and os.path.isfile(self.artifact):
            try:
                self.mlflow.log_artifact(self.artifact)
            except Exception as error:  # noqa: BLE001
                self._warn("mlflow artifact failed", error)

    def _say(self, text: str) -> None:
        if self.telegram is not None:
            self.telegram.send(text)

    def _close(self) -> None:
        if self.telegram is not None:
            self.telegram.close()

    def _clean(self, text: str) -> str:
        """`text` without the bot token, URL credentials, secret query values or
        the home directory (when `redact`)."""
        return redact_text(text, self._secrets) if self.redact else text

    def _warn(self, what: str, error: BaseException) -> None:
        secrets = getattr(self, "_secrets", [])
        print(f"[mlflow-telegram] {what}: {type(error).__name__}: "
              f"{redact_text(str(error), secrets, home=False)}", file=sys.stderr, flush=True)
