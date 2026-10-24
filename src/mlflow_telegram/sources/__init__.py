"""Where runs come from: a small protocol the bot reads through, and its MLflow
implementation. Another tracker (W&B, TensorBoard logs, a database) can be
plugged in by implementing `RunSource`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["RunInfo", "Point", "RunSource", "MlflowSource"]


@dataclass
class Point:
    """One logged value of a metric."""

    step: int
    value: float
    timestamp: float  # epoch seconds


@dataclass
class RunInfo:
    """One run, as the bot sees it, whatever tracker it came from."""

    id: str
    name: str
    experiment_id: str
    experiment: str
    status: str                       # RUNNING, FINISHED, FAILED, KILLED, ...
    start_time: float                 # epoch seconds
    end_time: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)   # latest value per key
    params: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class RunSource(Protocol):
    """What the bot needs from a tracker. Read-mostly: the only writes are a
    run's status and its note."""

    def experiments(self) -> dict[str, str]:
        """Experiment id → name."""

    def list_runs(self, *, experiment_ids: list[str] | None = None, status: str | None = None,
                  since: float | None = None, limit: int = 50) -> list[RunInfo]:
        """Runs, newest first; `since` is an epoch time their start must follow."""

    def get_run(self, run_id: str) -> RunInfo:
        """One run by id (raises if unknown)."""

    def find_runs(self, name: str) -> list[RunInfo]:
        """Runs called `name`, newest first; else whose name contains it; else whose
        id starts with it."""

    def metric_history(self, run_id: str, key: str) -> list[Point]:
        """Every logged value of `key`, by step."""

    def mark_killed(self, run_id: str) -> None:
        """Records the run as KILLED (its process is not touched)."""

    def set_note(self, run_id: str, text: str) -> None:
        """Sets the run's note."""


def __getattr__(name):
    if name == "MlflowSource":
        from mlflow_telegram.sources.mlflow import MlflowSource
        return MlflowSource
    raise AttributeError(name)
