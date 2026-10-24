"""Mark runs that died without telling MLflow as KILLED, keeping a record to undo it.

    mlflow-telegram cleanup            (lists what it would mark)
    mlflow-telegram cleanup --apply    (marks them)

A run counts as dead when MLflow says RUNNING but its process is gone (`monitor.pid`
on this host) or, for runs without a pid tag, when it has been silent for more
than `--hours` (default 2) and longer than its own reporting rhythm. `--keep NAME`
protects a run by name. Every change is appended to
`marked-killed.jsonl` beside the state file (run id, name, old status, old end time);
`--revert FILE` puts those runs back to RUNNING.
"""

from __future__ import annotations

import json
import os
import time

from mlflow_telegram import runs as run_views
from mlflow_telegram.config import Config
from mlflow_telegram.runs import RunView
from mlflow_telegram.sources import RunSource


def record_path(config: Config | None = None) -> str:
    """Beside the state file: `marked-killed.jsonl`."""
    config = config or Config.from_env()
    return os.path.join(os.path.dirname(config.state_path) or ".", "marked-killed.jsonl")


def dead(view: RunView, hours: float, now: float) -> bool:
    if not view.live or view.pid_alive:
        return False
    if view.pid_alive is False:
        return True
    silent = now - (view.last_seen or view.started)
    return view.stale(now) and silent > hours * 3600


def mark(source: RunSource, views: list[RunView], path: str | None = None) -> list[dict]:
    """Marks `views` KILLED, appending one line per run to the record first."""
    path = path or record_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = []
    for view in views:
        run = source.get_run(view.id)
        if run.status != "RUNNING":
            continue
        entry = {"run_id": view.id, "run_name": view.name, "experiment": view.experiment,
                 "old_status": run.status, "old_end_time": run.end_time,
                 "marked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        with open(path, "a") as handle:
            handle.write(json.dumps(entry) + "\n")
        source.mark_killed(view.id)
        done.append(entry)
    return done


def revert(source, path: str) -> int:
    """Puts the runs in a record back to their old status (needs a source with
    `restore_status`, as `MlflowSource` has)."""
    count = 0
    with open(path) as handle:
        for line in handle:
            entry = json.loads(line)
            source.restore_status(entry["run_id"], entry["old_status"])
            count += 1
    return count


def run(source: RunSource, *, apply: bool = False, hours: float = 2.0,
        keep: list[str] | None = None, record: str | None = None) -> tuple[list, list]:
    """(views of the live runs, those found dead); marks the dead ones when `apply`."""
    now = time.time()
    views = [run_views.view(source, info, now) for info in run_views.live(source)]
    found = [v for v in views if v.name not in (keep or []) and dead(v, hours, now)]
    if apply:
        mark(source, found, record)
    return views, found
