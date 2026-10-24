"""`RunSource` over an MLflow tracking store (any URI MLflow accepts)."""

from __future__ import annotations

import time

from mlflow_telegram.sources import Point, RunInfo

__all__ = ["MlflowSource"]


class MlflowSource:
    """Runs from MLflow through `MlflowClient`.

    >>> source = MlflowSource("sqlite:///mlflow.db")      # doctest: +SKIP
    >>> [run.name for run in source.list_runs(status="RUNNING")]   # doctest: +SKIP
    """

    def __init__(self, tracking_uri: str | None = None, client=None):
        if client is None:
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=tracking_uri)
        self.client = client
        self._experiments: dict[str, str] = {}

    # -- experiments ---------------------------------------------------------

    def experiments(self) -> dict[str, str]:
        if not self._experiments:
            self._experiments = {e.experiment_id: e.name
                                 for e in self.client.search_experiments()}
        return self._experiments

    def experiment_name(self, experiment_id: str) -> str:
        if experiment_id not in self.experiments():
            self._experiments = {}
        return self.experiments().get(experiment_id, experiment_id)

    def experiment_id(self, name: str) -> str | None:
        for refresh in (False, True):
            if refresh:
                self._experiments = {}
            for key, value in self.experiments().items():
                if value == name:
                    return key
        return None

    # -- runs ----------------------------------------------------------------

    def _info(self, run) -> RunInfo:
        info, data = run.info, run.data
        return RunInfo(
            id=info.run_id, name=info.run_name or info.run_id[:8],
            experiment_id=info.experiment_id,
            experiment=self.experiment_name(info.experiment_id), status=info.status,
            start_time=info.start_time / 1000.0,
            end_time=info.end_time / 1000.0 if info.end_time else None,
            metrics=dict(data.metrics), params=dict(data.params), tags=dict(data.tags))

    def _search(self, filter_string: str = "", limit: int = 50,
                experiment_ids: list[str] | None = None) -> list[RunInfo]:
        runs = self.client.search_runs(
            experiment_ids=experiment_ids or list(self.experiments()),
            filter_string=filter_string, max_results=limit,
            order_by=["attributes.start_time DESC"])
        return [self._info(run) for run in runs]

    def list_runs(self, *, experiment_ids: list[str] | None = None, status: str | None = None,
                  since: float | None = None, limit: int = 50) -> list[RunInfo]:
        clauses = []
        if status:
            clauses.append(f"attributes.status = '{status}'")
        if since is not None:
            clauses.append(f"attributes.start_time > {int(since * 1000)}")
        return self._search(" and ".join(clauses), limit, experiment_ids)

    def get_run(self, run_id: str) -> RunInfo:
        return self._info(self.client.get_run(run_id))

    def find_runs(self, name: str, limit: int = 2000) -> list[RunInfo]:
        """Matched here rather than in an MLflow filter string, so no user text is
        ever parsed as a filter (quotes, % and _ mean nothing special)."""
        runs = self._search(limit=limit)
        exact = [run for run in runs if run.name == name]
        if exact:
            return exact
        folded = name.lower()
        partial = [run for run in runs if folded in run.name.lower()]
        if partial:
            return partial
        if len(name) >= 6 and all(c in "0123456789abcdef" for c in name):
            return [run for run in runs if run.id.startswith(name)]
        return []

    def metric_history(self, run_id: str, key: str) -> list[Point]:
        points = [Point(m.step, m.value, m.timestamp / 1000.0)
                  for m in self.client.get_metric_history(run_id, key)]
        return sorted(points, key=lambda p: (p.step, p.timestamp))

    # -- the two writes ------------------------------------------------------

    def mark_killed(self, run_id: str) -> None:
        self.client.set_terminated(run_id, status="KILLED")

    def set_note(self, run_id: str, text: str) -> None:
        self.client.set_tag(run_id, "mlflow.note.content", text)

    def restore_status(self, run_id: str, status: str) -> None:
        """Undoes `mark_killed` (used by `mlflow-telegram cleanup --revert`)."""
        self.client.update_run(run_id, status=status)

    def recent(self, hours: float = 24.0) -> list[RunInfo]:
        """Runs started in the last `hours`, plus any still running."""
        started = self.list_runs(since=time.time() - hours * 3600, limit=100)
        seen = {run.id for run in started}
        return started + [run for run in self.list_runs(status="RUNNING", limit=100)
                          if run.id not in seen]
