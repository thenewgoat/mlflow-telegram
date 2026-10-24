"""The monitoring package against a real (temporary sqlite) MLflow store, without a network:
what a Monitor records, how the bot finds and reads runs, the watcher's rules, the charts."""

import json
import os
import socket
import time
import urllib.request

import pytest

mlflow = pytest.importorskip("mlflow")

from mlflow_telegram import alerts, charts, state  # noqa: E402
from mlflow_telegram import runs as run_views  # noqa: E402
from mlflow_telegram.monitor import Monitor, Telegram  # noqa: E402
from mlflow_telegram.runs import Point, RunView, parse_command, seconds_per_step, stale_after  # noqa: E402
from mlflow_telegram.sources.mlflow import MlflowSource as Store  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # artifacts land in ./mlruns
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    yield uri, Store(uri)
    if mlflow.active_run():
        mlflow.end_run()


def _monitor(uri, name, **options):
    return Monitor("train.py", name, environ={"MLFLOW_TRACKING_URI": uri},
                   system_metrics=False, **options)


def test_monitor_logs_config_tags_checkpoint_and_artifact(store, tmp_path):
    uri, runs = store
    history = tmp_path / "history.jsonl"
    history.write_text('{"step": 1}\n')
    with _monitor(uri, "demo-a", params={"command": "python train.py --lr 3e-4"},
                  config={"lr": "3e-4", "batch": 32, "betas": [0.9, 0.98], "unset": None},
                  out=str(tmp_path), max_steps=1000, patience=3,
                  artifact=str(history)) as monitor:
        monitor.progress(10, training_mean=5.0, learning_rate=1e-4)
        monitor.evaluation(10, best=True, checkpoint="/x/best.pt", validation=4.0)
        monitor.evaluation(20, best=False, validation=4.5)
        monitor.progress(30, training_mean=float("nan"))
    run = runs.find_runs("demo-a")[0]
    assert run.status == "FINISHED"
    assert run.params["batch"] == "32" and run.params["betas"] == "[0.9, 0.98]"
    assert "unset" not in run.params and "command" in run.params
    tags = run.tags
    assert tags["monitor.pid"] == str(os.getpid())
    assert tags["monitor.host"] == socket.gethostname()
    assert tags["monitor.max_steps"] == "1000" and tags["monitor.patience"] == "3"
    assert tags["monitor.out"] == str(tmp_path)
    assert tags["best_checkpoint"] == "/x/best.pt" and tags["best_step"] == "10"
    assert tags["monitor.nonfinite"] == "training_mean@30"
    assert run.metrics["patience_used"] == 1
    artifacts = runs.client.list_artifacts(run.id)
    assert [a.path for a in artifacts] == ["history.jsonl"]


def test_store_finds_by_name_newest_first_then_substring(store):
    uri, runs = store
    for name in ("demo-b", "demo-b", "other"):
        with _monitor(uri, name) as monitor:
            monitor.progress(1, training_mean=1.0)
        time.sleep(0.01)
    found = runs.find_runs("demo-b")
    assert len(found) == 2 and found[0].start_time >= found[1].start_time
    assert [r.name for r in runs.find_runs("oth")] == ["other"]
    assert runs.find_runs(found[0].id[:10])[0].id == found[0].id
    assert runs.find_runs("nothing-like-it") == []


def test_view_progress_eta_best_and_live_pid(store):
    uri, runs = store
    monitor = _monitor(uri, "demo-c", max_steps=100, patience=5).start()
    for step in (10, 20, 30):
        monitor.progress(step, training_mean=10.0 / step, seconds_per_step=2.0)
        monitor.evaluation(step, best=step != 30, validation=[3.0, 2.0, 2.5][step // 10 - 1])
    run = runs.find_runs("demo-c")[0]
    view = run_views.view(runs, run)
    assert view.live and view.step == 30 and view.max_steps == 100
    assert view.fraction == pytest.approx(0.3)
    assert view.eta == pytest.approx(140.0)
    assert view.best == ("eval/validation", 2.0, 20)
    assert view.train_key == "training_mean"
    assert view.pid_alive is True and not view.dead and not view.stale()
    assert alerts.for_run(view) == []
    monitor.finish()
    assert not run_views.view(runs, runs.get_run(run.id)).live


def test_dead_and_old_runs_from_tags_and_command(store):
    uri, runs = store
    client = runs.client
    experiment = client.create_experiment("old")
    run = client.create_run(experiment, run_name="demo-old", tags={
        "monitor.pid": "999999999", "monitor.host": socket.gethostname()})
    client.log_param(run.info.run_id, "command", "python train.py --max-steps 400 "
                     "--patience 0 --out runs/x")
    client.log_metric(run.info.run_id, "training_mean", 1.0, step=100,
                      timestamp=int((time.time() - 3600) * 1000))
    view = run_views.view(runs, runs.get_run(run.info.run_id))
    assert view.max_steps == 400 and not view.patience
    assert view.pid_alive is False and view.dead and view.stale()
    found = alerts.for_run(view)
    assert [a.key for a in found] == [f"{view.id}:died"] and found[0].can_kill
    runs.mark_killed(view.id)
    assert runs.get_run(view.id).status == "KILLED"


def test_parse_command():
    assert parse_command("python -m x --max-steps 9 --no-bf16 --lr=0.1 --out 'a b'") == {
        "max_steps": "9", "no_bf16": "true", "lr": "0.1", "out": "a b"}


def test_staleness_follows_the_run_rhythm_with_a_ten_minute_floor():
    assert stale_after([0, 60, 120, 180]) == 600
    assert stale_after([0, 900, 1800, 2700]) == 2700
    assert stale_after([0]) == 600


def test_seconds_per_step_from_timestamps():
    points = [Point(0, 1.0, 1000.0), Point(100, 1.0, 1050.0)]
    assert seconds_per_step({}, points) == pytest.approx(0.5)
    assert seconds_per_step({"steps_per_s": 4.0}, points) == 0.25


def _view(**fields):
    base = dict(id="r" * 32, name="demo", experiment="train.py", status="RUNNING",
                started=time.time() - 3600, ended=None, metrics={}, params={}, tags={},
                last_seen=time.time() - 30)
    return RunView(**{**base, **fields})


def test_stalled_without_a_pid_tag():
    now = time.time()
    view = _view(last_seen=now - 1200)
    [alert] = alerts.for_run(view, now=now)
    assert alert.key.endswith(":stalled") and alert.can_kill


def test_diverging_and_nonfinite():
    points = [Point(i, 1.0 + 0.01 * i, 0) for i in range(10)] + [Point(10, 5.0, 0)]
    view = _view(train_key="training_mean", histories={"training_mean": points})
    assert alerts.diverging(view).key.endswith(":diverging")
    assert alerts.diverging(_view(train_key="training_mean",
                                  histories={"training_mean": points[:-1]})) is None
    assert alerts.diverging(_view(tags={"monitor.nonfinite": "loss@5"})).key.endswith(
        ":nonfinite")


def test_gpu_idle_needs_five_quiet_minutes():
    now = time.time()
    idle = [((now - s) * 1000, 3.0) for s in (300, 240, 180, 120, 60, 10)]
    assert alerts.gpu_idle(_view(), idle, now).key.endswith(":gpu-idle")
    busy = idle[:-1] + [((now - 10) * 1000, 90.0)]
    assert alerts.gpu_idle(_view(), busy, now) is None
    assert alerts.gpu_idle(_view(), idle[-2:], now) is None


def test_early_stop_near_and_disk_low():
    view = _view(patience=5, metrics={"patience_used": 4.0}, tags={"best_step": "300"})
    assert alerts.early_stop_near(view).key.endswith(":patience@300")
    assert alerts.early_stop_near(_view(patience=5, metrics={"patience_used": 2.0})) is None
    assert alerts.disk_low(5 * 2**30, 20, "/data").key == "host:disk-low"
    assert alerts.disk_low(50 * 2**30, 20, "/data") is None


def test_charts_render_pngs():
    series = {"training_mean": [(i, 10 / (i + 1)) for i in range(20)],
              "learning_rate": [(i, 1e-4) for i in range(20)],
              "eval/validation": [(i, 5 / (i + 1)) for i in range(0, 20, 5)]}
    assert charts.run_panels(series, "run").startswith(PNG)
    assert charts.compare({"a": series["training_mean"], "b": series["training_mean"]},
                          "training_mean").startswith(PNG)
    gpu = {"system/gpu_0_utilization_percentage": [(t * 10_000, 90.0) for t in range(5)],
           "system/gpu_0_memory_usage_megabytes": [(t * 10_000, 5000.0) for t in range(5)]}
    assert charts.system(gpu, "sys").startswith(PNG)
    assert charts.run_panels({}, "empty") is None


def test_state_round_trip_and_mute_reaches_the_monitor(tmp_path, monkeypatch):
    path = str(tmp_path / "state.json")
    state.write(path, {"mute_until": time.time() + 600})
    assert state.muted(state.read(path))
    sent = []
    monkeypatch.setattr(Telegram, "_post_text", lambda self, text: sent.append(text))
    environ = {"TELEGRAM_BOT_TOKEN": "1:x", "TELEGRAM_CHAT_ID": "1", "TRAINBOT_STATE": path}
    with Monitor("s", "r", environ=environ) as monitor:
        monitor.progress(1, loss=1.0)
    assert [t.split()[0] for t in sent] == ["▶️", "✅"]   # start and finish, no progress


def test_eval_push_sends_a_chart_and_falls_back_to_text(monkeypatch):
    calls = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"{}"

    def urlopen(request, data=None, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        calls.append(url.rsplit("/", 1)[-1])
        if url.endswith("sendPhoto"):
            raise OSError("photo refused")
        return Reply()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    environ = {"TELEGRAM_BOT_TOKEN": "1:x", "TELEGRAM_CHAT_ID": "1"}
    with Monitor("s", "r", environ=environ, every_minutes=0) as monitor:
        monitor.evaluation(1, validation=2.0)
        monitor.evaluation(2, validation=1.0, best=True)
    assert calls == ["sendMessage", "sendMessage", "sendPhoto", "sendMessage", "sendMessage"]


def test_bot_text_helpers():
    pytest.importorskip("telegram")
    from mlflow_telegram.bot import format as bot
    assert bot.parse_duration("90") == 5400
    assert bot.parse_duration("1h30m") == 5400
    assert bot.parse_duration("2x") is None
    now = time.time()
    view = _view(step=50, max_steps=200, seconds_per_step=1.0, metrics={"training_mean": 2.5},
                 train_key="training_mean", best=("eval/validation", 1.5, 40))
    text = bot.card(view, now)
    assert "step 50/200" in text and "ETA 2m" in text and "best eval/validation" in text
    other = _view(params={"lr": "1", "seed": "0", "command": "x"})
    first = _view(params={"lr": "2", "seed": "0", "command": "y"})
    assert bot.differing(first, other) == ["lr: <code>2</code> → <code>1</code>"]
    status = bot.status_text([view], "GPU 1%", "disk 9 GB free", now, now + 60)
    assert "demo" in status and "muted" in status
    assert json.dumps(bot.card_keyboard(view, now).to_dict())


def test_cleanup_marks_only_dead_runs_and_reverts(store, tmp_path):
    from mlflow_telegram import cleanup
    uri, runs = store
    client = runs.client
    experiment = client.create_experiment("old2")
    stamp = int((time.time() - 5 * 3600) * 1000)
    quiet = client.create_run(experiment, run_name="quiet", start_time=stamp)
    client.log_metric(quiet.info.run_id, "loss", 1.0, step=1, timestamp=stamp)
    fresh = client.create_run(experiment, run_name="fresh")
    client.log_metric(fresh.info.run_id, "loss", 1.0, step=1)
    now = time.time()
    views = [run_views.view(runs, runs.get_run(r.info.run_id), now) for r in (quiet, fresh)]
    found = [v for v in views if cleanup.dead(v, 2.0, now)]
    assert [v.name for v in found] == ["quiet"]
    record = str(tmp_path / "killed.jsonl")
    assert [e["run_name"] for e in cleanup.mark(runs, found, record)] == ["quiet"]
    assert runs.get_run(quiet.info.run_id).status == "KILLED"
    assert cleanup.revert(runs, record) == 1
    assert runs.get_run(quiet.info.run_id).status == "RUNNING"


def test_long_params_page_under_the_limit_each_with_a_keyboard():
    pytest.importorskip("telegram")
    from mlflow_telegram.bot import format as bot
    params = {f"parameter_number_{i:03d}_with_a_long_name": "v<&>" * 40 for i in range(200)}
    params.update(lr="3e-4", batch="32", empty="", command="python -m x " + "--a b " * 500)
    view = _view(params=params)
    pages = bot.params_pages(view)
    assert len(pages) > 1
    assert all(len(page) <= bot.PAGE_LIMIT for page in pages)
    assert "lr" in pages[0].split("<pre>")[1].split("\n")[0]       # key params first
    assert "empty" not in "".join(pages) and "--a b" not in "".join(pages)
    for index in range(len(pages)):
        rows = bot.params_keyboard(view.id, index, len(pages)).to_dict()["inline_keyboard"]
        data = [button["callback_data"] for row in rows for button in row]
        assert f"card:{view.id}" in data and f"pfile:{view.id}" in data
        assert all(len(d.encode()) <= 64 for d in data)
        assert (f"params:{view.id}:{index + 1}" in data) == (index < len(pages) - 1)
    assert b"lr = 3e-4" in bot.params_file(view)


def test_fit_cuts_at_a_line_and_closes_tags():
    pytest.importorskip("telegram")
    from mlflow_telegram.bot import format as bot
    text = "<b>title</b>\n<pre>" + "\n".join("x" * 50 for _ in range(200)) + "</pre>"
    short = bot.fit(text, 1024)
    assert len(short) <= 1024 and short.endswith("</pre>\n…")
    assert bot.fit("short", 10) == "short" and bot.fit(None, 10) is None
