"""The chart explorer (buttons over a PNG) and the Mini App's server: state and
buttons, the robust y range, rendering variants, initData checks, downsampling
and the JSON API against a temporary MLflow store."""

import json

import pytest

mlflow = pytest.importorskip("mlflow")

from mlflow_telegram import charts, chartview  # noqa: E402
from mlflow_telegram.sources.mlflow import MlflowSource as Store  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n"
TOKEN = "123456:secret"
USER = 4242


def _series(scale=1.0, n=200):
    return {"training_mean": [(s, scale * (5000 / (s + 10) + 300)) for s in range(0, n * 100, 100)],
            "eval/validation": [(s, scale * (4000 / (s + 10) + 350)) for s in range(0, n * 100, 500)],
            "learning_rate": [(s, 1e-4) for s in range(0, n * 100, 100)]}


def test_buttons_change_state_and_stay_under_64_bytes():
    chart = chartview.ChartState(runs=["a" * 32], keys=["training_mean"],
                                 offered=["training_mean", "eval/validation", "learning_rate"],
                                 candidates=["a" * 32, "b" * 32])
    chartview.apply(chart, "r", "20")
    chartview.apply(chart, "l")
    chartview.apply(chart, "s")
    chartview.apply(chart, "k", "1")
    chartview.apply(chart, "a", "1")
    assert (chart.mode, chart.log, chart.smooth) == ("20", True, 0.6)
    assert chart.keys == ["training_mean", "eval/validation"] and chart.runs[1] == "b" * 32
    chartview.apply(chart, "k", "0")
    chartview.apply(chart, "k", "1")            # the last metric cannot be hidden
    assert chart.keys == ["eval/validation"]
    chartview.apply(chart, "a", "0")            # nor the chart's own run
    assert chart.runs[0] == "a" * 32
    chartview.apply(chart, "k", "99")           # out of range: nothing happens
    names = {"a" * 32: "demo-a", "b" * 32: "demo-b"}
    for picker in ("", "metrics", "runs"):
        chart.picker = picker
        rows = chartview.keyboard_rows("abcd1234", chart, names)
        assert all(len(data.encode()) <= 64 for row in rows for _, data in row)
    assert "demo-a vs demo-b" in chartview.caption(chart, names)


def test_states_survive_in_the_state_file_and_old_ones_go():
    state = {}
    for i in range(chartview.KEEP + 5):
        chart = chartview.ChartState(runs=["r"], keys=["k"], made=i)
        chartview.remember(state, f"id{i}", chart)
    assert len(state["charts"]) == chartview.KEEP and "id0" not in state["charts"]
    again = chartview.recall(json.loads(json.dumps(state)), f"id{chartview.KEEP + 4}")
    assert again.keys == ["k"] and chartview.recall(state, "nope") is None


def test_robust_limits_skip_the_early_descent():
    points = _series()["training_mean"]
    low, high, clipped = charts.robust_limits(points, 0, 19900, log=False, skip_head=True)
    assert high < 400 and clipped          # the 5000/10 start is off-scale
    low2, high2, _ = charts.robust_limits(points, 15000, 19900, log=False, skip_head=False)
    assert low2 < min(v for s, v in points if s >= 15000) <= high2


def test_explore_renders_overlays_log_smoothing_and_ranges():
    runs = [("demo-a", _series()), ("demo-b", _series(1.05))]
    keys = ["training_mean", "eval/validation", "learning_rate"]
    for mode in ("all", "50", "20", "best"):
        assert charts.explore(runs, keys, mode, best_step=12000).startswith(PNG)
    assert charts.explore(runs[:1], keys, "all", log=True, smooth=0.9).startswith(PNG)
    assert charts.explore(runs, ["no-such-key"]) is None
    assert charts.ema([1, 1, 1], 0.9) == pytest.approx([1, 1, 1])


def test_system_metrics_move_onto_the_step_axis():
    clock = [(1000.0, 0), (2000.0, 1000)]
    assert chartview.to_steps([(1500.0, 50.0), (3000.0, 9.0)], clock) == [(500.0, 50.0)]


def test_echarts_cache_rejects_a_wrong_hash(tmp_path):
    from mlflow_telegram import page
    (tmp_path / page.ECHARTS_FILE).write_text("tampered")
    assert page.echarts_source(str(tmp_path)) is None


def test_build_page_writes_the_site_without_data(tmp_path):
    from mlflow_telegram import page
    names = page.build_page(str(tmp_path), echarts="window.echarts = {};")
    assert names == sorted([".nojekyll", page.ECHARTS_FILE, "index.html", "robots.txt"])
    html = (tmp_path / "index.html").read_text()
    assert "fetch(" not in html and "EMBEDDED" not in html and page.ECHARTS_FILE in html
    assert html.index("__FRAGMENT") < html.index("telegram-web-app.js")


def test_album_resolution_png():
    runs = [("demo-a", _series())]
    png = charts.explore(runs, ["training_mean", "eval/validation"], "20", dpi=300)
    import struct
    width, height = struct.unpack(">II", png[16:24])
    assert png.startswith(PNG) and 1800 <= width <= 2560 and height <= 10000


# -- curves in the link (static Mini App) ------------------------------------

def _dense_payload(runs=3, n=20000):
    import math
    series = lambda i: {
        "training_mean": [(s, 5000 / (s / 50 + 1) + 300 + 10 * math.sin(s * (i + 1))) for s in range(n)],
        "eval/validation": [(s, 4000 / (s / 50 + 1) + 350) for s in range(0, n, 50)],
        "learning_rate": [(s, 3e-4 * (1 + math.cos(math.pi * s / n)) / 2) for s in range(n)]}
    return {"initial": {"runs": [f"r{i}" for i in range(runs)], "keys": ["training_mean"]},
            "runs": [{"id": f"r{i}", "name": f"demo-{i}", "experiment": "e", "series": series(i)}
                     for i in range(runs)]}


def test_fragment_round_trip_is_precise():
    from mlflow_telegram import payload as fragment
    payload = _dense_payload(runs=1, n=300)
    back = fragment.decode(fragment.encode(payload))
    for key, points in payload["runs"][0]["series"].items():
        decoded = back["runs"][0]["series"][key]
        assert [s for s, _ in decoded] == [s for s, _ in points]
        span = max(v for _, v in points) - min(v for _, v in points)
        assert all(abs(a[1] - b[1]) <= max(1e-4 * abs(a[1]), 1e-4 * span)
                   for a, b in zip(points, decoded))
    assert back["initial"] == payload["initial"]


def test_fragment_fits_a_budget_keeping_shape():
    from mlflow_telegram import payload as fragment
    payload = _dense_payload()
    text, limit = fragment.fit(payload, 4000)
    assert len(text) <= 4000 and limit is not None and limit >= 50
    points = [(i, (i % 100 == 0) * 50.0) for i in range(1000)]
    kept = fragment.lttb(points, 100)
    assert len(kept) == 100 and max(v for _, v in kept) == 50.0     # spikes survive


def _bot(tmp_path, **settings):
    from mlflow_telegram.bot.app import TrainBot
    from mlflow_telegram.config import Config
    store = Store(f"sqlite:///{tmp_path / 'mlflow.db'}")
    config = Config(token="1:x", chat_id=1, state_path=str(tmp_path / "s.json"), **settings)
    return TrainBot(config, store)


def test_chart_link_fits_the_limit_and_is_off_without_a_page(tmp_path, monkeypatch):
    pytest.importorskip("telegram")
    monkeypatch.chdir(tmp_path)
    chart = chartview.ChartState(runs=["r0"], keys=["training_mean"])
    packed = _dense_payload(runs=1)
    runs = [(packed["runs"][0]["name"], packed["runs"][0]["series"])]
    on = _bot(tmp_path, pages_url="https://x.github.io/charts", max_url=3000)
    link = on.chart_link(chart, runs)
    assert link.startswith("https://x.github.io/charts/#d=") and len(link) <= 3000
    assert not _bot(tmp_path).pages_url


def test_card_offers_the_chart_button_only_with_a_link():
    pytest.importorskip("telegram")
    import time as clock

    from mlflow_telegram.bot import format as bot
    from mlflow_telegram.runs import RunView
    view = RunView(id="r" * 32, name="demo", experiment="e", status="FINISHED",
                   started=clock.time() - 60, ended=clock.time(), metrics={}, params={}, tags={})
    plain = bot.card_keyboard(view, clock.time()).to_dict()["inline_keyboard"]
    linked = bot.card_keyboard(view, clock.time(), "https://x.github.io/c/#d=AA").to_dict()
    assert not any("web_app" in b for row in plain for b in row)
    assert linked["inline_keyboard"][0][0]["web_app"]["url"].endswith("#d=AA")


def test_link_carries_siblings_off_by_default_within_budget(tmp_path, monkeypatch):
    pytest.importorskip("telegram")
    from mlflow_telegram import payload as fragment
    from mlflow_telegram.bot import app as bot
    monkeypatch.chdir(tmp_path)
    tb = _bot(tmp_path, pages_url="https://x.github.io/c", max_url=6000)
    small = {"training_mean": [(s, 1.0 / (s + 1)) for s in range(0, 2000, 100)]}
    chart = chartview.ChartState(runs=["a"], keys=["training_mean"])
    siblings = [(f"s{i}", "same-name", dict(small)) for i in range(5)]
    link = tb.chart_link(chart, [("main", small)], siblings)
    data = fragment.decode(link.split("#d=")[1])
    assert data["initial"]["runs"] == ["a"] and len(data["runs"]) == 6
    assert len({run["name"] for run in data["runs"]}) == 6          # shared names told apart
    import random
    noise = random.Random(0)
    huge = {"training_mean": [(s, noise.random()) for s in range(50_000)]}
    link = tb.chart_link(chart, [("main", huge)], siblings)
    data = fragment.decode(link.split("#d=")[1])
    assert len(link) <= 6000 - bot.LINK_MARGIN and len(data["runs"]) == 1   # main run first
