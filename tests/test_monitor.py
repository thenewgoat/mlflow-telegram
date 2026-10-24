"""The monitor: off by default, throttled on Telegram, and never fatal."""

import sys
import types
import urllib.parse
import urllib.request

import pytest

from mlflow_telegram.monitor import Monitor, load_env_file

TELEGRAM = {"TELEGRAM_BOT_TOKEN": "123:abc", "TELEGRAM_CHAT_ID": "42"}


@pytest.fixture
def sent(monkeypatch):
    """The texts the monitor posts to Telegram, without a network."""
    texts = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"{}"

    def urlopen(url, data=None, timeout=None):
        assert url == "https://api.telegram.org/bot123:abc/sendMessage"
        fields = dict(pair.split("=", 1) for pair in data.decode().split("&"))
        assert fields["chat_id"] == "42"
        texts.append(urllib.parse.unquote_plus(fields["text"]))
        return Reply()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return texts


@pytest.fixture
def fake_mlflow(monkeypatch):
    calls = []
    module = types.ModuleType("mlflow")
    for name in ("set_tracking_uri", "set_experiment", "start_run", "log_params",
                 "log_dict", "log_metrics", "set_tag", "end_run"):
        setattr(module, name, lambda *a, _n=name, **k: calls.append((_n, a, k)))
    monkeypatch.setitem(sys.modules, "mlflow", module)
    return calls


def test_off_without_the_environment(sent, fake_mlflow):
    with Monitor("script", "run", {"lr": 1e-3}, environ={}) as monitor:
        assert not monitor.enabled
        monitor.progress(1, loss=2.0)
        monitor.evaluation(1, validation=1.0)
    assert sent == [] and fake_mlflow == []


def test_telegram_start_throttled_progress_and_finish(sent):
    with Monitor("script", "run", environ=TELEGRAM, every_minutes=30) as monitor:
        for step in range(1, 6):
            monitor.progress(step, loss=1.0 / step)
            monitor.evaluation(step, best=True, validation=0.5)
        monitor.finish(best_validation=0.5)
    assert sent[0].startswith("▶️ script run started")
    # One progress and one evaluation in the first interval, not five of each.
    assert sum("⏳" in text for text in sent) == 1
    assert sum("📊" in text for text in sent) == 1
    assert "step 1" in sent[1] and "loss 1" in sent[1]
    assert "new best" in sent[2]
    assert sent[-1].startswith("✅ script run finished") and "best_validation 0.5" in sent[-1]
    assert len(sent) == 4


def test_crash_is_reported_and_raised(sent, fake_mlflow):
    environ = {**TELEGRAM, "MLFLOW_TRACKING_URI": "file:/nowhere"}
    with pytest.raises(RuntimeError, match="out of memory"):
        with Monitor("script", "run", environ=environ) as monitor:
            monitor.progress(7, loss=1.0)
            raise RuntimeError("out of memory")
    assert "💥 script run crashed at step 7" in sent[-1]
    assert "RuntimeError: out of memory" in sent[-1]
    assert ("end_run", (), {"status": "FAILED"}) in fake_mlflow


def test_interrupt_is_not_a_crash(sent):
    with pytest.raises(KeyboardInterrupt):
        with Monitor("script", "run", environ=TELEGRAM):
            raise KeyboardInterrupt
    assert sent[-1].startswith("⏹ script run interrupted")


def test_mlflow_gets_params_metrics_and_status(fake_mlflow):
    environ = {"MLFLOW_TRACKING_URI": "file:/tmp/x", "MLFLOW_EXPERIMENT_NAME": "exp"}
    with Monitor("script", "run", {"lr": 1e-3, "betas": (0.9, 0.98), "unset": None},
                 environ=environ) as monitor:
        monitor.progress(10, loss=1.5, bad=float("nan"), label="text")
        monitor.evaluation(10, validation=1.2)
        monitor.finish(best=1.2)
    names = [name for name, _, _ in fake_mlflow]
    assert names[:3] == ["set_tracking_uri", "set_experiment", "start_run"]
    assert ("set_experiment", ("exp",), {}) in fake_mlflow
    params = next(a[0] for n, a, _ in fake_mlflow if n == "log_params")
    assert params == {"lr": "0.001", "betas": "[0.9, 0.98]"}
    metrics = [(a[0], k.get("step")) for n, a, k in fake_mlflow if n == "log_metrics"]
    assert metrics == [({"loss": 1.5}, 10), ({"eval/validation": 1.2}, 10),
                       ({"final/best": 1.2}, None)]
    assert fake_mlflow[-1] == ("end_run", (), {"status": "FINISHED"})


def test_failures_never_reach_the_loop(monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise OSError("network is down")

    monkeypatch.setattr(urllib.request, "urlopen", broken)
    with Monitor("script", "run", environ=TELEGRAM) as monitor:
        monitor.progress(1, loss=1.0)
    assert "telegram send failed: OSError: network is down" in capsys.readouterr().err


def test_env_file_fills_in_but_never_overrides(monkeypatch, tmp_path):
    pytest.importorskip("dotenv")
    settings = tmp_path / "monitor.env"
    settings.write_text("MONITOR_TEST_A=from-file\nMONITOR_TEST_B=from-file\n")
    monkeypatch.setenv("MONITOR_ENV_FILE", str(settings))
    monkeypatch.setenv("MONITOR_TEST_B", "from-shell")
    monkeypatch.delenv("MONITOR_TEST_A", raising=False)
    load_env_file()
    import os
    assert os.environ["MONITOR_TEST_A"] == "from-file"
    assert os.environ["MONITOR_TEST_B"] == "from-shell"
    monkeypatch.delenv("MONITOR_TEST_A")


def test_missing_env_file_is_fine(monkeypatch, tmp_path):
    monkeypatch.setenv("MONITOR_ENV_FILE", str(tmp_path / "absent.env"))
    load_env_file()
