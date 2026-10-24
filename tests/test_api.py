"""The public surface: settings, the source protocol, the command registry, the
alert rule plug-in point and the command line. CPU only, no network."""

import time

import pytest

import mlflow_telegram
from mlflow_telegram import alerts
from mlflow_telegram.config import Config
from mlflow_telegram.runs import RunView
from mlflow_telegram.sources import RunSource


def test_config_prefers_new_names_and_falls_back_to_old_ones(tmp_path):
    new = Config.from_env({"MLFLOW_TELEGRAM_TOKEN": "1:new", "TELEGRAM_BOT_TOKEN": "1:old",
                           "MLFLOW_TELEGRAM_CHAT_ID": "42", "MLFLOW_TELEGRAM_MAX_URL": "9000",
                           "MLFLOW_TELEGRAM_PAGES_URL": "https://x.github.io/c/",
                           "MLFLOW_TELEGRAM_STATE": str(tmp_path / "s.json")})
    assert (new.token, new.chat_id, new.max_url) == ("1:new", 42, 9000)
    assert new.pages_url == "https://x.github.io/c" and new.state_path == str(tmp_path / "s.json")
    old = Config.from_env({"TELEGRAM_BOT_TOKEN": "1:old", "TELEGRAM_CHAT_ID": "-100",
                           "TRAINBOT_PAGES_MAX_URL": "5000", "TRAINBOT_DISK_GB": "5"})
    assert (old.token, old.chat_id, old.max_url, old.disk_gb) == ("1:old", -100, 5000, 5.0)
    assert old.telegram_ready and not Config.from_env({}).telegram_ready


def test_state_path_follows_xdg():
    path = Config.from_env({"XDG_STATE_HOME": "/somewhere"}).state_path
    assert path == "/somewhere/mlflow-telegram/state.json"


def test_lazy_top_level_exports():
    assert mlflow_telegram.Monitor.__name__ == "Monitor"
    assert callable(mlflow_telegram.encode) and callable(mlflow_telegram.build_page)
    with pytest.raises(AttributeError):
        mlflow_telegram.nothing_here  # noqa: B018


def test_mlflow_source_satisfies_the_protocol(tmp_path):
    pytest.importorskip("mlflow")
    from mlflow_telegram.sources.mlflow import MlflowSource
    assert isinstance(MlflowSource(f"sqlite:///{tmp_path / 'm.db'}"), RunSource)


def test_a_custom_rule_plugs_into_evaluate():
    class TooSlow(alerts.Rule):
        def check(self, view, now, gpu=None):
            if view.seconds_per_step and view.seconds_per_step > 1:
                return alerts.Alert(f"{view.id}:slow", view.id, f"{view.name} is slow")

    view = RunView(id="r", name="demo", experiment="e", status="RUNNING",
                   started=time.time() - 60, ended=None, metrics={}, params={}, tags={},
                   seconds_per_step=2.0, last_seen=time.time())
    found = alerts.evaluate(None, [view], rules=[TooSlow()], host_rules=[])
    assert [a.key for a in found] == ["r:slow"]


def test_commands_are_registered_from_the_list(tmp_path):
    pytest.importorskip("telegram")
    from telegram.ext import CommandHandler

    from mlflow_telegram.bot import DEFAULT_COMMANDS, Command, create_application

    async def hello(bot, update, context):
        pass

    class Source:      # nothing is read while building the application
        pass

    config = Config(token="123:abc", chat_id=1, state_path=str(tmp_path / "s.json"))
    commands = [c for c in DEFAULT_COMMANDS if c.name != "gpu"] + [Command("hello", "hi", hello)]
    app = create_application(config, Source(), commands=commands, watch_interval=0)
    names = {name for handler in app.handlers[0] if isinstance(handler, CommandHandler)
             for name in handler.commands}
    assert "hello" in names and "gpu" not in names and "runs" in names


def test_create_application_needs_telegram_settings(tmp_path):
    pytest.importorskip("telegram")
    from mlflow_telegram.bot import create_application
    with pytest.raises(ValueError):
        create_application(Config(state_path=str(tmp_path / "s.json")), object())


def test_cli_config_hides_the_token(capsys, monkeypatch, tmp_path):
    from mlflow_telegram import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TELEGRAM_TOKEN", "123:secret")
    assert cli.main(["config"]) == 0
    out = capsys.readouterr().out
    assert "secret" not in out and "<set>" in out


def test_cli_build_page(monkeypatch, tmp_path, capsys):
    from mlflow_telegram import cli, page
    monkeypatch.setattr(page, "echarts_source", lambda cache_dir=None: "window.echarts={};")
    assert cli.main(["build-page", str(tmp_path / "site")]) == 0
    assert (tmp_path / "site" / "index.html").exists()
