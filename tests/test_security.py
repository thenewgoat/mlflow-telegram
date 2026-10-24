"""The security properties the docs promise: secrets masked, the token never
printed or logged, one chat (and optionally some users) answered, hostile chart
links refused, owner-only state files, no inline script in the page."""

import json
import logging
import os
import re
import shutil
import subprocess
import time
import types
import urllib.request

import pytest

from mlflow_telegram import payload
from mlflow_telegram.config import Config
from mlflow_telegram.redact import MASK, redact_command, redact_params, redact_text

TOKEN = "123456789:TESTONLY" + "x" * 30   # shaped like a token, obviously fake


def test_secret_params_and_flags_are_masked():
    params = redact_params({"lr": "3e-4", "wandb_api_key": "k1", "HF_TOKEN": "t",
                            "db_password": "p", "command": "python t.py --api-key k2 --lr 1 "
                            "--token=t3 --hf-token 'q r' --name demo"})
    assert params["lr"] == "3e-4" and params["wandb_api_key"] == MASK
    assert params["HF_TOKEN"] == MASK and params["db_password"] == MASK
    command = params["command"]
    assert "k2" not in command and "t3" not in command and "q r" not in command
    assert "--lr 1" in command and "--name demo" in command
    assert redact_command("--seed 0") == "--seed 0"


def test_parsed_command_flags_are_masked_on_the_run_view():
    from mlflow_telegram.runs import all_params
    from mlflow_telegram.sources import RunInfo
    run = RunInfo(id="r", name="n", experiment_id="0", experiment="e", status="FINISHED",
                  start_time=0, params={"command": "python t.py --hf-token SECRET --lr 1"})
    shown = all_params(run)
    assert "SECRET" not in json.dumps(shown) and shown["lr"] == "1"


def test_text_redaction():
    home = os.path.expanduser("~")
    text = (f"POST https://api.telegram.org/bot{TOKEN}/sendMessage failed; "
            f"https://user:pw@host/x?api_key=abc&x=1 in {home}/runs/a.py")
    clean = redact_text(text, [TOKEN])
    assert TOKEN not in clean and "pw@" not in clean and "abc" not in clean
    assert home not in clean and "~/runs/a.py" in clean and "x=1" in clean


def test_token_is_stripped_and_checked():
    config = Config.from_env({"MLFLOW_TELEGRAM_TOKEN": TOKEN + "\r",
                              "MLFLOW_TELEGRAM_CHAT_ID": "1"})
    assert config.token == TOKEN and config.token_looks_valid
    assert not Config.from_env({"MLFLOW_TELEGRAM_TOKEN": "12:a b"}).token_looks_valid


def test_sender_refuses_a_malformed_token_and_masks_errors(monkeypatch, capsys):
    from mlflow_telegram.notify import Telegram
    with pytest.raises(ValueError):
        Telegram("123:a b", "1")

    def urlopen(url, data=None, timeout=None):
        raise OSError(f"cannot reach {url}")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    sender = Telegram(TOKEN, "1")
    sender.send("hi")
    sender.close()
    err = capsys.readouterr().err
    assert "telegram send failed" in err and TOKEN not in err


def test_monitor_masks_secrets_and_the_crash_text(monkeypatch):
    calls, sent = [], []
    fake = types.ModuleType("mlflow")
    for name in ("set_tracking_uri", "set_experiment", "start_run", "log_params",
                 "log_metrics", "set_tag", "end_run", "log_artifact"):
        setattr(fake, name, lambda *a, _n=name, **k: calls.append((_n, a, k)))
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
    from mlflow_telegram.monitor import Monitor
    from mlflow_telegram.notify import Telegram
    monkeypatch.setattr(Telegram, "_post_text", lambda self, text: sent.append(text))
    environ = {"MLFLOW_TRACKING_URI": "file:/x", "MLFLOW_TELEGRAM_TOKEN": TOKEN,
               "MLFLOW_TELEGRAM_CHAT_ID": "1"}
    with pytest.raises(RuntimeError):
        with Monitor("t.py", "demo", config={"api_key": "K", "lr": 1,
                                             "command": "x --token T2"},
                     environ=environ, system_metrics=False):
            raise RuntimeError(f"bad url https://u:secretpw@h/ in {os.path.expanduser('~')}/a")
    logged = next(a[0] for n, a, _ in calls if n == "log_params")
    assert logged["api_key"] == MASK and "T2" not in logged["command"]
    crash = sent[-1]
    assert "secretpw" not in crash and os.path.expanduser("~") + "/" not in crash
    error_tag = next(a[1] for n, a, _ in calls if n == "set_tag" and a[0] == "error")
    assert "secretpw" not in error_tag


def test_traceback_can_be_left_out(monkeypatch):
    from mlflow_telegram.monitor import Monitor
    from mlflow_telegram.notify import Telegram
    sent = []
    monkeypatch.setattr(Telegram, "_post_text", lambda self, text: sent.append(text))
    environ = {"MLFLOW_TELEGRAM_TOKEN": TOKEN, "MLFLOW_TELEGRAM_CHAT_ID": "1"}
    with pytest.raises(ValueError):
        with Monitor("t.py", "demo", environ=environ, send_traceback=False):
            raise ValueError("boom")
    assert "Traceback" not in sent[-1] and "boom" in sent[-1]


def test_state_file_is_owner_only(tmp_path):
    from mlflow_telegram import state
    path = tmp_path / "deep" / "state.json"
    state.write(str(path), {"a": 1})
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert oct(path.parent.stat().st_mode & 0o777) == "0o700"
    assert not [p for p in path.parent.iterdir() if p.name.endswith(".tmp")]


def test_names_with_quotes_and_wildcards_are_found_literally(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from mlflow_telegram.sources.mlflow import MlflowSource
    monkeypatch.chdir(tmp_path)
    source = MlflowSource(f"sqlite:///{tmp_path / 'm.db'}")
    experiment = source.client.create_experiment("e")
    for name in ("it's 100%", "it_s", "plain"):
        source.client.create_run(experiment, run_name=name)
    assert [r.name for r in source.find_runs("it's 100%")] == ["it's 100%"]
    assert [r.name for r in source.find_runs("_s")] == ["it_s"]
    assert source.find_runs("%") == source.find_runs("it's 100%")


# -- the bot --------------------------------------------------------------------

telegram = pytest.importorskip("telegram")


def _bot(tmp_path, **settings):
    from mlflow_telegram.bot.app import TrainBot
    config = Config(token=TOKEN, chat_id=settings.pop("chat_id", 42),
                    state_path=str(tmp_path / "s.json"), **settings)
    return TrainBot(config, source=object())


def _update(chat=42, user=7, **extra):
    return types.SimpleNamespace(effective_chat=types.SimpleNamespace(id=chat),
                                 effective_user=types.SimpleNamespace(id=user), **extra)


def test_only_the_chat_and_allowed_users_are_answered(tmp_path):
    bot = _bot(tmp_path)
    assert bot.allowed(_update()) and not bot.allowed(_update(chat=1))
    assert not bot.allowed(types.SimpleNamespace(effective_chat=None, effective_user=None))
    group = _bot(tmp_path, chat_id=-100, allowed_users=(7,))
    assert group.allowed(_update(chat=-100, user=7)) and not group.allowed(_update(chat=-100,
                                                                                    user=8))


def test_handlers_are_filtered_and_updates_limited(tmp_path):
    from telegram.ext import ChatMemberHandler, CommandHandler, MessageHandler

    from mlflow_telegram.bot.app import ALLOWED_UPDATES
    app = _bot(tmp_path, allowed_users=(7,)).application(hello=False, watch_interval=0)
    handlers = app.handlers[0]
    for handler in handlers:
        if isinstance(handler, (CommandHandler, MessageHandler)):
            text = repr(handler.filters)
            assert "42" in text and "7" in text, text
    assert any(isinstance(h, ChatMemberHandler) for h in handlers)
    assert set(ALLOWED_UPDATES) == {"message", "callback_query", "my_chat_member"}


def test_buttons_from_elsewhere_do_nothing(tmp_path):
    import asyncio
    bot = _bot(tmp_path)
    answered = []

    async def answer():
        answered.append(True)

    query = types.SimpleNamespace(answer=answer, data="killok:abc")
    asyncio.run(bot.button(_update(chat=999, callback_query=query), None))
    assert answered == [True]          # answered (to stop the spinner) and nothing else


def test_note_only_from_a_reply_to_the_prompt(tmp_path):
    import asyncio
    notes = []
    bot = _bot(tmp_path)
    bot.source = types.SimpleNamespace(set_note=lambda run, text: notes.append((run, text)))

    async def reply_text(text):
        pass

    def message(text, reply_to=None):
        return types.SimpleNamespace(text=text, reply_text=reply_text,
                                     reply_to_message=reply_to and types.SimpleNamespace(
                                         message_id=reply_to))

    context = types.SimpleNamespace(user_data={"pending_note": {"run": "r1", "prompt": 5}})
    asyncio.run(bot.text(types.SimpleNamespace(effective_message=message("x", 4)), context))
    asyncio.run(bot.text(types.SimpleNamespace(effective_message=message("y")), context))
    assert notes == []
    asyncio.run(bot.text(types.SimpleNamespace(effective_message=message("ok", 5)), context))
    assert notes == [("r1", "ok")] and "pending_note" not in context.user_data


def test_errors_reach_the_chat_without_their_text(tmp_path):
    import asyncio
    bot = _bot(tmp_path)
    replies = []

    async def reply_text(text):
        replies.append(text)

    fake = _update(effective_message=types.SimpleNamespace(reply_text=reply_text))
    error = RuntimeError(f"cannot open /home/someone/secret.db with {TOKEN}")
    context = types.SimpleNamespace(error=error)
    logging.disable(logging.CRITICAL)
    try:
        asyncio.run(bot.error(fake, context))
    finally:
        logging.disable(logging.NOTSET)
    assert replies and "secret.db" not in replies[0] and TOKEN not in replies[0]


def test_token_filter_masks_log_records():
    from mlflow_telegram.bot.app import RedactToken
    record = logging.LogRecord("httpx", logging.INFO, __file__, 1,
                               "HTTP Request: POST https://api.telegram.org/bot%s/getMe",
                               (TOKEN,), None)
    RedactToken(TOKEN).filter(record)
    assert TOKEN not in record.getMessage()


def test_alert_callback_data_fits_64_bytes():
    from mlflow_telegram.bot.app import short_key
    assert len(f"ign:{short_key('r' * 32 + ':patience@' + '9' * 500)}".encode()) <= 64


# -- the page -----------------------------------------------------------------------

def test_page_has_csp_and_no_inline_script():
    from mlflow_telegram import page
    html = page.page_html()
    assert "Content-Security-Policy" in html and "connect-src 'none'" in html
    assert "'unsafe-inline'" not in page.CSP.split("style-src")[0]       # not for scripts
    assert 'name="referrer" content="no-referrer"' in html
    scripts = re.findall(r"<script([^>]*)>(.*?)</script>", html, re.S)
    assert scripts and all('src="' in attrs and not body.strip() for attrs, body in scripts)
    assert "on" + "click=" not in html and "javascript:" not in html


NODE = shutil.which("node")


def _decode_in_node(fragment: str) -> dict:
    """Runs the page's decoder (app.js's first half) in node on `fragment`."""
    from importlib import resources
    decoder = resources.files("mlflow_telegram.page").joinpath("assets", "fragment.js").read_text()
    script = ("globalThis.window = {}; globalThis.location = {hash: "
              "require('fs').readFileSync(0, 'utf8')};\n"
              + decoder + "\nDECODED.then(d => console.log(JSON.stringify({ok: true, runs: "
              "Object.keys(d.runs), n: d.catalogue.length})), e => console.log(JSON.stringify("
              "{ok: false, error: e.message})));")
    out = subprocess.run([NODE, "-e", script], input=fragment, capture_output=True, text=True,
                         timeout=60)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _pack(header: dict, body: bytes = b"") -> str:
    import base64
    import struct
    import zlib
    head = json.dumps(header).encode()
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    raw = compressor.compress(struct.pack(">I", len(head)) + head + body) + compressor.flush()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


@pytest.mark.skipif(NODE is None, reason="needs node")
def test_page_decoder_accepts_real_links():
    text = payload.encode({"initial": {"runs": ["a"], "keys": ["loss"]},
                           "runs": [{"id": "a", "name": "demo", "series": {
                               "loss": [(s, 1.0 / (s + 1)) for s in range(100)]}}]})
    assert _decode_in_node("#d=" + text + "&tgWebAppData=x") == {"ok": True, "runs": ["a"], "n": 1}


@pytest.mark.skipif(NODE is None, reason="needs node")
@pytest.mark.parametrize("fragment, why", [
    ("#d=" + "A" * 300000, "encoding"),
    ("#d=<script>", "encoding"),
    ("#d=" + _pack({"v": 2, "initial": {"runs": [], "keys": []}, "runs": []}), "version"),
    ("#d=" + _pack({"v": 1, "initial": {"runs": [1], "keys": []}, "runs": []}), "runs"),
    ("#d=" + _pack({"v": 1, "initial": {"runs": ["a"], "keys": ["k"]}, "runs": [
        {"id": "a", "name": "n", "series": [{"key": "k", "n": 10 ** 9, "lo": 0, "hi": 1,
                                             "log": False}]}]}), "series"),
    ("#d=" + _pack({"v": 1, "initial": {"runs": ["a"], "keys": ["k"]}, "runs": [
        {"id": "a", "name": "n", "series": [{"key": "k", "n": 50, "lo": 0, "hi": 1,
                                             "log": False}]}]}, b"\x02"), "truncated"),
    ("#d=" + _pack({"v": 1, "initial": {"runs": ["__proto__"], "keys": ["constructor"]},
                    "runs": [{"id": "__proto__", "name": "<img src=x onerror=alert(1)>",
                              "series": [{"key": "constructor", "n": 0, "lo": 0, "hi": 1,
                                          "log": False}]}]}), None),
], ids=["oversized", "not-base64", "version", "bad-run-id", "huge-n", "truncated",
        "odd-names"])
def test_page_decoder_refuses_hostile_links(fragment, why):
    result = _decode_in_node(fragment)
    if why is None:      # odd names are only data: decoded, never interpreted
        assert result["ok"] and result["runs"] == ["__proto__"]
    else:
        assert not result["ok"] and why in result["error"], result


@pytest.mark.skipif(NODE is None, reason="needs node")
def test_page_decoder_stops_a_deflate_bomb():
    bomb = _pack({"v": 1, "initial": {"runs": [], "keys": []}, "runs": []}, b"\0" * (64 << 20))
    assert len(bomb) < 200000
    result = _decode_in_node("#d=" + bomb)
    assert not result["ok"] and "too large" in result["error"]
    assert time.time()      # (finishes: the cap stops inflating at 4 MB)
