"""The interactive chart: a static Telegram Mini App page, with no data in it.

`build_page(out_dir)` writes the site (``index.html``, the pinned ECharts build,
``.nojekyll``, ``robots.txt``) for any static HTTPS host; GitHub Pages works.
The bot's 📊 Chart button opens it with the chart's curves packed after
``#d=`` (see `mlflow_telegram.payload`); browsers never send the fragment to
the host, so the host only ever serves the empty page.

`probe_link_limit` measures the longest button link Telegram accepts (it is not
documented): it posts one message to your chat, edits its button with longer
and longer links, and deletes it.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from importlib import resources

__all__ = ["build_page", "page_html", "echarts_source", "probe_link_limit",
           "ECHARTS_VERSION", "ECHARTS_FILE"]

ECHARTS_VERSION = "5.6.0"
ECHARTS_URL = f"https://cdnjs.cloudflare.com/ajax/libs/echarts/{ECHARTS_VERSION}/echarts.min.js"
ECHARTS_SHA256 = "bf4a223524e40b77c304bec67e1222cf551f14880cf42c69dc046558e11c07b1"
ECHARTS_FILE = f"echarts-{ECHARTS_VERSION}.min.js"   # versioned name: safe to cache for good


def _asset(name: str) -> str:
    return resources.files("mlflow_telegram.page").joinpath("assets", name).read_text("utf-8")


def page_html() -> str:
    """The page: chart code, the fragment decoder, Telegram's script, and ECharts
    loaded from beside it."""
    libs = ("<script>window.__FRAGMENT = location.hash;</script>\n"
            '<script src="https://telegram.org/js/telegram-web-app.js"></script>\n'
            f'<script src="{ECHARTS_FILE}"></script>')
    source = f"<script>\n{_asset('fragment.js')}</script>"
    return _asset("page.html").replace("<!--LIBS-->", libs).replace("<!--SOURCE-->", source)


def _cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "mlflow-telegram")


def echarts_source(cache_dir: str | None = None) -> str | None:
    """The pinned ECharts build, from a local cache filled once from cdnjs and
    checked against its SHA-256; None if it cannot be had or does not match."""
    path = os.path.join(cache_dir or _cache_dir(), ECHARTS_FILE)
    try:
        with open(path, "rb") as handle:
            body = handle.read()
    except OSError:
        try:
            with urllib.request.urlopen(ECHARTS_URL, timeout=30) as reply:
                body = reply.read()
        except Exception:  # noqa: BLE001
            return None
        if hashlib.sha256(body).hexdigest() != ECHARTS_SHA256:
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(body)
    if hashlib.sha256(body).hexdigest() != ECHARTS_SHA256:
        return None
    return body.decode()


def build_page(out_dir: str, echarts: str | None = None) -> list[str]:
    """Writes the site into `out_dir`; returns the file names written."""
    echarts = echarts if echarts is not None else echarts_source()
    if echarts is None:
        raise RuntimeError(f"could not get ECharts {ECHARTS_VERSION} from {ECHARTS_URL}")
    os.makedirs(out_dir, exist_ok=True)
    files = {"index.html": page_html(), ECHARTS_FILE: echarts, ".nojekyll": "",
             "robots.txt": "User-agent: *\nDisallow: /\n"}
    for name, text in files.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as handle:
            handle.write(text)
    return sorted(files)


def probe_link_limit(token: str, chat_id: str | int, page_url: str, top: int = 65536) -> int:
    """The longest web_app button link Telegram accepts, by binary search."""
    base = page_url.rstrip("/") + "/"

    def call(method, **fields):
        data = urllib.parse.urlencode({k: json.dumps(v) if isinstance(v, dict) else v
                                       for k, v in fields.items()}).encode()
        try:
            with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/{method}",
                                        data=data, timeout=20) as reply:
                return json.loads(reply.read())
        except urllib.error.HTTPError as error:
            return json.loads(error.read())

    def markup(length):
        url = base + "#d=" + "A" * max(0, length - len(base) - 3)
        return {"inline_keyboard": [[{"text": f"probe {length}", "web_app": {"url": url}}]]}

    sent = call("sendMessage", chat_id=chat_id, text="Measuring the longest chart link "
                "(this message deletes itself).", reply_markup=markup(100))
    if not sent.get("ok"):
        raise RuntimeError(f"Telegram refused even a short link: {sent.get('description')}")
    message = sent["result"]["message_id"]
    low, high = 100, top
    while low < high:
        middle = (low + high + 1) // 2
        reply = call("editMessageReplyMarkup", chat_id=chat_id, message_id=message,
                     reply_markup=markup(middle))
        if reply.get("ok") or "not modified" in reply.get("description", ""):
            low = middle
        else:
            high = middle - 1
    call("deleteMessage", chat_id=chat_id, message_id=message)
    return low
