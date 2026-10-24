"""What the bot writes: message texts, keyboards and Telegram's length limits.

Pure functions over `RunView`s (no network), so they can be reused or replaced.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import ExtBot

from mlflow_telegram.notify import CAPTION_LIMIT, TEXT_LIMIT, fit
from mlflow_telegram.redact import redact_text
from mlflow_telegram.runs import RunView, bar, duration, number

__all__ = ["esc", "fit", "FittingBot", "parse_duration", "health", "progress_line", "runs_line",
           "card", "system_summary", "card_keyboard", "nav_keyboard", "shown_params",
           "params_pages", "params_keyboard", "params_file", "differing", "status_text",
           "nvidia_smi", "disk_text", "STATUS_EMOJI", "KEY_PARAMS", "TEXT_LIMIT", "CAPTION_LIMIT"]

STATUS_EMOJI = {"RUNNING": "🟢", "FINISHED": "✅", "FAILED": "💥", "KILLED": "⏹",
                "SCHEDULED": "🕓"}
PAGE_LIMIT = 3500
# Shown first on the Params page, in this order; the rest follow alphabetically.
KEY_PARAMS = ("learning_rate", "lr", "batch_size", "batch", "epochs", "max_steps", "steps",
              "warmup", "schedule", "optimizer", "weight_decay", "dropout", "patience",
              "eval_every", "evaluate_every", "model", "d_model", "layers", "heads", "seed")
EMPTY = ("", "none", "null", "[]", "{}")
HIDDEN_PARAMS = {"command", "out", "output_dir", "cache", "dataset", "data", "run"}


def esc(value) -> str:
    return html.escape(str(value), quote=False)


class FittingBot(ExtBot):
    """Every message and caption cut to Telegram's limits before it is sent, so an
    overlong reply still arrives, with its keyboard, instead of failing."""

    async def send_message(self, chat_id, text, *args, **kwargs):
        return await super().send_message(chat_id, fit(text, TEXT_LIMIT), *args, **kwargs)

    async def edit_message_text(self, text, *args, **kwargs):
        return await super().edit_message_text(fit(text, TEXT_LIMIT), *args, **kwargs)

    async def send_photo(self, chat_id, photo, *args, caption=None, **kwargs):
        return await super().send_photo(chat_id, photo, *args,
                                        caption=fit(caption, CAPTION_LIMIT), **kwargs)

    async def send_document(self, chat_id, document, *args, caption=None, **kwargs):
        return await super().send_document(chat_id, document, *args,
                                           caption=fit(caption, CAPTION_LIMIT), **kwargs)


def parse_duration(text: str) -> float | None:
    """'30m', '2h', '1h30m', '45' (minutes) → seconds."""
    text = text.strip().lower()
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text) * 60
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*([dhms])", text)
    if not parts or "".join(n + u for n, u in parts) != text.replace(" ", ""):
        return None
    scale = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    return sum(float(n) * scale[u] for n, u in parts)


# -- text ------------------------------------------------------------------

def health(view: RunView, now: float) -> str:
    if not view.live:
        return f"{STATUS_EMOJI.get(view.status, '•')} {view.status}"
    if view.dead:
        return "💀 RUNNING in the tracker, but its process is gone"
    if view.stale(now):
        return f"🥶 RUNNING but quiet for {duration(view.silent_for(now))}"
    alive = " · process alive" if view.pid_alive else ""
    return f"🟢 RUNNING{alive}"


def progress_line(view: RunView) -> str:
    if view.step is None:
        return "no steps yet"
    if view.max_steps:
        return (f"step {view.step}/{view.max_steps} {bar(view.fraction)} "
                f"{view.fraction:.0%}")
    return f"step {view.step}"


def runs_line(view: RunView, now: float) -> str:
    flag = "💀 " if view.dead else ("🥶 " if view.stale(now) else "")
    eta = f" · ETA {duration(view.eta)}" if view.eta is not None else ""
    seen = view.silent_for(now)
    seen = f"seen {duration(seen)} ago" if seen is not None else "no metrics yet"
    return (f"{flag}<b>{esc(view.name)}</b> <i>{esc(view.experiment)}</i>\n"
            f"   {progress_line(view)}{eta} · {seen}")


def card(view: RunView, now: float) -> str:
    lines = [f"<b>{esc(view.name)}</b> · {esc(view.experiment)}", health(view, now),
             progress_line(view)]
    end = view.ended or now
    timing = f"elapsed {duration(end - view.started)}"
    if view.live and view.eta is not None:
        timing += f" · ETA {duration(view.eta)}"
    if view.seconds_per_step:
        timing += f" · {view.seconds_per_step:.3g} s/step"
    lines.append(timing)
    if view.last_seen:
        lines.append(f"last metric {duration(now - view.last_seen)} ago")
    train = [f"{key} {number(view.metrics[key])}"
             for key in (view.train_key, "learning_rate", "lr")
             if key and key in view.metrics]
    if train:
        lines.append("train: " + " · ".join(train))
    if view.best:
        key, value, step = view.best
        lines.append(f"best {esc(key)} <b>{number(value)}</b> @ {step}")
    if view.patience and "patience_used" in view.metrics:
        lines.append(f"patience {int(view.metrics['patience_used'])}/{view.patience}")
    finals = [f"{key[6:]} {number(value)}" for key, value in sorted(view.metrics.items())
              if key.startswith("final/")]
    if finals:
        lines.append("final: " + esc(" · ".join(finals)))
    system = system_summary(view.metrics)
    if system:
        lines.append("system: " + system)
    if view.tags.get("best_checkpoint"):
        lines.append(f"checkpoint <code>{esc(redact_text(view.tags['best_checkpoint']))}</code>")
    if view.tags.get("error"):
        lines.append(f"error: <code>{esc(redact_text(view.tags['error'])[:300])}</code>")
    note = view.tags.get("mlflow.note.content")
    if note:
        lines.append(f"📝 {esc(note[:500])}")
    return "\n".join(lines)


def system_summary(metrics: dict) -> str:
    parts = []
    pick = {"system/gpu_0_utilization_percentage": "GPU {:.0f}%",
            "system/gpu_0_memory_usage_megabytes": "GPU mem {:.1f} GB",
            "system/cpu_utilization_percentage": "CPU {:.0f}%",
            "system/system_memory_usage_percentage": "RAM {:.0f}%",
            "system/disk_available_megabytes": "disk {:.0f} GB free"}
    for key, form in pick.items():
        if key in metrics:
            value = metrics[key]
            if key.endswith("megabytes"):
                value /= 1024
            parts.append(form.format(value))
    return " · ".join(parts)


def card_keyboard(view: RunView, now: float, link: str | None = None) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton("📈 Curves", callback_data=f"curves:{view.id}"),
             InlineKeyboardButton("⚙️ Params", callback_data=f"params:{view.id}"),
             InlineKeyboardButton("🆚 Compare", callback_data=f"cmpmenu:{view.id}")],
            [InlineKeyboardButton("🖥 System", callback_data=f"sys:{view.id}"),
             InlineKeyboardButton("📝 Note", callback_data=f"note:{view.id}"),
             InlineKeyboardButton("🔄 Refresh", callback_data=f"card:{view.id}")]]
    if view.live and (view.dead or view.stale(now)):
        rows.append([InlineKeyboardButton("🪦 Mark killed", callback_data=f"kill:{view.id}")])
    if link:   # the static Mini App with the run's usual curves, one tap from the card
        rows[0].insert(0, InlineKeyboardButton("📊 Chart", web_app=WebAppInfo(url=link)))
    return InlineKeyboardMarkup(rows)


def nav_keyboard(run_id: str, *extra: list) -> InlineKeyboardMarkup:
    """The row every run view carries, so the card is always one tap away."""
    return InlineKeyboardMarkup([*extra, [
        InlineKeyboardButton("⬅️ Card", callback_data=f"card:{run_id}"),
        InlineKeyboardButton("📈 Curves", callback_data=f"curves:{run_id}"),
        InlineKeyboardButton("⚙️ Params", callback_data=f"params:{run_id}:0"),
        InlineKeyboardButton("🆚 Compare", callback_data=f"cmpmenu:{run_id}")]])


def shown_params(view: RunView, key_params=KEY_PARAMS) -> list[tuple[str, str]]:
    """Params worth reading: the key ones first, then the rest; no empty values and
    no `command` (its flags are already among them)."""
    params = {k: v for k, v in view.params.items()
              if k != "command" and str(v).strip().lower() not in EMPTY}
    first = [k for k in key_params if k in params]
    return [(k, params[k]) for k in first + sorted(set(params) - set(first))]


def params_pages(view: RunView, limit: int = PAGE_LIMIT, key_params=KEY_PARAMS) -> list[str]:
    """The params as pages of `key  value` lines in a monospace block, each page
    under `limit` characters."""
    shown = shown_params(view, key_params)
    width = min(24, max((len(k) for k, _ in shown), default=0))
    lines = [f"{k[:width]:<{width}}  {v if len(v) <= 48 else v[:47] + '…'}" for k, v in shown]
    pages, page = [], []
    for line in lines:
        if page and len(esc("\n".join(page + [line]))) > limit - 200:
            pages.append(page)
            page = []
        page.append(line)
    pages.append(page)
    title = f"<b>{esc(view.name)}</b> · params ({len(shown)})"
    return [f"{title}{f' · page {i + 1}/{len(pages)}' if len(pages) > 1 else ''}\n"
            f"<pre>{esc(chr(10).join(lines) or 'no params')}</pre>"
            for i, lines in enumerate(pages)]


def params_keyboard(run_id: str, page: int, pages: int) -> InlineKeyboardMarkup:
    row = []
    if page > 0:
        row.append(InlineKeyboardButton("◀️ Prev", callback_data=f"params:{run_id}:{page - 1}"))
    if page < pages - 1:
        row.append(InlineKeyboardButton("Next ▶️", callback_data=f"params:{run_id}:{page + 1}"))
    extra = [row] if row else []
    extra.append([InlineKeyboardButton("📄 Full list as file", callback_data=f"pfile:{run_id}")])
    return nav_keyboard(run_id, *extra)


def params_file(view: RunView) -> bytes:
    lines = [f"{view.experiment} {view.name} ({view.id})", ""]
    lines += [f"{k} = {v}" for k, v in sorted(view.params.items())]
    return "\n".join(lines).encode()


def differing(a: RunView, b: RunView) -> list[str]:
    keys = sorted((set(a.params) | set(b.params)) - HIDDEN_PARAMS)
    return [f"{esc(k)}: <code>{esc(a.params.get(k, '—')[:40])}</code> → "
            f"<code>{esc(b.params.get(k, '—')[:40])}</code>"
            for k in keys if a.params.get(k) != b.params.get(k)]


def status_text(views: list[RunView], gpu: str, disk: str, now: float, muted_until: float):
    lines = [f"🛰 <b>Training status</b> · {time.strftime('%H:%M', time.localtime(now))}"]
    if views:
        lines += [runs_line(view, now) for view in views]
    else:
        lines.append("no live runs")
    lines.append(" · ".join(part for part in (gpu, disk) if part))
    if muted_until > now:
        lines.append(f"🔕 muted for {duration(muted_until - now)}")
    return "\n".join(lines)


def nvidia_smi() -> str:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        util, used, total, temperature, power = [p.strip() for p in output.split(",")[:5]]
        return (f"GPU {util}% · {float(used) / 1024:.1f}/{float(total) / 1024:.1f} GB · "
                f"{temperature}°C · {power} W")
    except Exception:  # noqa: BLE001
        return "GPU ?"


def disk_text(path: str) -> str:
    try:
        return f"disk {shutil.disk_usage(path).free / 2**30:.0f} GB free"
    except OSError:
        return ""
