"""The bot: ask Telegram about the runs in your tracker.

    from mlflow_telegram.bot import run_bot
    run_bot()                       # settings from the environment / .env

or, to change what it does::

    from mlflow_telegram import Config, MlflowSource
    from mlflow_telegram.alerts import DEFAULT_RULES, DiskLow
    from mlflow_telegram.bot import DEFAULT_COMMANDS, Command, create_application

    async def hello(bot, update, context):
        await update.effective_message.reply_text("hi")

    app = create_application(Config.from_env(), commands=DEFAULT_COMMANDS + [
        Command("hello", "say hi", hello)], rules=DEFAULT_RULES)
    app.run_polling()

Long polling with python-telegram-bot 21. It answers only the configured chat
and ignores everyone else; its only writes to the tracker are a run's status
("Mark killed", which never touches a process) and its note.

Every minute a watcher checks the live runs against the alert rules, sends each
new alert once (not while muted; more than three at once as one digest), and
edits one pinned status message in place rather than posting routine status.
With a page URL configured, every chart and run card carries a 📊 Chart button
that opens the interactive Mini App with the curves packed into its link.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from telegram import (
    BotCommand,
    ForceReply,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Update,
    WebAppInfo,
)
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from mlflow_telegram import alerts, charts, chartview, payload
from mlflow_telegram import runs as run_views
from mlflow_telegram import state as bot_state
from mlflow_telegram.bot.format import (
    CAPTION_LIMIT,
    KEY_PARAMS,
    STATUS_EMOJI,
    FittingBot,
    card,
    card_keyboard,
    differing,
    disk_text,
    esc,
    fit,
    health,
    nav_keyboard,
    nvidia_smi,
    params_file,
    params_keyboard,
    params_pages,
    parse_duration,
    runs_line,
    status_text,
    system_summary,
)
from mlflow_telegram.config import Config
from mlflow_telegram.redact import redact_text
from mlflow_telegram.runs import RunView, duration, higher_is_better, number
from mlflow_telegram.sources import RunSource

__all__ = ["TrainBot", "Command", "DEFAULT_COMMANDS", "create_application", "run_bot"]

log = logging.getLogger("mlflow_telegram.bot")

LINK_MARGIN = 200     # characters kept free under the link limit
LINK_SIBLINGS = 5     # recent runs of the experiment offered as overlays in the Mini App
SIBLING_POINTS = 200  # points per curve for those
ALBUM_DPI = 300       # ~2,100 px wide: Telegram's viewer keeps up to 2,560
DIGEST_OVER = 3       # more new alerts than this in one minute go out as one message

#: The update types the bot asks Telegram for; nothing else reaches it.
ALLOWED_UPDATES = [Update.MESSAGE, Update.CALLBACK_QUERY, Update.MY_CHAT_MEMBER]


def short_key(key: str) -> str:
    """A fixed-length stand-in for an alert key inside callback data (64 bytes max)."""
    return hashlib.sha1(key.encode()).hexdigest()[:16]


class RedactToken(logging.Filter):
    """Masks the bot token in log records (httpx logs request URLs, which carry it)."""

    def __init__(self, token: str):
        super().__init__()
        self.token = token

    def filter(self, record: logging.LogRecord) -> bool:
        if self.token:
            message = record.getMessage()
            if self.token in message:
                record.msg, record.args = message.replace(self.token, "***"), None
        return True


Handler = Callable[["TrainBot", Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]


@dataclass(frozen=True)
class Command:
    """A slash command: `/name`, its line in /help and Telegram's menu, and the
    coroutine `handler(bot, update, context)` that answers it."""

    name: str
    help: str
    handler: Handler


class TrainBot:
    """The bot's behaviour, independent of how it is wired into an Application.

    `source` is where runs come from (any `RunSource`); `rules` and `host_rules`
    are the watcher's alert rules; `commands` are the slash commands it answers.
    """

    key_params = KEY_PARAMS

    def __init__(self, config: Config, source: RunSource, *, commands=None,
                 rules=alerts.DEFAULT_RULES, host_rules=None):
        if config.chat_id is None:
            raise ValueError("Config.chat_id is required")
        self.config = config
        self.source = source
        self.chat_id = config.chat_id
        self.state_path = config.state_path
        self.disk_path = config.disk_path
        self.pages_url = config.pages_url.rstrip("/")
        self.pages_max_url = config.max_url
        self.commands = list(DEFAULT_COMMANDS if commands is None else commands)
        self.rules = tuple(rules)
        self.host_rules = tuple(host_rules if host_rules is not None
                                else (alerts.DiskLow(config.disk_path, config.disk_gb),))

    # state

    def state(self) -> dict:
        return dict(bot_state.read(self.state_path))

    def save(self, value: dict) -> None:
        bot_state.write(self.state_path, value)

    # helpers

    async def views(self, runs) -> list[RunView]:
        now = time.time()
        return await asyncio.to_thread(lambda: [run_views.view(self.source, run, now) for run in runs])

    async def find_one(self, update: Update, name: str):
        runs = await asyncio.to_thread(self.source.find_runs, name)
        if not runs:
            await update.effective_message.reply_text(f"No run called “{name}”.")
            return None
        return runs[0]

    async def reply_photo(self, update: Update, png: bytes | None, caption: str,
                          markup=None):
        message = update.effective_message
        if png is None:
            await message.reply_text(caption + "\n(nothing to plot yet)",
                                     parse_mode=ParseMode.HTML, reply_markup=markup)
        else:
            await message.reply_photo(png, caption=caption, parse_mode=ParseMode.HTML,
                                      reply_markup=markup)

    async def send_card(self, update: Update, run, edit: bool = False):
        view = (await self.views([run]))[0]
        now = time.time()
        link = None
        if self.pages_url:
            run = await asyncio.to_thread(self.source.get_run, view.id)
            siblings = await asyncio.to_thread(lambda: self.source.list_runs(
                experiment_ids=[run.experiment_id], limit=LINK_SIBLINGS + 1))
            chart = chartview.ChartState(
                runs=[view.id], keys=chartview.default_keys(chartview.ordered_keys(view.metrics)),
                candidates=[view.id] + [r.id for r in siblings if r.id != view.id])
            link = await asyncio.to_thread(self.link_for, chart)
        text, markup = card(view, now), card_keyboard(view, now, link)
        if edit and update.callback_query:
            try:
                await update.callback_query.edit_message_text(
                    text, parse_mode=ParseMode.HTML, reply_markup=markup)
                return
            except BadRequest as error:
                if "not modified" in str(error):
                    return
        await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML,
                                                  reply_markup=markup)

    async def curves(self, update: Update, run, metric: str | None = None):
        """The chart explorer on one run, showing `metric` or its usual curves."""
        available = chartview.ordered_keys(run.metrics)
        keys = None
        if metric:
            keys = [k for k in available if k == metric] or [k for k in available if metric in k]
            if not keys:
                await update.effective_message.reply_text(
                    f"{run.name} has no metric “{metric}”. It has: "
                    + ", ".join(available))
                return
        await self.open_chart(update, [run], keys[:1] if keys else None)

    async def compare_runs(self, update: Update, first, second, metric: str | None = None):
        """Both runs in the chart explorer, then the params that differ."""
        a, b = await self.views([first, second])
        keys = None
        if metric:
            keys = [k for k in a.histories if metric in k and k in b.histories][:1] or None
        await self.open_chart(update, [second, first], keys)
        diffs = differing(a, b)
        text = (f"<b>{esc(a.name)}</b> vs <b>{esc(b.name)}</b>: "
                + (f"{len(diffs)} params differ\n" + "\n".join(diffs[:40]) if diffs
                   else "same params"))
        await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML,
                                                  reply_markup=nav_keyboard(b.id))

    async def open_chart(self, update: Update, runs: list, keys: list[str] | None = None):
        first = runs[0]
        available = chartview.ordered_keys(
            {k for run in runs for k in run.metrics})[:chartview.MAX_KEYS]
        siblings = await asyncio.to_thread(lambda: self.source.list_runs(
            experiment_ids=[first.experiment_id], limit=9))
        ids = [run.id for run in runs]
        candidates = ids + [r.id for r in siblings if r.id not in ids]
        chart = chartview.ChartState(runs=ids, keys=keys or chartview.default_keys(available),
                                     offered=available, candidates=candidates[:10])
        chart_id = chartview.new_id()
        value = self.state()
        chartview.remember(value, chart_id, chart)
        self.save(value)
        png, text, markup = await self.render_chart(chart_id, chart)
        await self.reply_photo(update, png, text, markup)

    async def render_chart(self, chart_id: str, chart, draw: bool = True):
        """(png or None, caption, keyboard) for a chart state."""
        def work():
            runs, names, best = chartview.gather(self.source, chart)
            for run_id in chart.candidates:
                if run_id not in names:
                    names[run_id] = self.source.get_run(run_id).name or run_id[:8]
            link = None
            if self.pages_url:
                link = self.link_for(chart)
            png = None
            if draw:
                png = charts.explore(runs, chart.keys, chart.mode, chart.log, chart.smooth,
                                     best, title=" vs ".join(n for n, _ in runs))
            return png, names, link

        png, names, link = await asyncio.to_thread(work)
        card_button = InlineKeyboardButton("⬅️ Card", callback_data=f"card:{chart.runs[0]}")
        if link:
            # Ranges, metrics, runs, log and smoothing are the Mini App's job; the
            # message keeps a preview picture and the way into it.
            return png, esc(chartview.caption(chart, names)), InlineKeyboardMarkup([[
                InlineKeyboardButton("📊 Chart", web_app=WebAppInfo(url=link)), card_button]])
        rows = [[InlineKeyboardButton(label, callback_data=data) for label, data in row]
                for row in chartview.keyboard_rows(chart_id, chart, names)]
        last = [card_button,
                InlineKeyboardButton("🖼 Zoom album", callback_data=f"cv:{chart_id}:z")]
        if not chart.picker:
            rows.append(last)
        return png, esc(chartview.caption(chart, names)), InlineKeyboardMarkup(rows)

    async def send_album(self, query, chart) -> None:
        """The chart at every range as one album of high-resolution images, for
        Telegram's own viewer: tap one, pinch to zoom, swipe between ranges."""
        def build():
            runs, names, best = chartview.gather(self.source, chart)
            title = " vs ".join(name for name, _ in runs)
            pictures = []
            for mode, label in chartview.RANGES:
                if mode == "best" and best is None:
                    continue
                png = charts.explore(runs, chart.keys, mode, chart.log, chart.smooth, best,
                                     title=f"{title} · {label}", dpi=ALBUM_DPI)
                if png is not None:
                    pictures.append((label, png))
            return pictures, title

        pictures, title = await asyncio.to_thread(build)
        if not pictures:
            await query.message.reply_text("Nothing to plot yet.")
            return
        settings = [", ".join(chart.keys)]
        if chart.log:
            settings.append("log y")
        if chart.smooth:
            settings.append(f"EMA {chart.smooth:g}")
        media = [InputMediaPhoto(png, caption=fit(
            f"🖼 {title}: {' / '.join(label for label, _ in pictures)}\n"
            f"{' · '.join(settings)}\nTap a picture to zoom; swipe for the next range.",
            CAPTION_LIMIT) if index == 0 else None)
            for index, (label, png) in enumerate(pictures)]
        await query.message.reply_media_group(media)

    def link_keys(self, chart) -> list[str]:
        """What the Mini App can switch between: the chart's metrics, every loss,
        the learning rate and the speed, for the chart's runs."""
        available = set()
        for run_id in chart.runs:
            available |= set(self.source.get_run(run_id).metrics)
        wanted = [k for k in chartview.ordered_keys(available)
                  if k in chart.keys or charts.panel_of(k) == "loss"
                  or k in ("learning_rate", "lr", "seconds_per_step", "steps_per_s")]
        return list(dict.fromkeys(list(chart.keys) + wanted))

    def siblings(self, chart, count: int = LINK_SIBLINGS) -> list:
        """Recent runs of the same experiment, offered in the Mini App as overlays:
        [(run id, name, series)] with their usual curves."""
        found = []
        for run_id in [r for r in chart.candidates if r not in chart.runs][:count]:
            run = self.source.get_run(run_id)
            keys = chartview.default_keys(chartview.ordered_keys(run.metrics))
            gathered, _, _ = chartview.gather(self.source, chartview.ChartState(runs=[run_id],
                                                                               keys=keys))
            found.append((run_id, gathered[0][0], gathered[0][1]))
        return found

    def link_for(self, chart) -> str:
        """The Mini App link for a chart, gathering what it carries (blocking)."""
        wide = chartview.ChartState(runs=list(chart.runs), keys=self.link_keys(chart),
                                    candidates=list(chart.candidates))
        runs, _, _ = chartview.gather(self.source, wide)
        return self.chart_link(chart, runs, self.siblings(chart))

    def chart_link(self, chart, runs: list, siblings: list | None = None) -> str:
        """The static Mini App's address with the curves after #d=, within
        `the link limit` less a margin. The chart's own runs come first
        and keep the most points (with all of `link_keys`); recent runs of the
        experiment follow at lower resolution while they fit, off until picked."""
        base = f"{self.pages_url}/#d="
        budget = self.pages_max_url - len(base) - LINK_MARGIN
        packed = {"initial": {"runs": list(chart.runs), "keys": list(chart.keys)},
                   "runs": [{"id": run_id, "name": name, "series": series}
                            for run_id, (name, series) in zip(chart.runs, runs)]}
        text, limit = payload.fit(packed, budget)
        if limit is not None:
            return base + text       # the chart's own runs fill the link
        for run_id, name, series in siblings or []:
            if any(run["name"] == name for run in packed["runs"]):
                name = f"{name} ({run_id[:4]})"      # two runs can share a name
            packed["runs"].append({"id": run_id, "name": name, "series": series,
                                    "limit": SIBLING_POINTS})
            longer = payload.encode(packed)
            if len(longer) > budget:
                packed["runs"].pop()
                break
            text = longer
        return base + text

    async def chart_button(self, query, chart_id: str, op: str, arg: str):
        value = self.state()
        chart = chartview.recall(value, chart_id)
        if chart is None:
            await query.message.reply_text("That chart has expired; open Curves again.")
            return
        if op == "z":
            await self.send_album(query, chart)
            return
        before = (list(chart.runs), list(chart.keys), chart.mode, chart.log, chart.smooth)
        chartview.apply(chart, op, arg)
        chartview.remember(value, chart_id, chart)
        self.save(value)
        redraw = before != (chart.runs, chart.keys, chart.mode, chart.log, chart.smooth)
        png, text, markup = await self.render_chart(chart_id, chart, draw=redraw)
        try:
            if redraw and png is not None:
                await query.edit_message_media(
                    InputMediaPhoto(png, caption=fit(text, CAPTION_LIMIT),
                                    parse_mode=ParseMode.HTML), reply_markup=markup)
            else:
                await query.edit_message_caption(caption=fit(text, CAPTION_LIMIT),
                                                 parse_mode=ParseMode.HTML, reply_markup=markup)
        except BadRequest as error:
            if "not modified" not in str(error):
                raise

    # commands

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        lines = ["🤖 <b>Training bot</b>"] + [f"/{c.name} {esc(c.help)}" for c in self.commands]
        lines.append("\nRun names match exactly, then by substring; newest first.")
        await update.effective_message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

    async def runs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        views = await self.views(await asyncio.to_thread(run_views.live, self.source))
        now = time.time()
        if not views:
            await update.effective_message.reply_text("No live runs. /recent shows the last day.")
            return
        text = "\n".join(runs_line(view, now) for view in views)
        buttons = [[InlineKeyboardButton(f"{'💀 ' if v.dead else ''}{v.name} · {v.experiment}",
                                         callback_data=f"card:{v.id}")] for v in views]
        await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML,
                                                  reply_markup=InlineKeyboardMarkup(buttons))

    async def run(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.effective_message.reply_text("Usage: /run <name>")
            return
        run = await self.find_one(update, " ".join(context.args))
        if run is not None:
            await self.send_card(update, run)

    async def plot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.effective_message.reply_text("Usage: /plot <name> [metric]")
            return
        run = await self.find_one(update, context.args[0])
        if run is not None:
            await self.curves(update, run, context.args[1] if len(context.args) > 1 else None)

    async def compare(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if len(context.args) < 2:
            await update.effective_message.reply_text("Usage: /compare <a> <b> [metric]")
            return
        first = await self.find_one(update, context.args[0])
        second = await self.find_one(update, context.args[1])
        if first is not None and second is not None:
            await self.compare_runs(update, first, second,
                                    context.args[2] if len(context.args) > 2 else None)

    async def best(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = list(context.args)
        experiment = args[0] if args else None
        metric = args[1] if len(args) > 1 else "eval/validation"

        def board():
            names = self.source.experiments()
            if experiment:
                experiment_id = next((k for k, v in names.items() if v == experiment), None)
                if experiment_id is None:
                    return None, f"No experiment “{experiment}”. Experiments: " + ", ".join(
                        sorted(names.values()))
            else:
                newest = self.source.list_runs(limit=1)
                if not newest:
                    return None, "No runs."
                experiment_id = newest[0].experiment_id
            runs = self.source.list_runs(experiment_ids=[experiment_id], limit=200)
            pick = max if higher_is_better(metric) else min
            rows = []
            for run in runs:
                if metric not in run.metrics:
                    continue
                points = self.source.metric_history(run.id, metric)
                point = pick(points, key=lambda p: p.value)
                rows.append((point.value, run, point.step))
            rows.sort(key=lambda row: row[0], reverse=pick is max)
            return (names.get(experiment_id, experiment_id), rows[:10]), None

        result, problem = await asyncio.to_thread(board)
        if problem:
            await update.effective_message.reply_text(problem)
            return
        name, rows = result
        if not rows:
            await update.effective_message.reply_text(f"No run in {name} has {metric}.")
            return
        lines = [f"🏆 <b>{esc(name)}</b> by {esc(metric)} "
                 f"({'higher' if higher_is_better(metric) else 'lower'} is better)"]
        for rank, (value, run, step) in enumerate(rows, 1):
            lines.append(f"{rank}. {STATUS_EMOJI.get(run.status, '•')} "
                         f"<b>{esc(run.name)}</b> {number(value)} @ {step}")
        buttons = [[InlineKeyboardButton(run.name or run.id[:8],
                                         callback_data=f"card:{run.id}")
                    for _, run, _ in rows[i:i + 3]] for i in range(0, min(len(rows), 6), 3)]
        await update.effective_message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML,
                                                  reply_markup=InlineKeyboardMarkup(buttons))

    async def recent(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        views = await self.views(await asyncio.to_thread(run_views.recent, self.source, 24.0))
        if not views:
            await update.effective_message.reply_text("Nothing in the last 24 hours.")
            return
        now = time.time()
        lines = ["🗓 <b>Last 24 hours</b>"]
        for view in views[:20]:
            best = f" · best {number(view.best[1])}" if view.best else ""
            took = duration((view.ended or now) - view.started)
            lines.append(f"{health(view, now).split(' ')[0]} <b>{esc(view.name)}</b> "
                         f"<i>{esc(view.experiment)}</i> · {took}{best}")
        buttons = [[InlineKeyboardButton(v.name, callback_data=f"card:{v.id}")
                    for v in views[i:i + 3]] for i in range(0, min(len(views), 9), 3)]
        await update.effective_message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML,
                                                  reply_markup=InlineKeyboardMarkup(buttons))

    async def gpu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        def gather():
            lines = [nvidia_smi()]
            try:
                import psutil
                memory = psutil.virtual_memory()
                lines.append(f"RAM {memory.used / 2**30:.1f}/{memory.total / 2**30:.1f} GB · "
                             f"CPU {psutil.cpu_percent(interval=0.5):.0f}%")
            except ImportError:
                pass
            lines.append(disk_text(self.disk_path))
            for run in run_views.live(self.source):
                summary = system_summary(run.metrics)
                if summary:
                    lines.append(f"{run.name}: {summary}")
            return "\n".join(line for line in lines if line)

        await update.effective_message.reply_text("🖥 " + await asyncio.to_thread(gather))

    async def note(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if len(context.args) < 2:
            await update.effective_message.reply_text("Usage: /note <name> <text>")
            return
        run = await self.find_one(update, context.args[0])
        if run is not None:
            await asyncio.to_thread(self.source.set_note, run.id,
                                    " ".join(context.args[1:]))
            await update.effective_message.reply_text(f"📝 Noted on {run.name}.")

    async def mute(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        value = self.state()
        word = context.args[0] if context.args else ""
        if word in ("off", "0", "unmute"):
            value.pop("mute_until", None)
            self.save(value)
            await update.effective_message.reply_text("🔔 Unmuted.")
            return
        seconds = parse_duration(word or "1h")
        if seconds is None:
            await update.effective_message.reply_text("Usage: /mute <30m|2h|1d|off>")
            return
        value["mute_until"] = time.time() + seconds
        self.save(value)
        await update.effective_message.reply_text(
            f"🔕 Muted for {duration(seconds)}: no progress, evaluation or alert pushes. "
            "Starts, finishes and crashes still come through.")

    async def every(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        value = self.state()
        word = context.args[0] if context.args else ""
        if word in ("off", "default"):
            value.pop("every_minutes", None)
            self.save(value)
            await update.effective_message.reply_text(
                "Progress pushes back to each run's own interval (TELEGRAM_EVERY_MINUTES).")
            return
        try:
            minutes = float(word)
            assert minutes > 0
        except (ValueError, AssertionError):
            await update.effective_message.reply_text("Usage: /every <minutes|off>")
            return
        value["every_minutes"] = minutes
        self.save(value)
        await update.effective_message.reply_text(
            f"Progress and evaluation pushes at most every {number(minutes)} min "
            "(runs started with this monitor pick it up at once).")

    async def text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """A reply to the 📝 Note prompt: only from the person who asked for it, and
        only as a reply to that prompt."""
        message = update.effective_message
        pending = context.user_data.get("pending_note")
        reply = message.reply_to_message
        if not pending or reply is None or reply.message_id != pending["prompt"]:
            return
        context.user_data.pop("pending_note", None)
        await asyncio.to_thread(self.source.set_note, pending["run"], message.text)
        await message.reply_text("📝 Noted.")

    def allowed(self, update: Update) -> bool:
        """The configured chat, and (when `allowed_users` is set) one of those users."""
        chat, user = update.effective_chat, update.effective_user
        if chat is None or chat.id != self.chat_id:
            return False
        users = self.config.allowed_users
        return not users or (user is not None and user.id in users)

    # buttons

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if not self.allowed(update):
            await query.answer()
            return
        action, _, rest = query.data.partition(":")
        await query.answer()
        if action == "cv":
            chart_id, _, op_arg = rest.partition(":")
            op, _, arg = op_arg.partition(":")
            await self.chart_button(query, chart_id, op, arg)
            return
        if action == "ign":
            value = self.state()
            value["ignored"] = sorted(set(value.get("ignored", [])) | {rest[:16]})
            self.save(value)
            await query.edit_message_reply_markup(None)
            return
        if action == "ignall":
            value = self.state()
            value["ignored"] = sorted(set(value.get("ignored", []))
                                      | {short_key(k) for k in value.pop("digest_keys", [])})
            self.save(value)
            await query.edit_message_reply_markup(None)
            return
        run_id, _, other = rest.partition(":")
        run = await asyncio.to_thread(self.source.get_run, run_id)
        if action == "card":
            await self.send_card(update, run, edit=True)
        elif action == "curves":
            await self.curves(update, run)
        elif action == "params":
            view = (await self.views([run]))[0]
            pages = params_pages(view)
            page = min(int(other) if other.isdigit() else 0, len(pages) - 1)
            markup = params_keyboard(run_id, page, len(pages))
            if other:   # a page button: turn the page in place
                try:
                    await query.edit_message_text(pages[page], parse_mode=ParseMode.HTML,
                                                  reply_markup=markup)
                    return
                except BadRequest as error:
                    if "not modified" in str(error):
                        return
            await update.effective_message.reply_text(pages[page], parse_mode=ParseMode.HTML,
                                                      reply_markup=markup)
        elif action == "pfile":
            view = (await self.views([run]))[0]
            await update.effective_message.reply_document(
                params_file(view), filename=f"{view.name}-params.txt",
                caption=f"{view.name}: {len(view.params)} params", reply_markup=nav_keyboard(run_id))
        elif action == "cmpmenu":
            others = [r for r in await asyncio.to_thread(lambda: self.source.list_runs(
                experiment_ids=[run.experiment_id], limit=9))
                if r.id != run_id][:8]
            if not others:
                await update.effective_message.reply_text("No other run in this experiment.")
                return
            buttons = [[InlineKeyboardButton(
                f"{STATUS_EMOJI.get(r.status, '•')} {r.name}",
                callback_data=f"cmp:{run_id}:{r.id[:12]}")
                for r in others[i:i + 2]] for i in range(0, len(others), 2)]
            await update.effective_message.reply_text(
                f"Compare {run.name} with…", reply_markup=InlineKeyboardMarkup(buttons))
        elif action == "cmp":
            runs = await asyncio.to_thread(lambda: self.source.list_runs(
                experiment_ids=[run.experiment_id], limit=200))
            second = next((r for r in runs if r.id.startswith(other)), None)
            if second is not None:
                await self.compare_runs(update, second, run)
        elif action == "sys":
            history = await asyncio.to_thread(run_views.system_history, self.source, run)
            png = await asyncio.to_thread(charts.system, history,
                                          f"{run.name} · system")
            if png is None:
                await update.effective_message.reply_text(
                    f"{run.name} has no system metrics (it started before the "
                    "monitor logged them). /gpu shows the machine now.",
                    reply_markup=nav_keyboard(run_id))
            else:
                await self.reply_photo(
                    update, png, f"<b>{esc(run.name)}</b> · "
                    + system_summary(run.metrics), nav_keyboard(run_id))
        elif action == "note":
            prompt = await update.effective_message.reply_text(
                f"Reply to this message with the note for {run.name}:",
                reply_markup=ForceReply(selective=True))
            context.user_data["pending_note"] = {"run": run_id, "prompt": prompt.message_id}
        elif action == "kill":
            await update.effective_message.reply_text(
                f"Mark {run.name} KILLED in MLflow? This only changes the record; "
                "no process is touched.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("Yes, mark killed", callback_data=f"killok:{run_id}"),
                    InlineKeyboardButton("Cancel", callback_data=f"card:{run_id}")]]))
        elif action == "killok":
            view = (await self.views([run]))[0]
            if view.live and view.pid_alive:
                await query.edit_message_text(
                    f"{run.name}'s process is still running; not marked.")
                return
            await asyncio.to_thread(self.source.mark_killed, run_id)
            await query.edit_message_text(f"🪦 {run.name} marked KILLED.")

    # the watcher

    async def watch(self, context: ContextTypes.DEFAULT_TYPE):
        try:
            await self._watch(context)
        except Exception:  # noqa: BLE001 -- the next minute tries again
            log.exception("watch failed")

    async def _watch(self, context: ContextTypes.DEFAULT_TYPE):
        now = time.time()

        def gather():
            views = [run_views.view(self.source, run, now) for run in run_views.live(self.source)]
            found = alerts.evaluate(self.source, views, self.rules, self.host_rules, now)
            return views, found, nvidia_smi()

        views, found, gpu = await asyncio.to_thread(gather)
        value = self.state()
        muted = bot_state.muted(value, now)
        active = {alert.key for alert in found}
        alerted = set(value.get("alerted", [])) & active
        ignored = set(value.get("ignored", [])) & {short_key(key) for key in active}
        fresh = [alert for alert in found
                 if not (alert.key in alerted or short_key(alert.key) in ignored or muted)]
        if len(fresh) > DIGEST_OVER:
            await self.digest(context, fresh)
            alerted |= {alert.key for alert in fresh}
            fresh = []
        for alert in fresh:
            buttons = [InlineKeyboardButton("🙈 Ignore",
                                            callback_data=f"ign:{short_key(alert.key)}")]
            if alert.run_id:
                buttons.insert(0, InlineKeyboardButton("Open", callback_data=f"card:{alert.run_id}"))
            if alert.can_kill:
                buttons.insert(0, InlineKeyboardButton("🪦 Mark killed",
                                                       callback_data=f"kill:{alert.run_id}"))
            try:
                await context.bot.send_message(self.chat_id, alert.text,
                                               reply_markup=InlineKeyboardMarkup([buttons]))
            except BadRequest as error:   # one bad alert must not stall the watcher
                log.warning("alert %s not sent: %s", alert.key, error)
            alerted.add(alert.key)
        status_id = await self.pin_status(
            context, status_text(views, gpu, disk_text(self.disk_path), now,
                                 value.get("mute_until", 0)),
            value.get("status_message_id"))
        # Read, change and write with no await between, so nothing a button
        # wrote meanwhile (a chart, an ignore) is lost.
        value = self.state()
        value["alerted"], value["ignored"] = sorted(alerted), sorted(ignored)
        value["status_message_id"] = status_id
        self.save(value)

    async def digest(self, context, fresh: list) -> None:
        """Many alerts at once (say, the first start after runs died unnoticed) as
        one message; each run's card still offers its own Mark killed."""
        lines = [f"⚠️ <b>{len(fresh)} alerts</b>"] + [f"• {esc(a.text)}" for a in fresh[:25]]
        if len(fresh) > 25:
            lines.append(f"… and {len(fresh) - 25} more")
        value = self.state()
        value["digest_keys"] = [a.key for a in fresh]
        self.save(value)
        await context.bot.send_message(
            self.chat_id, "\n".join(lines), parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🙈 Ignore all", callback_data="ignall:")]]))

    async def pin_status(self, context, text: str, message_id: int | None) -> int | None:
        """Edits the pinned status in place; posts and pins a new one if it is gone."""
        if message_id:
            try:
                await context.bot.edit_message_text(text, self.chat_id, message_id,
                                                    parse_mode=ParseMode.HTML)
                return message_id
            except BadRequest as error:
                if "not modified" in str(error):
                    return message_id
                log.info("status message gone (%s); posting a new one", error)
        message = await context.bot.send_message(self.chat_id, text, parse_mode=ParseMode.HTML,
                                                 disable_notification=True)
        try:
            await context.bot.pin_chat_message(self.chat_id, message.message_id,
                                               disable_notification=True)
        except BadRequest as error:
            log.warning("could not pin the status: %s", error)
        return message.message_id

    # wiring

    def application(self, hello: bool = True, watch_interval: float = 60) -> Application:
        """A python-telegram-bot Application answering `self.commands` in the
        configured chat, with the watcher on its job queue."""
        async def post_init(app: Application):
            await app.bot.set_my_commands([BotCommand(c.name, c.help[:256])
                                           for c in self.commands])
            if hello:
                await app.bot.send_message(
                    self.chat_id, "🤖 Training bot online. /runs for live runs, /help for the "
                    "rest. Alerts and a pinned status follow from here.",
                    disable_notification=True)

        redactor = RedactToken(self.config.token)
        for name in ("httpx", "httpcore", "telegram", "mlflow_telegram"):
            logger = logging.getLogger(name)
            if not any(isinstance(f, RedactToken) for f in logger.filters):
                logger.addFilter(redactor)
        logging.getLogger("httpx").setLevel(logging.WARNING)   # its URLs carry the token
        if self.chat_id < 0 and not self.config.allowed_users:
            log.warning("the bot answers a group: every member can use it; set "
                        "MLFLOW_TELEGRAM_ALLOWED_USERS to limit it to some users")
        app = (Application.builder().bot(FittingBot(self.config.token))
               .post_init(post_init).build())
        only = filters.Chat(chat_id=self.chat_id)
        if self.config.allowed_users:
            only = only & filters.User(user_id=list(self.config.allowed_users))
        for command in self.commands:
            async def answer(update, context, command=command):
                await command.handler(self, update, context)
            app.add_handler(CommandHandler(command.name, answer, filters=only))
        app.add_handler(CommandHandler("start", self.help, filters=only))
        app.add_handler(CallbackQueryHandler(self.button))
        app.add_handler(MessageHandler(only & filters.TEXT & filters.REPLY & ~filters.COMMAND,
                                       self.text))
        app.add_handler(ChatMemberHandler(self.joined, ChatMemberHandler.MY_CHAT_MEMBER))
        app.add_error_handler(self.error)
        if watch_interval:
            app.job_queue.run_repeating(self.watch, interval=watch_interval, first=5,
                                        name="watch")
        return app

    async def joined(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Leaves any group or channel it is added to other than the configured chat."""
        change = update.my_chat_member
        if change is None or change.chat.id == self.chat_id:
            return
        if change.new_chat_member.status in ("member", "administrator"):
            log.warning("added to chat %s, which is not the configured chat: leaving",
                        change.chat.id)
            try:
                await context.bot.leave_chat(change.chat.id)
            except Exception as error:  # noqa: BLE001
                log.warning("could not leave chat %s: %s", change.chat.id, error)

    async def error(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Logs the details on this machine (token masked) and tells the chat only
        the kind of error, never its text (paths, hosts, secrets)."""
        log.error("handler failed: %s", redact_text(repr(context.error), [self.config.token]),
                  exc_info=context.error)
        message = getattr(update, "effective_message", None)
        if message is not None and hasattr(update, "effective_chat") and self.allowed(update):
            await message.reply_text(
                f"⚠️ Something went wrong ({type(context.error).__name__}); the details are "
                "in the bot's log.")


def _bound(name: str) -> Handler:
    async def handler(bot: TrainBot, update, context):
        await getattr(bot, name)(update, context)
    handler.__name__ = name
    return handler


#: The commands the bot answers unless told otherwise; extend or filter this list.
DEFAULT_COMMANDS: list[Command] = [
    Command("runs", "live runs, with progress and ETA", _bound("runs")),
    Command("run", "<name> — one run's card", _bound("run")),
    Command("plot", "<name> [metric] — curves", _bound("plot")),
    Command("compare", "<a> <b> [metric] — overlaid curves and differing params",
            _bound("compare")),
    Command("best", "[experiment] [metric] — leaderboard", _bound("best")),
    Command("recent", "runs of the last 24 hours", _bound("recent")),
    Command("gpu", "GPU, memory and disk now", _bound("gpu")),
    Command("note", "<name> <text> — set a run's note", _bound("note")),
    Command("mute", "<30m|2h|off> — silence pushes and alerts", _bound("mute")),
    Command("every", "<minutes|off> — progress push interval", _bound("every")),
    Command("help", "this list", _bound("help")),
]


def create_application(config: Config | None = None, source: RunSource | None = None, *,
                       commands: list[Command] | None = None, rules=alerts.DEFAULT_RULES,
                       host_rules=None, hello: bool = True,
                       watch_interval: float = 60) -> Application:
    """A ready-to-run Application (call `.run_polling()` on it).

    `config` defaults to `Config.from_env()`; `source` to an `MlflowSource` on
    `config.tracking_uri`."""
    config = config or Config.from_env()
    if not config.telegram_ready:
        raise ValueError("a bot token and chat id are required (see mlflow_telegram.config)")
    if source is None:
        from mlflow_telegram.sources.mlflow import MlflowSource
        if not config.tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI is not set")
        source = MlflowSource(config.tracking_uri)
    bot = TrainBot(config, source, commands=commands, rules=rules, host_rules=host_rules)
    return bot.application(hello=hello, watch_interval=watch_interval)


def run_bot(config: Config | None = None, **options) -> None:
    """Runs the bot until interrupted (long polling)."""
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        level=logging.INFO)
    create_application(config, **options).run_polling(allowed_updates=ALLOWED_UPDATES,
                                                      drop_pending_updates=True)
