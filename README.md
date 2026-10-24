# mlflow-telegram

Watch your MLflow training runs from Telegram. Get a message when a run starts,
evaluates, finishes or crashes. Ask a bot how your runs are doing. Open
zoomable, comparable loss curves in a Telegram Mini App, with nothing to host
but one static page.

> **Unofficial community project.** Not affiliated with, endorsed by or
> sponsored by MLflow, the Linux Foundation or Databricks. "MLflow" is a
> trademark of its owners.

<!-- screenshot: run card · 📊 Chart Mini App · alert -->

## What you get

- **`Monitor`**: wrap your training loop and every param, metric and tag goes to
  MLflow. Telegram gets throttled progress and evaluation messages, with a chart
  attached. It also records what the bot needs to tell a crashed run from a slow
  one: pid and host tags, a NaN flag, patience, and system metrics. It needs only
  the standard library and never stops a run.
- **The bot**:
  - Commands: `/runs` (live runs with progress and ETA), `/run <name>` (a run
    card), `/plot`, `/compare`, `/best`, `/recent`, `/gpu`, `/note`, `/mute`,
    `/every`.
  - A pinned status message, edited in place.
  - Alerts: a run died or stalled, a loss is diverging, the GPU is idle, the disk
    is low, early stopping is one evaluation away.
- **📊 Chart**: a Telegram Mini App with pinch-zoom on the step axis, range
  presets (all / last 50% / last 20% / since best), metric and run toggles, log
  scale, smoothing, and a readout strip that never covers the curves.
- **Pluggable**: run sources (`RunSource`), alert rules (`Rule`), bot commands
  (`Command`) and message formats (`Formatter`) are all swappable.

## Privacy model

- **The chart data is never hosted.** When the bot sends a chart, it packs the
  curves into the 📊 Chart button's link, after `#`. Browsers never send that
  part of a link to the server, so your static page host (GitHub Pages, say)
  serves the same empty page to everyone. The curves only travel inside your
  private Telegram chat. The format is specified in
  [docs/payload.md](docs/payload.md).
- **Nothing listens on your machine.** The bot uses outbound long polling.
  There's no webhook and no tunnel.
- **The bot answers only your chat id** and ignores everyone else. Its only
  writes to MLflow are a run's status (a "Mark killed" button, after a check
  that the process is gone) and its note.

## Install

```bash
pip install "mlflow-telegram[all]"          # everything
pip install mlflow-telegram                 # Monitor only (+ mlflow, if you log to it)
pip install "mlflow-telegram[bot]"          # the bot
pip install "mlflow-telegram[system]"       # CPU/GPU system metrics (psutil, NVML)
```

Python 3.10+. A `sqlite:///` tracking store needs the full `mlflow` package, or
`sqlalchemy` + `alembic`.

## Quickstart (5 minutes)

1. **Create a bot**: message [@BotFather](https://t.me/BotFather), send
   `/newbot`, and copy the token.
2. **Find your chat id**: send your bot any message, then open
   `https://api.telegram.org/bot<token>/getUpdates` and copy `message.chat.id`.
3. **Configure** in a `.env` file in your working directory (or as environment
   variables):

   ```bash
   MLFLOW_TRACKING_URI=sqlite:///mlflow.db
   MLFLOW_TELEGRAM_TOKEN=123456:ABC...
   MLFLOW_TELEGRAM_CHAT_ID=123456789
   ```

4. **Run the bot** (keep it running in tmux, systemd, or similar):

   ```bash
   mlflow-telegram bot
   ```

5. **Optional: the interactive chart.** Build the page and publish it on any
   static HTTPS host. With GitHub Pages: create an empty public repo, push the
   directory to `main`, and enable Pages from the root.

   ```bash
   mlflow-telegram build-page site/
   ```

   Then add the page's address and measure Telegram's link limit once:

   ```bash
   MLFLOW_TELEGRAM_PAGES_URL=https://<you>.github.io/<repo>
   mlflow-telegram probe       # prints the value for MLFLOW_TELEGRAM_MAX_URL
   ```

   Every chart and run card now has a 📊 Chart button.

## Use the Monitor in a training loop

```python
from mlflow_telegram import Monitor

with Monitor("train.py", name="lr-3e-4", config=vars(args),
             max_steps=args.steps, patience=10, out="runs/lr-3e-4") as monitor:
    best = float("inf")
    for step in range(1, args.steps + 1):
        loss = train_step()
        monitor.progress(step, train_loss=loss, lr=scheduler.get_last_lr()[0])
        if step % 500 == 0:
            val = evaluate()
            improved = val < best
            best = min(best, val)
            monitor.evaluation(step, best=improved, validation=val,
                               checkpoint="runs/lr-3e-4/best.pt" if improved else None)
    monitor.finish(best_validation=best)
```

- **When it's on:** without `MLFLOW_TRACKING_URI` and Telegram settings, the
  Monitor does nothing, so tests and quick runs stay quiet.
- **Crashes:** they're reported with the end of the traceback, and Ctrl-C as
  "interrupted". The exception still propagates.
- **Messages:** to change the wording, subclass `Formatter` and pass
  `formatter=`.
- **Examples:** see [`examples/`](examples).

## Library use

| Import | What it is |
|---|---|
| `mlflow_telegram.Monitor`, `Formatter` | training-side logging and messages |
| `mlflow_telegram.Config` | every setting, from the environment |
| `mlflow_telegram.MlflowSource`, `RunSource`, `RunInfo`, `Point` | reading runs; implement `RunSource` for another tracker |
| `mlflow_telegram.runs.view(source, run)` | progress, ETA, staleness and best evaluation for a run |
| `mlflow_telegram.alerts.Rule`, `HostRule`, `DEFAULT_RULES`, `evaluate` | alert rules |
| `mlflow_telegram.bot.create_application`, `run_bot`, `Command`, `DEFAULT_COMMANDS`, `TrainBot` | the bot |
| `mlflow_telegram.encode`, `decode`, `payload.fit` | the chart link format |
| `mlflow_telegram.build_page`, `page.probe_link_limit` | the Mini App page |
| `mlflow_telegram.charts.explore`, `run_panels`, `compare` | PNG charts (matplotlib) |

A custom command and a custom alert:

```python
from mlflow_telegram.alerts import DEFAULT_RULES, Alert, Rule
from mlflow_telegram.bot import DEFAULT_COMMANDS, Command, run_bot

class TooSlow(Rule):
    def check(self, view, now, gpu=None):
        if view.live and view.seconds_per_step and view.seconds_per_step > 2:
            return Alert(f"{view.id}:slow", view.id, f"🐢 {view.name}: {view.seconds_per_step:.1f} s/step")

async def ping(bot, update, context):
    await update.effective_message.reply_text("pong")

run_bot(commands=DEFAULT_COMMANDS + [Command("ping", "check the bot is alive", ping)],
        rules=(*DEFAULT_RULES, TooSlow()))
```

## Settings

| Variable | Fallback names | Default | Meaning |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | | | the MLflow store |
| `MLFLOW_TELEGRAM_TOKEN` | `TELEGRAM_BOT_TOKEN` | | bot token |
| `MLFLOW_TELEGRAM_CHAT_ID` | `TELEGRAM_CHAT_ID` | | the only chat answered |
| `MLFLOW_TELEGRAM_PAGES_URL` | `TRAINBOT_PAGES_URL` | | the Mini App page; unset = no 📊 Chart |
| `MLFLOW_TELEGRAM_MAX_URL` | `TRAINBOT_PAGES_MAX_URL` | 4096 | longest chart link (`probe`) |
| `MLFLOW_TELEGRAM_STATE` | `TRAINBOT_STATE` | `~/.local/state/mlflow-telegram/state.json` | mute, interval, alerts sent, charts |
| `MLFLOW_TELEGRAM_DISK_PATH` | `TRAINBOT_DISK_PATH` | `.` | disk watched by the low-disk alert |
| `MLFLOW_TELEGRAM_DISK_GB` | `TRAINBOT_DISK_GB` | 20 | low-disk threshold |
| `MLFLOW_TELEGRAM_EVERY_MINUTES` | `TELEGRAM_EVERY_MINUTES` | 30 | Monitor's progress-push interval |
| `MLFLOW_TELEGRAM_ENV_FILE` | `MONITOR_ENV_FILE` | `./.env` | the `.env` to load |

`mlflow-telegram config` shows what is in effect (the token hidden).

## Command line

```
mlflow-telegram bot                    run the bot
mlflow-telegram build-page DIR         write the Mini App site
mlflow-telegram probe                  measure the longest chart link Telegram accepts
mlflow-telegram cleanup [--apply]      list / mark KILLED runs that died without telling MLflow
mlflow-telegram cleanup --revert FILE  undo that
mlflow-telegram config                 show settings
```

## Licence

MIT. See [LICENSE](LICENSE). The page loads [Apache ECharts](https://echarts.apache.org)
(Apache-2.0) and Telegram's `telegram-web-app.js`.
