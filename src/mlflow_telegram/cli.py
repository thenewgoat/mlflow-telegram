"""`mlflow-telegram`: the command line.

    mlflow-telegram bot                     run the bot (long polling)
    mlflow-telegram build-page DIR          write the Mini App site into DIR
    mlflow-telegram probe                   measure the longest chart link Telegram takes
    mlflow-telegram cleanup [--apply]       list (or mark KILLED) runs that died silently
    mlflow-telegram cleanup --revert FILE   undo a cleanup
    mlflow-telegram config                  show the settings in effect (secrets hidden)

Settings come from the environment and `.env` (see `mlflow_telegram.config`).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from mlflow_telegram import __version__
from mlflow_telegram.config import Config

__all__ = ["main"]


def _source(config: Config):
    from mlflow_telegram.sources.mlflow import MlflowSource
    if not config.tracking_uri:
        raise SystemExit("MLFLOW_TRACKING_URI is not set")
    return MlflowSource(config.tracking_uri)


def _bot(args, config):
    from mlflow_telegram.bot import run_bot
    run_bot(config, hello=not args.quiet)
    return 0


def _build_page(args, config):
    from mlflow_telegram.page import build_page
    for name in build_page(args.directory):
        print(f"{args.directory.rstrip('/')}/{name}")
    print("Publish this directory on any static HTTPS host (GitHub Pages works) and set "
          "MLFLOW_TELEGRAM_PAGES_URL to its address.")
    return 0


def _probe(args, config):
    from mlflow_telegram.page import probe_link_limit
    if not (config.telegram_ready and config.pages_url):
        raise SystemExit("needs a bot token, chat id and MLFLOW_TELEGRAM_PAGES_URL")
    longest = probe_link_limit(config.token, config.chat_id, config.pages_url)
    print(f"longest accepted link: {longest} characters; "
          f"set MLFLOW_TELEGRAM_MAX_URL={longest - 64}")
    return 0


def _cleanup(args, config):
    from mlflow_telegram import cleanup
    source = _source(config)
    if args.revert:
        print(f"reverted {cleanup.revert(source, args.revert)} runs")
        return 0
    record = cleanup.record_path(config)
    views, found = cleanup.run(source, apply=args.apply, hours=args.hours, keep=args.keep,
                               record=record)
    for view in views:
        print(f"{'dead' if view in found else 'keep'}  {view.experiment:26} {view.name:22} "
              f"{view.id}")
    if args.apply:
        print(f"marked {len(found)} KILLED; record in {record}")
    else:
        print(f"{len(found)} would be marked KILLED; --apply to do it")
    return 0


def _config(args, config):
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if field.name == "token" and value:
            value = "<set>"
        print(f"{field.name:14} {value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlflow-telegram",
                                     description="Watch MLflow training runs from Telegram.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--env-file", help="a .env to load (default: ./.env)")
    commands = parser.add_subparsers(dest="command", required=True)
    bot = commands.add_parser("bot", help="run the bot")
    bot.add_argument("--quiet", action="store_true", help="no 'online' message at start")
    bot.set_defaults(run=_bot)
    page = commands.add_parser("build-page", help="write the Mini App site")
    page.add_argument("directory")
    page.set_defaults(run=_build_page)
    commands.add_parser("probe", help="measure the longest chart link").set_defaults(run=_probe)
    clean = commands.add_parser("cleanup", help="runs that died without telling MLflow")
    clean.add_argument("--apply", action="store_true")
    clean.add_argument("--hours", type=float, default=2.0,
                       help="silence that counts as dead for runs without a pid tag")
    clean.add_argument("--keep", action="append", default=[], metavar="NAME")
    clean.add_argument("--revert", metavar="FILE")
    clean.set_defaults(run=_cleanup)
    commands.add_parser("config", help="show the settings in effect").set_defaults(run=_config)
    args = parser.parse_args(argv)
    return args.run(args, Config.from_env(env_file=args.env_file))


if __name__ == "__main__":
    sys.exit(main())
