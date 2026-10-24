"""The bot with an extra command, `/lr <name>`: the latest learning rate of a run.

    mlflow-telegram's settings in .env, then: python examples/custom_command.py
"""

import asyncio

from mlflow_telegram.bot import DEFAULT_COMMANDS, Command, run_bot


async def latest_lr(bot, update, context):
    """`bot` is the TrainBot: `bot.source` reads runs (any RunSource)."""
    if not context.args:
        await update.effective_message.reply_text("Usage: /lr <run name>")
        return
    runs = await asyncio.to_thread(bot.source.find_runs, context.args[0])
    if not runs:
        await update.effective_message.reply_text("No such run.")
        return
    run = runs[0]
    value = run.metrics.get("lr", run.metrics.get("learning_rate"))
    await update.effective_message.reply_text(f"{run.name}: lr = {value}")


if __name__ == "__main__":
    run_bot(commands=DEFAULT_COMMANDS + [Command("lr", "<name> — latest learning rate",
                                                 latest_lr)])
