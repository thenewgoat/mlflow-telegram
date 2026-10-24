"""The bot with one extra alert rule: warn when a run is slower than 2 s/step.

    mlflow-telegram's settings in .env, then: python examples/custom_alert.py
"""

from mlflow_telegram.alerts import DEFAULT_RULES, Alert, Rule
from mlflow_telegram.bot import run_bot


class TooSlow(Rule):
    """Fires once per run while it stays slow; the bot forgets it when it speeds up."""

    def __init__(self, seconds_per_step: float = 2.0):
        self.limit = seconds_per_step

    def check(self, view, now, gpu=None):
        if view.live and view.seconds_per_step and view.seconds_per_step > self.limit:
            return Alert(f"{view.id}:slow", view.id,
                         f"🐢 {view.name} runs at {view.seconds_per_step:.1f} s/step")
        return None


if __name__ == "__main__":
    run_bot(rules=(*DEFAULT_RULES, TooSlow()))
