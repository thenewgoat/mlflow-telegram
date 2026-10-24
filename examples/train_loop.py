"""A minimal training loop reporting through Monitor.

    MLFLOW_TRACKING_URI=sqlite:///mlflow.db python examples/train_loop.py

Add MLFLOW_TELEGRAM_TOKEN and MLFLOW_TELEGRAM_CHAT_ID to get the Telegram
messages too. Without any settings the Monitor stays silent.
"""

import math
import random

from mlflow_telegram import Monitor

STEPS, EVAL_EVERY = 2000, 200
config = {"lr": 3e-4, "batch_size": 32, "seed": 0, "optimizer": "adamw"}

random.seed(config["seed"])
with Monitor("train_loop.py", name="demo-lr-3e-4", config=config, max_steps=STEPS,
             patience=5, every_minutes=1) as monitor:
    best = float("inf")
    for step in range(1, STEPS + 1):
        loss = 3.0 * math.exp(-step / 500) + 0.3 + random.gauss(0, 0.02)
        monitor.progress(step, train_loss=loss, lr=config["lr"])
        if step % EVAL_EVERY == 0:
            validation = 3.2 * math.exp(-step / 600) + 0.35 + random.gauss(0, 0.01)
            improved = validation < best
            best = min(best, validation)
            monitor.evaluation(step, best=improved, validation=validation)
    monitor.finish(best_validation=best)
