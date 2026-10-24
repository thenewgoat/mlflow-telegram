# Contributing

```bash
git clone https://github.com/thenewgoat/mlflow-telegram
cd mlflow-telegram
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check src tests
pytest
```

The tests need no network and no GPU. They use a temporary sqlite MLflow store
and a stubbed Telegram.

## Guidelines

- **The Monitor must never stop a training run.** Every MLflow or Telegram
  failure inside it is printed and dropped.
- **The bot answers one chat only.** Its writes to the tracker are limited to a
  run's status and note, and it never touches processes.
- **Payload changes need a new version.** If you change the chart link format,
  bump `payload.VERSION`, update `docs/payload.md`, and keep `fragment.js` in
  step with it.
- **New trackers** go behind `RunSource` in `mlflow_telegram/sources/`.
