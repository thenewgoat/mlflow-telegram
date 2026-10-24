# Changelog

## 0.1.0 (unreleased)

First public version.

- `Monitor`: MLflow logging (params per config key, metrics, tags, system
  metrics, artifact on finish) and throttled Telegram messages with charts;
  pid/host/patience/NaN tags for the bot; replaceable `Formatter`.
- Bot (python-telegram-bot 21): `/runs`, `/run`, `/plot`, `/compare`, `/best`,
  `/recent`, `/gpu`, `/note`, `/mute`, `/every`; run cards; pinned status;
  alert rules (died, stalled, diverging, GPU idle, disk low, early stop near);
  commands and rules pluggable.
- 📊 Chart Mini App: static page (`build-page`), curves in the link fragment
  (payload format v1, `docs/payload.md`), pinch zoom, range presets, toggles,
  log scale, smoothing, readout strip; Telegram safe areas respected.
- `RunSource` protocol with `MlflowSource`.
- CLI: `bot`, `build-page`, `probe`, `cleanup`, `config`.
