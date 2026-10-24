# Security and privacy audit, 2026-10-25 (before first publication)

## Scope and method

**What was reviewed:**
- this package, its git history, and the build output of its Mini App page;
- the workspace adapter that uses it;
- the chart-page repository that had already been published.

**How:**
- **Secret scanning:** gitleaks 8.30.1 (checksum-verified binary, run from a
  temporary directory) over the working trees and full git histories.
- **Value search:** the real bot token and chat id were searched for in every
  file, git history, log file and the bot's tmux scrollback, without printing
  them.
- **Dependencies:** pip-audit over all 68 resolved dependencies.
- **Code review:** an independent adversarial review covering authorization,
  HTML injection, Mini App XSS and DoS, secret leakage, filter strings,
  subprocesses, CI and privacy.
- **Runtime check:** the hardened page was loaded in headless Chrome under its
  CSP.

## Results

| # | Severity | Finding | Location | Status |
|---|---|---|---|---|
| 1 | — | gitleaks: no leaks in either repo's tree or history, or in the workspace repository's full history | all | clean |
| 2 | — | Real token and chat id: 0 occurrences in files, history, 52 log files and tmux scrollback; no logged `api.telegram.org/bot…` URLs | all | clean |
| 3 | — | pip-audit: no known vulnerabilities | dependencies | clean |
| 4 | High | The first draft of SECURITY.md promised protections the code didn't have | SECURITY.md | fixed: implemented, and the doc rewritten to match |
| 5 | Medium | Secret params (`config=vars(args)` with `--api-key`, `hf_token`, …) would be logged to MLflow and shown on the Params page, the full-list file and /compare | monitor.py, runs.py, bot/format.py | fixed: `redact.py`, masked in `Monitor` and in `all_params` |
| 6 | Medium | Token could be printed: a token with `\r` (CRLF `.env`) makes http.client raise InvalidURL quoting the URL | notify.py, config.py | fixed: stripped and shape-checked; masked in every printed error |
| 7 | Medium | In a group, every member could use the bot (params, mute, Mark killed); notes could be hijacked by any member's next message | bot/app.py | fixed: `MLFLOW_TELEGRAM_ALLOWED_USERS`, notes only as a reply from the asker, startup warning, README and SECURITY.md |
| 8 | Medium | Handler errors and crash tracebacks sent raw (paths, hosts, URL secrets) | bot/app.py, monitor.py | fixed: chat gets the error type only; crash text redacted (token, URL credentials, secret query values, home directory); `send_traceback=False` option |
| 9 | Low | httpx could log the token for `create_application` users | bot/app.py | fixed: token-masking log filter and httpx at WARNING inside the application |
| 10 | Low | Page had no CSP or referrer policy | page/ | fixed: every script is a file; CSP `default-src 'none'; script-src 'self' https://telegram.org; connect-src 'none'`; `no-referrer` |
| 11 | Low | A crafted link could make the page exhaust memory (`n: 1e9`, a deflate bomb) or crash on odd names | page/assets/fragment.js | fixed: size caps, streamed inflate capped at 4 MB, schema checks, bounds-checked varints, prototype-free objects; tested in node with hostile links |
| 12 | Low | A long alert key could exceed Telegram's 64-byte callback limit and stall the watcher; the digest was cut with a raw `[:4000]` | bot/app.py | fixed: hashed keys, per-alert error handling, `fit()` |
| 13 | Low | State file and cleanup record created 0644 through a predictable temp name | state.py, cleanup.py | fixed: 0600 files and 0700 directory, created with `mkstemp` |
| 14 | Low | `config` printed `tracking_uri` with any `user:password@` | cli.py | fixed: masked |
| 15 | Low | Run search built MLflow filter strings from user text (no injection, but names with `'`, `%` or `_` misbehaved) | sources/mlflow.py | fixed: matched in Python, no filter strings |
| 16 | Low | `$` in run names was parsed as matplotlib maths; charts switched the process's matplotlib backend | charts.py | fixed: `parse_math` off; figures drawn without pyplot |
| 17 | Low | CI actions unpinned, with default token permissions | .github/workflows/ci.yml | fixed: SHA-pinned, `contents: read`, `persist-credentials: false` |
| 18 | Privacy | The previously published chart-page repo had the owner's name and Gmail in every commit, and commit messages naming a private branch and private hashes | old chart-page repo | fixed by the owner: repo deleted. The replacement repo (mlflow-telegram-page) is prepared locally with a single commit by the GitHub no-reply identity and a neutral message, not yet pushed |
| 19 | Info | The Monitor's start message includes the machine's hostname; MLflow tags include host, pid, script and output path | monitor.py | by design, documented |
| 20 | Info | Commit timestamps carry the author's time zone | git metadata | accepted |

## Checked and clean

- **Strangers:** commands, buttons, replies and the error reply are limited to
  the configured chat. There is no inline-query or `web_app_data` handler.
- **HTML in messages:** every run, experiment, metric, param, tag, note and
  error value inserted into HTML-formatted text is escaped.
- **Page XSS:** no route from link data to HTML. Option lists are built node by
  node, and the readout escapes its labels.
- **`nvidia-smi`:** fixed arguments, no shell, with a timeout.
- **Privacy:**
  - the package tree and history contain no personal emails, home paths,
    hostnames, IPs, private project names or run names;
  - its single commit uses the GitHub no-reply identity;
  - the replacement chart-page repo, prepared locally, is the same.

## Not changed (owner's decision)

- Nothing has been published: the mlflow-telegram repo does not exist on
  GitHub, and the replacement page repo is empty until pushed.
- The bot token was never found logged or committed, so rotating it is not
  required. Rotate it in @BotFather if it was ever pasted somewhere shared.
