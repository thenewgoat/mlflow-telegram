# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately, using GitHub's **Report a
vulnerability** button on the repository's Security tab (private vulnerability
reporting). Don't open a public issue for them.

## What the package does to protect you

### Access

- **One chat.** The bot answers only the configured chat. Commands, buttons,
  replies and the error reply are all filtered, and `run_bot` asks Telegram only for
  messages, button presses and membership changes.
- **Other groups.** If it is added to any other group or channel, it leaves.
- **Groups need an allow-list.** Use a **private** chat. In a group, every
  member can use the bot unless `MLFLOW_TELEGRAM_ALLOWED_USERS` lists the user
  ids allowed to. The bot warns at start when it answers a group without one.
- **Notes.** A note is taken only from the person who tapped 📝 Note, and only
  as a reply to the bot's prompt.
- **Writes.** The bot's only writes to the tracker are a run's status ("Mark
  killed", with a confirmation step, refused while the run's process is alive on
  this host) and a run's note. It never starts or stops processes.

### Secrets

- **The bot token** gives full control of the bot. Keep it in the environment or
  an uncommitted `.env`. It is stripped of stray whitespace (such as `\r` from
  Windows line endings) and checked for Telegram's shape before it is used.
  It is masked in log records (the `httpx`, `httpcore`, `telegram` and
  `mlflow_telegram` loggers, and `httpx` is kept at WARNING) and in every error
  the package prints.
- **Params and tags.** Values of params whose names look secret (`token`,
  `secret`, `password`, `api_key`, `auth`, `credential`, ...) are masked, and so
  are the values of secret-looking flags inside a logged command line (for
  example `--api-key …`). `Monitor` masks them before logging to MLflow, and the
  bot masks them again before showing params (for runs logged before).
- **Errors.** Handler errors reach the chat only as their type; the details go
  to the local log. Crash messages from `Monitor` mask the token, URL
  credentials, secret query values and the home directory. Pass
  `send_traceback=False` to leave the traceback out entirely.
- **Local files.** The state file and the cleanup record are created readable
  by their owner only.

### The Mini App page

- **Where chart data goes.** Chart data travels in the 📊 Chart link's fragment
  (`#d=…`), which browsers never send to the page's host. Telegram's servers do
  store it, as part of the message, just like the bot's other messages and
  pictures.
- **Untrusted links.** Anyone can open the page with a hand-made link, so the
  decoder treats the link as hostile:
  - size caps before and after inflating, so a deflate bomb stops at 4 MB;
  - a field-by-field header check, and caps on runs, series and points;
  - no reads past the end of the data;
  - prototype-free objects for names taken from the link;
  - names are never inserted as HTML.
- **Content Security Policy.** The page runs under
  `default-src 'none'; script-src 'self' https://telegram.org; connect-src 'none'`
  (plus inline styles and `data:` images). It has no inline or eval'd scripts
  and makes no network requests, so even a bug could not send data anywhere.
  It also sets `referrer: no-referrer`.
- **Scripts it loads.** ECharts is served from the page's own host and checked
  against a pinned SHA-256 when the page is built. Telegram's
  `telegram-web-app.js` comes from telegram.org, as Telegram requires.
