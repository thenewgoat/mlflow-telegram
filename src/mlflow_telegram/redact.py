"""Keeping secrets out of what is logged, shown in Telegram or printed.

- `is_secret_key`: whether a param/tag name looks like it holds a secret.
- `redact_params`: a params dict with such values masked, including the values
  of secret-looking flags inside a logged command line.
- `redact_text`: free text (an error, a traceback) with given secrets (the bot
  token), `user:password@` in URLs, `?token=`-style query values and the home
  directory replaced.
"""

from __future__ import annotations

import os
import re

__all__ = ["MASK", "is_secret_key", "redact_params", "redact_command", "redact_text"]

MASK = "***"
SECRET_KEY = re.compile(r"(token|secret|passw|pwd|api[_-]?key|apikey|auth|credential|"
                        r"private[_-]?key|access[_-]?key|session[_-]?key|cookie)", re.I)
_FLAG = re.compile(r"(--?[\w-]+)(=|\s+)(\"[^\"]*\"|'[^']*'|\S+)")
_USERINFO = re.compile(r"(\b[a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@", re.I)
_QUERY = re.compile(r"([?&](?:[\w-]*(?:token|secret|key|password|auth|sig)[\w-]*)=)[^&\s#]+",
                    re.I)
_BOT_TOKEN = re.compile(r"\bbot\d{5,}:[A-Za-z0-9_-]{20,}")


def is_secret_key(key: str) -> bool:
    return bool(SECRET_KEY.search(key or ""))


def redact_command(command: str) -> str:
    """`--api-key abc --lr 3e-4` → `--api-key *** --lr 3e-4`."""
    def mask(match):
        flag, joiner, _ = match.groups()
        return f"{flag}{joiner}{MASK}" if is_secret_key(flag) else match.group(0)
    return _FLAG.sub(mask, command or "")


def redact_params(params: dict) -> dict:
    """A copy with secret-looking keys masked and secret flags masked in any
    value that looks like a command line."""
    out = {}
    for key, value in params.items():
        if is_secret_key(str(key)):
            out[key] = MASK
        elif isinstance(value, str) and "--" in value:
            out[key] = redact_command(value)
        else:
            out[key] = value
    return out


def redact_text(text: str, secrets: tuple[str, ...] | list[str] = (), home: bool = True) -> str:
    """`text` with each of `secrets`, bot tokens, URL credentials and secret query
    values masked, and the home directory shown as `~`."""
    if not text:
        return text
    for secret in secrets:
        if secret:
            text = text.replace(secret, MASK)
    text = _BOT_TOKEN.sub("bot" + MASK, text)
    text = _USERINFO.sub(lambda m: m.group(1) + MASK + "@", text)
    text = _QUERY.sub(lambda m: m.group(1) + MASK, text)
    if home:
        user_home = os.path.expanduser("~")
        if user_home and user_home not in ("/", "~"):
            text = text.replace(user_home, "~")
    return text
