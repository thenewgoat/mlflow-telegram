"""Sending to Telegram with nothing but the standard library: text and photos,
in order, from a background thread, cut to Telegram's limits. Used by `Monitor`
so a training process needs no Telegram library.
"""

import queue
import re
import sys
import threading
import urllib.parse
import urllib.request
import uuid

from mlflow_telegram.redact import redact_text

__all__ = ["Telegram", "fit", "TEXT_LIMIT", "CAPTION_LIMIT"]

TEXT_LIMIT = 4096
CAPTION_LIMIT = 1024


def fit(text, limit: int):
    """`text` cut to `limit` characters at a line break, ending in "…", with any
    HTML tag the cut left open closed again. None stays None."""
    if text is None or len(text) <= limit:
        return text
    room = limit - 20
    cut = text.rfind("\n", 0, room)
    if cut < room // 2:
        cut = room
    head = text[:cut]
    if head.count("<") > head.count(">"):        # a tag cut in half
        head = head[:head.rfind("<")]
    closing = "".join(f"</{tag}>" for tag in ("code", "pre", "i", "b")
                      if head.count(f"<{tag}>") > head.count(f"</{tag}>"))
    return head + closing + "\n…"


class Telegram:
    """Sends messages (and charts) to one chat from a background thread, in order."""

    def __init__(self, token: str, chat_id: str, timeout: float = 10.0):
        token = token.strip()
        if not re.fullmatch(r"\d+:[A-Za-z0-9_-]+", token):
            raise ValueError("the bot token has an unexpected shape")
        self.token = token
        self.base = f"https://api.telegram.org/bot{token}"
        self.url = f"{self.base}/sendMessage"
        self.chat_id = chat_id
        self.timeout = timeout
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._work, daemon=True, name="telegram")
        self.thread.start()

    def send(self, text: str) -> None:
        self.queue.put(("text", text))

    def send_photo(self, png: bytes, caption: str) -> None:
        """A chart with a caption; the caption alone if the photo cannot be sent."""
        self.queue.put(("photo", (png, caption)))

    def close(self, wait: float = 15.0) -> None:
        """Sends what is queued, waiting at most `wait` seconds for it."""
        self.queue.put(None)
        self.thread.join(wait)

    def _work(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            kind, payload = item
            if kind == "photo":
                png, caption = payload
                if self._post_photo(png, caption):
                    continue
                payload = caption
            self._post_text(payload)

    def _post_text(self, text: str) -> None:
        data = urllib.parse.urlencode({"chat_id": self.chat_id, "text": fit(text, TEXT_LIMIT),
                                       "disable_web_page_preview": "true"}).encode()
        try:
            with urllib.request.urlopen(self.url, data=data, timeout=self.timeout) as reply:
                reply.read()
        except Exception as error:  # noqa: BLE001 -- a message is never worth a run
            # The token is part of the URL; some errors quote it, so it is masked.
            print(f"[mlflow-telegram] telegram send failed: {type(error).__name__}: "
                  f"{redact_text(str(error), [self.token])}", file=sys.stderr, flush=True)

    def _post_photo(self, png: bytes, caption: str) -> bool:
        boundary = uuid.uuid4().hex
        parts = []
        for name, value in (("chat_id", self.chat_id), ("caption", fit(caption, CAPTION_LIMIT))):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
                         f"\r\n\r\n{value}\r\n".encode())
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; '
                     f'filename="curves.png"\r\nContent-Type: image/png\r\n\r\n'.encode())
        parts.append(png + f"\r\n--{boundary}--\r\n".encode())
        request = urllib.request.Request(
            f"{self.base}/sendPhoto", data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout * 3) as reply:
                reply.read()
            return True
        except Exception as error:  # noqa: BLE001
            print(f"[mlflow-telegram] telegram photo failed, sending text: "
                  f"{type(error).__name__}: {redact_text(str(error), [self.token])}",
                  file=sys.stderr, flush=True)
            return False
