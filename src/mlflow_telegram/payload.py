"""Curves packed into a link's #fragment, for a static Mini App page to draw.

A web_app button can open a page on any HTTPS host (GitHub Pages, say) with the
chart's data after `#d=`. Browsers never send the fragment to the server, so the
host only ever serves the empty page; the curves travel in the button, inside
the private chat, like the PNGs do.

Encoding, made small because the link has a length limit and short because it is
built every time a chart is drawn:

- each series is downsampled with LTTB (largest triangle three buckets, which
  keeps the curve's shape) to at most N points, N chosen so the link fits;
- steps are delta-encoded integers; values are quantised to 16 bits across the
  series' own range (on log10 when the series is positive and spans more than a
  decade), then delta-encoded; all as zigzag varints;
- a small JSON header (names, keys, ranges) and the binary body are deflated
  (raw) and base64url-encoded without padding.

The format is specified in `docs/payload.md`; `page/assets/fragment.js` is the
decoder the page runs.
"""

from __future__ import annotations

import base64
import json
import math
import struct
import zlib

__all__ = ["encode", "decode", "fit", "lttb", "VERSION"]

LEVELS = 65535
VERSION = 1


def lttb(points: list[tuple[float, float]], limit: int) -> list[tuple[float, float]]:
    """Largest-triangle-three-buckets: `limit` points that keep the curve's shape."""
    if limit >= len(points) or limit < 3:
        return list(points) if limit >= len(points) else list(points[:1] + points[-1:])
    out = [points[0]]
    size = (len(points) - 2) / (limit - 2)
    a = 0
    for i in range(limit - 2):
        start, end = int(i * size) + 1, int((i + 1) * size) + 1
        nxt_start, nxt_end = end, min(int((i + 2) * size) + 1, len(points))
        nxt = points[nxt_start:nxt_end] or [points[-1]]
        ax, ay = points[a]
        cx = sum(p[0] for p in nxt) / len(nxt)
        cy = sum(p[1] for p in nxt) / len(nxt)
        best, best_area = start, -1.0
        for j in range(start, min(end, len(points) - 1)):
            bx, by = points[j]
            area = abs((ax - cx) * (by - ay) - (ax - bx) * (cy - ay))
            if area > best_area:
                best, best_area = j, area
        out.append(points[best])
        a = best
    out.append(points[-1])
    return out


def _varint(value: int, out: bytearray) -> None:
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return


def _zigzag(value: int) -> int:
    return 2 * value if value >= 0 else -2 * value - 1


def _quantise(values: list[float]) -> tuple[list[int], float, float, bool]:
    use_log = min(values) > 0 and max(values) / min(values) > 10
    scaled = [math.log10(v) for v in values] if use_log else list(values)
    low, high = min(scaled), max(scaled)
    span = (high - low) or 1.0
    return [round((v - low) / span * LEVELS) for v in scaled], low, high, use_log


def _pack(payload: dict, limit: int | None) -> bytes:
    header = {"v": VERSION, "initial": payload["initial"], "runs": []}
    body = bytearray()
    for run in payload["runs"]:
        entry = {"id": run["id"], "name": run["name"], "experiment": run.get("experiment", ""),
                 "series": []}
        for key, points in run["series"].items():
            points = [(s, v) for s, v in points if v is not None and math.isfinite(v)]
            if not points:
                continue
            cap = run.get("limit", limit) if limit is None else min(limit, run.get("limit", limit))
            if cap is not None:
                points = lttb(points, cap)
            quantised, low, high, use_log = _quantise([v for _, v in points])
            entry["series"].append({"key": key, "n": len(points), "lo": low, "hi": high,
                                    "log": use_log})
            last = 0
            for step, _ in points:
                step = int(round(step))
                _varint(_zigzag(step - last), body)
                last = step
            last = 0
            for q in quantised:
                _varint(_zigzag(q - last), body)
                last = q
        header["runs"].append(entry)
    head = json.dumps(header, separators=(",", ":")).encode()
    raw = struct.pack(">I", len(head)) + head + bytes(body)
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    return compressor.compress(raw) + compressor.flush()


def encode(payload: dict, limit: int | None = None) -> str:
    """`payload` ({"initial": {runs, keys}, "runs": [{id, name, experiment, series:
    {key: [(step, value)]}}]}) as base64url text, each series cut to `limit` points."""
    return base64.urlsafe_b64encode(_pack(payload, limit)).decode().rstrip("=")


def fit(payload: dict, budget: int, floor: int = 20) -> tuple[str, int | None]:
    """The encoding with the most points per series that stays within `budget`
    characters, and that point limit (None: every point fitted)."""
    whole = encode(payload)
    if len(whole) <= budget:
        return whole, None
    longest = max((len(points) for run in payload["runs"] for points in run["series"].values()),
                  default=0)
    low, high, best = floor, max(floor, longest), None
    while low <= high:
        middle = (low + high) // 2
        text = encode(payload, middle)
        if len(text) <= budget:
            best, low = (text, middle), middle + 1
        else:
            high = middle - 1
    if best is None:
        return encode(payload, floor), floor
    return best


def decode(text: str) -> dict:
    """The inverse of `encode` (the page does this in JavaScript; here for tests)."""
    raw = zlib.decompress(base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)), -15)
    size = struct.unpack(">I", raw[:4])[0]
    header = json.loads(raw[4:4 + size])
    if header.get("v") != VERSION:
        raise ValueError(f"payload version {header.get('v')} is not {VERSION}")
    position = 4 + size

    def varint():
        nonlocal position
        result, shift = 0, 0
        while True:
            byte = raw[position]
            position += 1
            result |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                return result

    def signed():
        value = varint()
        return -(value + 1) // 2 if value & 1 else value // 2

    out = {"initial": header["initial"], "runs": []}
    for run in header["runs"]:
        series = {}
        for entry in run["series"]:
            steps, last = [], 0
            for _ in range(entry["n"]):
                last += signed()
                steps.append(last)
            values, last = [], 0
            span = (entry["hi"] - entry["lo"]) or 1.0
            for _ in range(entry["n"]):
                last += signed()
                value = entry["lo"] + last / LEVELS * span
                values.append(10 ** value if entry["log"] else value)
            series[entry["key"]] = list(zip(steps, values))
        out["runs"].append({"id": run["id"], "name": run["name"],
                            "experiment": run["experiment"], "series": series})
    return out
