# Chart link payload, version 1

The bot opens the Mini App page with a chart's curves packed into the link's
fragment:

```
https://<your page>/#d=<payload>
```

Browsers never send the fragment (everything after `#`) to the server, so the
page's host only ever serves the same empty page. Telegram appends its own
parameters to the fragment (`&tgWebAppData=...`); the page reads `d` and
ignores the rest.

`mlflow_telegram.payload.encode` / `decode` implement this in Python;
`src/mlflow_telegram/page/assets/fragment.js` is the browser decoder.

## Layout

`<payload>` is **base64url without padding** of a **raw DEFLATE** stream
(RFC 1951, no zlib or gzip header). Inflated, it is:

| Bytes | Content |
|---|---|
| 4 | `H`, the header length, big-endian unsigned |
| H | the header, UTF-8 JSON |
| rest | the body: every series' steps then values, in header order |

### Header

```json
{
  "v": 1,
  "initial": {"runs": ["<run id>", "..."], "keys": ["train_loss", "eval/validation"]},
  "runs": [
    {"id": "<run id>", "name": "lr-3e-4", "experiment": "my-experiment",
     "series": [
       {"key": "train_loss", "n": 412, "lo": 0.21, "hi": 3.9, "log": true},
       {"key": "eval/validation", "n": 40, "lo": 0.35, "hi": 2.7, "log": false}
     ]}
  ]
}
```

- `initial`: the runs and metrics drawn when the page opens. Any other run or
  series in the payload is offered as a toggle.
- `n`: the number of points in the series.
- `lo`, `hi`, `log`: how the values were quantised (below).

### Body

For each run, then each series, in header order:

1. `n` steps: zigzag varints of the differences between consecutive integer
   steps. The first difference is taken from 0.
2. `n` values: zigzag varints of the differences between consecutive quantised
   values. The first difference is taken from 0.

A **varint** is little-endian base-128: 7 bits per byte, with the high bit set
on every byte except the last. **Zigzag** maps a signed integer to an unsigned
one: `0, -1, 1, -2, …` become `0, 1, 2, 3, …`.

### Quantisation

Each series is quantised on its own:

- If every value is positive and `max / min > 10`, the series is quantised on
  `log10(value)` and `log` is `true`. Otherwise it is quantised on the values
  themselves.
- `lo` and `hi` are the minimum and maximum of the (possibly log) values.
- `q = round((x - lo) / (hi - lo) × 65535)`, and back again
  `x = lo + q / 65535 × (hi - lo)`. If `hi == lo`, every value is `lo`.

The worst-case error is half a step of `(hi - lo) / 65535`, about 0.0008% of
the series' range.

## Fitting a length limit

Telegram does not document how long a button link can be. Measure it with
`mlflow-telegram probe`. `payload.fit(payload, budget)` finds the largest
per-series point count whose encoding fits the budget. Series are thinned with
LTTB (largest-triangle-three-buckets), which keeps a curve's shape and its
spikes.

## Versioning

`v` changes whenever the layout changes. Decoders must refuse versions they do
not know.
