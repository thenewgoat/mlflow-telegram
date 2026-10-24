// The Mini App's data: decoded from the link's #d=... (mlflow_telegram/payload.py
// writes it; docs/payload.md is the format). No network: the curves came inside
// the Telegram button.
//
// Anyone can make such a link, so everything here is treated as hostile input:
// sizes are capped before and after inflating, the header is checked field by
// field, the body is never read past its end, and run ids and metric keys only
// ever become keys of prototype-free objects. Nothing decoded is used as HTML.
"use strict";
const tg = window.Telegram && window.Telegram.WebApp;

const LIMITS = {
  text: 200000,          // characters of #d=
  inflated: 4 << 20,     // bytes after inflating (a deflate bomb stops here)
  runs: 50,
  series: 100,           // per run
  points: 200000,        // per series
  total: 2000000,        // points in all
  name: 200,             // characters in a run name, id or metric key
};

function fragmentParam(name) {
  // Saved by boot.js before telegram-web-app.js ran; Telegram adds its own params after ours.
  const hash = (window.__FRAGMENT || location.hash).replace(/^#/, "");
  return new URLSearchParams(hash).get(name);
}

function bad(why) { throw new Error("This chart link is damaged (" + why + "). Open it again from the bot."); }

async function inflate(packed) {
  const reader = new Blob([packed]).stream().pipeThrough(new DecompressionStream("deflate-raw"))
    .getReader();
  const chunks = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.length;
    if (size > LIMITS.inflated) { reader.cancel(); bad("too large"); }
    chunks.push(value);
  }
  const raw = new Uint8Array(size);
  let at = 0;
  chunks.forEach(chunk => { raw.set(chunk, at); at += chunk.length; });
  return raw;
}

const isName = v => typeof v === "string" && v.length > 0 && v.length <= LIMITS.name;
const isNumber = v => typeof v === "number" && Number.isFinite(v);

function checkHeader(header) {
  if (!header || typeof header !== "object" || header.v !== 1) bad("unknown version");
  const initial = header.initial || {};
  if (!Array.isArray(initial.runs) || !initial.runs.every(isName)) bad("runs");
  if (!Array.isArray(initial.keys) || !initial.keys.every(isName)) bad("metrics");
  if (!Array.isArray(header.runs) || header.runs.length > LIMITS.runs) bad("run list");
  let total = 0;
  header.runs.forEach(run => {
    if (!run || !isName(run.id) || !isName(run.name)) bad("run");
    if (run.experiment !== undefined && typeof run.experiment !== "string") bad("experiment");
    if (!Array.isArray(run.series) || run.series.length > LIMITS.series) bad("series list");
    run.series.forEach(entry => {
      if (!entry || !isName(entry.key) || !Number.isInteger(entry.n) || entry.n < 0 ||
          entry.n > LIMITS.points || !isNumber(entry.lo) || !isNumber(entry.hi) ||
          typeof entry.log !== "boolean") bad("series");
      total += entry.n;
    });
  });
  if (total > LIMITS.total) bad("too many points");
}

async function decodeCurves(text) {
  if (!text || text.length > LIMITS.text || !/^[A-Za-z0-9_-]+$/.test(text)) bad("encoding");
  const b64 = text.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((text.length + 3) % 4);
  let packed;
  try { packed = Uint8Array.from(atob(b64), c => c.charCodeAt(0)); } catch (e) { bad("encoding"); }
  const raw = await inflate(packed);
  if (raw.length < 4) bad("truncated");
  const size = new DataView(raw.buffer).getUint32(0);
  if (4 + size > raw.length) bad("truncated");
  let header;
  try { header = JSON.parse(new TextDecoder().decode(raw.subarray(4, 4 + size))); } catch (e) { bad("header"); }
  checkHeader(header);
  let position = 4 + size;
  const varint = () => {
    let result = 0, scale = 1, byte;
    for (let i = 0; ; i++) {
      if (position >= raw.length || i > 7) bad("truncated");
      byte = raw[position++];
      result += (byte & 0x7f) * scale;
      scale *= 128;
      if (!(byte & 0x80)) return result;
    }
  };
  const signed = () => { const u = varint(); return u % 2 ? -(u + 1) / 2 : u / 2; };
  const runs = Object.create(null);
  const catalogue = header.runs.map(run => {
    const series = Object.create(null);
    run.series.forEach(entry => {
      const steps = new Array(entry.n);
      let last = 0;
      for (let i = 0; i < entry.n; i++) { last += signed(); steps[i] = last; }
      const span = (entry.hi - entry.lo) || 1;
      const points = new Array(entry.n);
      last = 0;
      for (let i = 0; i < entry.n; i++) {
        last += signed();
        const value = entry.lo + (last / 65535) * span;
        points[i] = [steps[i], entry.log ? Math.pow(10, value) : value];
      }
      series[entry.key] = points;
    });
    runs[run.id] = { name: run.name, series };
    return { id: run.id, name: run.name, experiment: run.experiment || "", keys: Object.keys(series) };
  });
  return { initial: header.initial, catalogue, runs };
}

const DECODED = decodeCurves(fragmentParam("d") || "");
const INITIAL = { runs: [], keys: [] };   // filled from the link before the chart starts

const SOURCE = {
  catalogue: async () => {
    const data = await DECODED;
    INITIAL.runs.push(...data.initial.runs.filter(id => data.runs[id]));
    INITIAL.keys.push(...data.initial.keys);
    return data.catalogue;
  },
  series: async (runs, keys) => {
    const data = await DECODED;
    return {
      runs: runs.filter(id => data.runs[id]).map(id => {
        const run = data.runs[id];
        const series = Object.create(null);
        keys.forEach(key => { if (run.series[key]) series[key] = run.series[key]; });
        return { id, name: run.name, series };
      }),
    };
  },
};
