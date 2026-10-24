// The static Mini App's data: decoded from the link's #d=... (mlflow_telegram/payload.py
// writes it). No network: the curves came inside the Telegram button.
const tg = window.Telegram && window.Telegram.WebApp;

function fragmentParam(name) {
  // Saved before telegram-web-app.js ran; Telegram adds its own params after ours.
  const hash = (window.__FRAGMENT || location.hash).replace(/^#/, "");
  return new URLSearchParams(hash).get(name);
}

async function decodeCurves(text) {
  const b64 = text.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((text.length + 3) % 4);
  const packed = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const stream = new Blob([packed]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  const raw = new Uint8Array(await new Response(stream).arrayBuffer());
  const size = new DataView(raw.buffer).getUint32(0);
  const header = JSON.parse(new TextDecoder().decode(raw.subarray(4, 4 + size)));
  if (header.v !== 1) throw new Error("unknown payload version " + header.v);
  let position = 4 + size;
  const varint = () => {
    let result = 0, scale = 1, byte;
    do { byte = raw[position++]; result += (byte & 0x7f) * scale; scale *= 128; } while (byte & 0x80);
    return result;
  };
  const signed = () => { const u = varint(); return u % 2 ? -(u + 1) / 2 : u / 2; };
  const runs = {};
  const catalogue = header.runs.map(run => {
    const series = {};
    run.series.forEach(entry => {
      const steps = [];
      let last = 0;
      for (let i = 0; i < entry.n; i++) { last += signed(); steps.push(last); }
      const span = (entry.hi - entry.lo) || 1;
      const points = [];
      last = 0;
      for (let i = 0; i < entry.n; i++) {
        last += signed();
        const value = entry.lo + (last / 65535) * span;
        points.push([steps[i], entry.log ? Math.pow(10, value) : value]);
      }
      series[entry.key] = points;
    });
    runs[run.id] = { name: run.name, series };
    return { id: run.id, name: run.name, experiment: run.experiment, keys: Object.keys(series) };
  });
  return { initial: header.initial, catalogue, runs };
}

const DECODED = decodeCurves(fragmentParam("d") || "").catch(() => {
  throw new Error("This chart link is incomplete. Open it again from the bot.");
});
const INITIAL = { runs: [], keys: [] };   // filled from the link before the chart starts

const SOURCE = {
  catalogue: async () => {
    const data = await DECODED;
    INITIAL.runs.push(...data.initial.runs);
    INITIAL.keys.push(...data.initial.keys);
    return data.catalogue;
  },
  series: async (runs, keys) => {
    const data = await DECODED;
    return {
      runs: runs.filter(id => data.runs[id]).map(id => {
        const run = data.runs[id];
        const series = {};
        keys.forEach(key => { if (run.series[key]) series[key] = run.series[key]; });
        return { id, name: run.name, series };
      }),
    };
  },
};
