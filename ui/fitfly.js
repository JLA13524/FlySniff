// ---------------- FIT decoding (in the browser) ----------------
// Mirrors flysniff/fit.py. Only the messages we need: record (20), session (18), sport (12).
const FIT_EPOCH = 631065600;
const SPORT_ENUM = { 0: 'Workout', 1: 'Run', 2: 'Ride', 4: 'Workout', 5: 'Swim', 10: 'Workout', 11: 'Walk',
  12: 'Ski', 15: 'Row', 17: 'Hike', 21: 'Ride' };
const BASE = [ // [size, invalid, reader]
  [1, 0xFF, 'Uint8'], [1, 0x7F, 'Int8'], [1, 0xFF, 'Uint8'], [2, 0x7FFF, 'Int16'], [2, 0xFFFF, 'Uint16'],
  [4, 0x7FFFFFFF, 'Int32'], [4, 0xFFFFFFFF, 'Uint32'], [1, null, null], [4, null, 'Float32'], [8, null, 'Float64'],
  [1, 0, 'Uint8'], [2, 0, 'Uint16'], [4, 0, 'Uint32'],
];

async function fileBytes(file) {
  let buf = await file.arrayBuffer();
  const b = new Uint8Array(buf, 0, 2);
  if (b[0] === 0x1f && b[1] === 0x8b) {
    if (typeof DecompressionStream === 'undefined') throw new Error('this browser can’t unzip .fit.gz – unzip it first');
    buf = await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  }
  return buf;
}

function parseFit(buf) {
  const dv = new DataView(buf);
  const out = { session: null, sport: null, records: [] };
  let p = 0;
  while (p + 12 <= dv.byteLength) {
    const hs = dv.getUint8(p), size = dv.getUint32(p + 4, true);
    const sig = String.fromCharCode(dv.getUint8(p + 8), dv.getUint8(p + 9), dv.getUint8(p + 10), dv.getUint8(p + 11));
    if (sig !== '.FIT' || hs < 12) { if (p === 0) throw new Error('not a FIT file'); break; }
    let q = p + hs; const end = Math.min(q + size, dv.byteLength);
    const defs = {}; let lastTs = 0;
    const read = (def, ts) => {
      const o = {};
      for (const f of def.fields) {
        const bt = BASE[f.bt & 0x1F];
        if (bt && bt[2] && f.size === bt[0]) {
          const v = dv['get' + bt[2]](q, def.little);
          if (bt[1] === null ? Number.isFinite(v) : v !== bt[1]) o[f.num] = v;
        }
        q += f.size;
      }
      q += def.devSize;
      if (ts != null) o[253] = ts;
      if (o[253] != null) lastTs = o[253];
      return o;
    };
    while (q < end) {
      const h = dv.getUint8(q++);
      let local, msg = null, def;
      if (h & 0x80) {                                   // compressed-timestamp data message
        local = (h >> 5) & 3; const off = h & 0x1F;
        let ts = lastTs - (lastTs % 32) + off; if (off < lastTs % 32) ts += 32;
        def = defs[local]; if (!def) throw new Error('corrupt FIT (no definition)');
        msg = read(def, ts);
      } else if (h & 0x40) {                            // definition message
        local = h & 0x0F; const dev = h & 0x20;
        q++; const little = dv.getUint8(q++) === 0;
        const gmn = dv.getUint16(q, little); q += 2;
        const n = dv.getUint8(q++); const fields = [];
        for (let i = 0; i < n; i++, q += 3) fields.push({ num: dv.getUint8(q), size: dv.getUint8(q + 1), bt: dv.getUint8(q + 2) });
        let devSize = 0;
        if (dev) { const nd = dv.getUint8(q++); for (let i = 0; i < nd; i++, q += 3) devSize += dv.getUint8(q + 1); }
        defs[local] = { gmn, little, fields, devSize };
        continue;
      } else {
        local = h & 0x0F; def = defs[local]; if (!def) throw new Error('corrupt FIT (no definition)');
        msg = read(def);
      }
      if (def.gmn === 20) out.records.push(msg);
      else if (def.gmn === 18 && !out.session) out.session = msg;
      else if (def.gmn === 12 && out.sport == null && msg[0] != null) out.sport = msg[0];
    }
    p = end + 2;
  }
  return out;
}

// ---------------- stream metrics (mirror of fit.stream_metrics) ----------------
const isF = Number.isFinite;
const meanOf = a => { const f = a.filter(isF); return f.length ? f.reduce((s, x) => s + x, 0) / f.length : NaN; };
const maxOf = a => { const f = a.filter(isF); return f.length ? Math.max(...f) : NaN; };
function rolling(a, w) { const o = []; let s = 0; for (let i = 0; i < a.length; i++) { s += a[i]; if (i >= w) s -= a[i - w]; o.push(s / Math.min(i + 1, w)); } return o; }
function interp(a) {
  const idx = a.map((v, i) => isF(v) ? i : -1).filter(i => i >= 0); if (!idx.length) return a;
  return a.map((v, i) => {
    if (isF(v)) return v;
    let k = idx.findIndex(j => j > i);
    if (k === -1) return a[idx[idx.length - 1]]; if (k === 0) return a[idx[0]];
    const i0 = idx[k - 1], i1 = idx[k]; return a[i0] + (a[i1] - a[i0]) * (i - i0) / (i1 - i0);
  });
}
function streamMetrics(hr, v, alt) {
  const out = { drift_pct: NaN, ascent_m: NaN, speed_cv: NaN };
  if (alt.length > 10 && alt.filter(isF).length > 10) {
    const a = rolling(interp(alt), 10); let up = 0;
    for (let i = 1; i < a.length; i++) if (a[i] > a[i - 1]) up += a[i] - a[i - 1];
    out.ascent_m = up;
  }
  const moving = v.map(x => isF(x) && x > 0.5);
  if (moving.filter(Boolean).length >= 60) {
    const vm = rolling(v.filter((_, i) => moving[i]), 30), mu = meanOf(vm);
    const sd = Math.sqrt(meanOf(vm.map(x => (x - mu) ** 2)));
    out.speed_cv = mu > 0 ? sd / mu : NaN;
    const ok = v.map((x, i) => moving[i] && isF(hr[i]) && hr[i] > 40);
    const vv = v.filter((_, i) => ok[i]), hh = hr.filter((_, i) => ok[i]);
    if (vv.length >= 120) {
      const h = Math.floor(vv.length / 2);
      const e1 = meanOf(vv.slice(0, h)) / meanOf(hh.slice(0, h)), e2 = meanOf(vv.slice(h)) / meanOf(hh.slice(h));
      out.drift_pct = (e1 - e2) / e1 * 100;
    }
  }
  return out;
}

const INTERVAL_WORDS = /interval|reps|vo2|threshold|x\s*\d|\d+\s*x|track|fartlek|tempo|sweet ?spot/i;
function fitToActivity(fileName, f) {
  const s = f.session || {}, recs = f.records;
  const g = (r, ...ks) => { for (const k of ks) if (r[k] != null) return r[k]; return null; };
  const hr = recs.map(r => r[3] ?? NaN);
  const v = recs.map(r => { const x = g(r, 73, 6); return x == null ? NaN : x / 1000; });
  const alt = recs.map(r => { const x = g(r, 78, 2); return x == null ? NaN : x / 5 - 500; });
  const cad = recs.map(r => r[4] ?? NaN);
  const dist = recs.map(r => r[5] == null ? NaN : r[5] / 100);
  const ts = recs.map(r => r[253]).filter(x => x != null);
  if (!recs.length && !f.session) throw new Error('no activity data in file');
  const sm = streamMetrics(hr, v, alt);
  const start = s[2] ?? ts[0];
  const date = start != null ? new Date((start + FIT_EPOCH) * 1000) : null;
  const moving = s[8] != null ? s[8] / 1000 : (ts.length || NaN);
  const distance = s[9] != null ? s[9] / 100 : maxOf(dist);
  const sportNum = s[5] ?? f.sport ?? 0;
  const type = SPORT_ENUM[sportNum] || 'Workout';
  const speed = (s[124] ?? s[14]) != null ? (s[124] ?? s[14]) / 1000 : (moving ? distance / moving : meanOf(v));
  const stem = fileName.replace(/\.fit(\.gz)?$/i, '').replace(/\.gz$/i, '');
  const name = /^\d+$/.test(stem)
    ? [type, date ? date.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' }) : '', isF(distance) ? (distance / 1000).toFixed(1) + ' km' : ''].filter(Boolean).join(' · ')
    : stem.replace(/[_-]/g, ' ');
  const elev = s[22] != null ? s[22] : sm.ascent_m;
  const avg_hr = s[16] ?? meanOf(hr), max_hr = s[17] ?? maxOf(hr), cadence = s[18] ?? meanOf(cad);
  return {
    id: 'fit:' + stem, name, type, date: date ? date.toISOString() : null,
    distance_m: distance, moving_s: moving, elev_m: elev, avg_hr, max_hr,
    raw: {
      type, speed_ms: speed, avg_hr, max_hr, cadence, drift_pct: sm.drift_pct,
      climb: elev / Math.max(distance / 1000, 1), named: INTERVAL_WORDS.test(name) ? 1 : 0, speed_cv: sm.speed_cv,
    },
  };
}

// ---------------- features + fly (mirror of strava.features_from_raw and fly.Fly) ----------------
const FEATURES = ['speed', 'easy', 'cadence', 'hr_drift', 'climb', 'intervalness'];
const SMELLS = {
  speed: ['DM1', 'smells like a PB'], easy: ['DM4', 'smells like a lovely recovery jog'],
  cadence: ['VM2', 'smells nicely ripe'], hr_drift: ['V', 'smells like panic (CO2)'],
  climb: ['DA2', 'smells like a damp hillside, literally'], intervalness: ['DA1', 'smells like a rival male'],
};
const INNATE = { DM1: 1.0, DM4: 0.9, VM2: 0.6, V: -1.0, DA2: -1.3, DA1: -0.8 };
const num = x => (x == null || x === '' ? NaN : +x);

function zs(xs) {                                    // pandas: nan-skipping mean, ddof=1 std
  const f = xs.filter(isF); const m = meanOf(xs);
  const sd = f.length > 1 ? Math.sqrt(f.reduce((s, x) => s + (x - m) ** 2, 0) / (f.length - 1)) : NaN;
  return xs.map(x => (sd > 0 ? (x - m) / sd : x * 0));
}
function zsBy(xs, groups) {
  const out = new Array(xs.length).fill(NaN);
  for (const g of new Set(groups)) {
    const idx = groups.map((t, i) => t === g ? i : -1).filter(i => i >= 0);
    zs(idx.map(i => xs[i])).forEach((z, k) => { out[idx[k]] = z; });
  }
  return out;
}
function featuresFromRaw(raws) {
  const col = k => raws.map(r => num(r[k]));
  const type = raws.map(r => r.type || '');
  const maxHr = col('max_hr'), avgHr = col('avg_hr');
  const hrmax = maxHr.some(isF) ? maxOf(maxHr) : 190;
  const proxy = avgHr.map((a, i) => (maxHr[i] - a) / a * 100), pm = meanOf(proxy);
  const drift = col('drift_pct').map((d, i) => isF(d) ? d : proxy[i] - pm + 3);
  const fz = x => (isF(x) ? Math.max(-3, Math.min(3, x)) : 0);
  const speed = zsBy(col('speed_ms'), type), easy = zs(avgHr.map(a => a / hrmax)).map(x => -x);
  const cadence = zsBy(col('cadence'), type), hrd = zs(drift), climb = zs(col('climb'));
  const spiky = zs(maxHr.map((m, i) => m - avgHr[i])).map(x => isF(x) ? x : 0);
  const cv = zs(col('speed_cv')).map(x => isF(x) ? x : 0);
  const named = col('named').map(x => isF(x) ? x : 0);
  return raws.map((_, i) => ({
    speed: fz(speed[i]), easy: fz(easy[i]), cadence: fz(cadence[i]), hr_drift: fz(hrd[i]), climb: fz(climb[i]),
    intervalness: fz(1.5 * named[i] + 0.3 * spiky[i] + 0.6 * cv[i]),
  }));
}

function makeFly(c) {
  const nk = c.n_kc, G = c.glomeruli;
  const W = Array.from({ length: c.mbon_names.length }, () => []);  // per MBON: [kc, w] pairs
  c.kc_mbon[0].forEach((k, i) => W[c.kc_mbon[1][i]].push([k, c.kc_mbon[2][i]]));
  const absSign = Math.max(c.mbon_sign.reduce((s, x) => s + Math.abs(x), 0), 1);
  function drive(f) { const d = G.map(() => 0); for (const ft of FEATURES) d[G.indexOf(SMELLS[ft][0])] = 1 / (1 + Math.exp(-2 * (f[ft] - 0.3))); return d; }
  function kc(d) {
    const x = new Float64Array(nk);
    d.forEach((dv, g) => { if (dv) { const row = c.g_kc[g]; for (let k = 0; k < nk; k++) x[k] += dv * row[k]; } });
    let m = 0; for (let k = 0; k < nk; k++) m += x[k]; m /= nk;
    for (let k = 0; k < nk; k++) x[k] /= c.apl_gain[k] * (1 + 0.2 * m);
    const s = Array.from(x).sort((a, b) => a - b), pos = (nk - 1) * 0.95, lo = Math.floor(pos);
    const thr = s[lo] + (s[Math.min(lo + 1, nk - 1)] - s[lo]) * (pos - lo), mx = s[nk - 1];
    for (let k = 0; k < nk; k++) x[k] = Math.max(x[k] - thr, 0) / (mx - thr + 1e-9);
    return x;
  }
  const mbon = k => W.map(ws => ws.reduce((s, [i, w]) => s + k[i] * w, 0));
  return { drive, kc, mbon, absSign };
}

const VERDICT = v => v > .55 ? 'OBSESSED' : v > .2 ? 'LOVES' : v > .05 ? 'into it' : v > -.05 ? 'indifferent' : v > -.2 ? 'not a fan' : v > -.55 ? 'HATES' : 'FLED THE ROOM';
const PHRASE = { OBSESSED: 'is obsessed with', LOVES: 'loves', 'into it': 'is into', 'not a fan': 'is not a fan of', HATES: 'hates' };
function topSmell(contrib, v) { const s = v >= 0 ? 1 : -1; return FEATURES.reduce((b, f) => s * contrib[f] > s * contrib[b] ? f : b, FEATURES[0]); }
function caption(a, v, contrib) {
  const feat = topSmell(contrib, v), smell = SMELLS[feat][1];
  const day = a.date ? new Date(a.date).toLocaleDateString('en-GB', { weekday: 'long' }) : '';
  const nm = a.name || a.type, low = nm.toLowerCase();
  const subj = (low.includes('interval') || low.includes('vo2')) ? (day ? `your ${day} intervals` : 'your intervals') : `“${nm}”`;
  const verb = VERDICT(v);
  if (verb === 'indifferent') return `the fly sniffed ${subj} and felt nothing.`;
  if (contrib[feat] * v <= 0) return `the fly ${PHRASE[verb] || 'is fleeing'} ${subj}. no idea why – it's a mushroom body thing.`;
  if (verb === 'FLED THE ROOM') return `the fly smelled ${subj} and fled the room. it ${smell}.`;
  return `the fly ${PHRASE[verb]} ${subj}. it ${smell}.`;
}

// Score `items` (new uploads) against a reference pool of raw metrics, the way flysniff does a batch.
function scoreAgainst(items, poolRaws, circuit) {
  const fly = makeFly(circuit), G = circuit.glomeruli;
  const raws = [...items.map(a => a.raw), ...poolRaws];
  const F = featuresFromRaw(raws), D = F.map(fly.drive), K = D.map(fly.kc), M = K.map(fly.mbon);
  const nm = M[0].length, mu = [], sd = [];
  for (let j = 0; j < nm; j++) { const col = M.map(r => r[j]); const m = meanOf(col); mu.push(m); sd.push(Math.sqrt(meanOf(col.map(x => (x - m) ** 2)))); }
  const sampleIdx = Array.from({ length: 400 }, (_, i) => Math.floor(i * (circuit.n_kc - 1) / 399));
  return items.map((a, i) => {
    const d = D[i], mz = M[i].map((x, j) => (x - mu[j]) / (sd[j] + 1e-9));
    const mb = Math.tanh(mz.reduce((s, z, j) => s + z * circuit.mbon_sign[j], 0) / fly.absSign * 2);
    const innate = Math.tanh(1.5 * G.reduce((s, g, j) => s + (d[j] - 0.3) * INNATE[g], 0));
    const valence = 0.6 * innate + 0.4 * mb;
    const contrib = Object.fromEntries(FEATURES.map(f => { const g = SMELLS[f][0]; return [f, Math.max(d[G.indexOf(g)] - 0.35, 0) * INNATE[g]]; }));
    const merged = {}; circuit.mbon_names.forEach((n, j) => (merged[n] = merged[n] || []).push(mz[j]));
    const r2 = x => isF(x) ? Math.round(x * 10) / 10 : null;
    return {
      id: a.id, name: a.name, type: a.type, date: a.date,
      weekday: a.date ? new Date(a.date).toLocaleDateString('en-GB', { weekday: 'long' }) : null,
      distance_km: r2(a.distance_m / 1000), moving_min: r2(a.moving_s / 60), elev_m: r2(a.elev_m), avg_hr: r2(a.avg_hr), kudos: null,
      features: F[i], glomeruli: Object.fromEntries(G.map((g, j) => [g, d[j]])),
      kc_active: K[i].reduce((s, x) => s + (x > 0), 0), kc_pattern: sampleIdx.map((k, j) => K[i][k] > 0 ? j : -1).filter(j => j >= 0),
      mbon: Object.fromEntries(Object.entries(merged).sort().map(([n, v]) => [n, meanOf(v)])),
      innate, learned: mb, valence, verdict: VERDICT(valence), caption: caption(a, valence, contrib),
      top_smell: topSmell(contrib, valence), raw: a.raw, uploaded: true,
    };
  });
}
