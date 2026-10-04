// ---------------- 3D brain view ----------------
// Real neuron skeletons from MaleCNS v1.0 (via neuPrint), drawn with three.js.
// Each session's computed response is replayed as a wave: PNs (by glomerulus drive)
// -> the Kenyon cells that actually fired -> APL -> MBONs (by output).
// The timing is illustrative; which neurons light, and how strongly, is the model's.
function initBrain(BRAIN, opts) {
  const host = document.getElementById('brain'), cv = document.getElementById('bcv');
  if (!BRAIN || !window.THREE) return null;
  const T = window.THREE;
  const renderer = new T.WebGLRenderer({ canvas: cv, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  const scene = new T.Scene(); scene.background = new T.Color(opts.bg);
  const camera = new T.PerspectiveCamera(35, 1, 1, 20000);
  const root = new T.Group(); scene.add(root);

  // centre + flip: neuPrint y points down, z is depth
  const sh = BRAIN.neurons.flatMap(n => n.p);   // frame the circuit; the brain shell sits around it
  let mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
  for (let i = 0; i < sh.length; i += 3) for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], sh[i + k]); mx[k] = Math.max(mx[k], sh[i + k]); }
  const c = mn.map((v, k) => (v + mx[k]) / 2), span = Math.max(mx[0] - mn[0], mx[1] - mn[1]);
  const X = (x, y, z) => [x - c[0], -(y - c[1]), -(z - c[2])];

  const col = s => new T.Color(s);
  const COLORS = { pn: col(opts.pn), kc: col(opts.kc), mbon_ap: col(opts.approach), mbon_av: col(opts.avoid), apl: col(opts.apl), shell: col(opts.shell) };

  function cloud(points, color, size, opacity) {
    const pos = new Float32Array(points.length);
    for (let i = 0; i < points.length; i += 3) pos.set(X(points[i], points[i + 1], points[i + 2]), i);
    const g = new T.BufferGeometry(); g.setAttribute('position', new T.BufferAttribute(pos, 3));
    const m = new T.PointsMaterial({ color, size, sizeAttenuation: true, transparent: true, opacity, depthWrite: false, blending: T.AdditiveBlending });
    const p = new T.Points(g, m); root.add(p); return p;
  }
  Object.values(BRAIN.shell).forEach(p => p && cloud(p, COLORS.shell, 4, 0.45));
  const NP_COL = { AL: opts.pn, CA: opts.kc, PED: opts.kc, aL: opts.kc, "a'L": opts.kc, bL: opts.kc, "b'L": opts.kc, gL: opts.kc, LH: opts.lh };
  const npCentres = {};
  for (const [name, p] of Object.entries(BRAIN.neuropils)) {
    if (!p) continue;
    const base = name.replace(/\(.\)$/, '');
    cloud(p, col(NP_COL[base] || opts.shell), 3.6, 0.22);
    if (['AL', 'CA', 'LH'].includes(base) || base === 'gL') {
      let s = [0, 0, 0]; for (let i = 0; i < p.length; i += 3) { s[0] += p[i]; s[1] += p[i + 1]; s[2] += p[i + 2]; }
      npCentres[name] = X(...s.map(v => v / (p.length / 3)));
    }
  }

  // neurons as one big LineSegments with per-vertex colours we rewrite each frame
  const N = BRAIN.neurons, segs = [], meta = [];
  N.forEach((n, ni) => { const cnt = n.a.length; n.a.forEach((par, j) => { if (par >= 0) segs.push([ni, par, j, cnt]); }); });
  const pos = new Float32Array(segs.length * 6), vNeuron = new Uint16Array(segs.length * 2), vFrac = new Float32Array(segs.length * 2);
  segs.forEach(([ni, a, b, cnt], s) => {
    const p = N[ni].p;
    pos.set(X(p[a * 3], p[a * 3 + 1], p[a * 3 + 2]), s * 6); pos.set(X(p[b * 3], p[b * 3 + 1], p[b * 3 + 2]), s * 6 + 3);
    vNeuron[s * 2] = vNeuron[s * 2 + 1] = ni; vFrac[s * 2] = a / cnt; vFrac[s * 2 + 1] = b / cnt;
  });
  const colors = new Float32Array(segs.length * 6);
  const lg = new T.BufferGeometry();
  lg.setAttribute('position', new T.BufferAttribute(pos, 3));
  lg.setAttribute('color', new T.BufferAttribute(colors, 3));
  const lines = new T.LineSegments(lg, new T.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.8, depthWrite: false, blending: T.AdditiveBlending }));
  root.add(lines);

  const glomOf = n => n.type.split('_')[0];
  const AVOID = new Set(['MBON01', 'MBON03', 'MBON04', 'MBON05', 'MBON29']);
  const hash = s => { let h = 0; for (const ch of String(s)) h = (h * 31 + ch.charCodeAt(0)) | 0; return ((h >>> 0) % 1000) / 1000; };
  const jit = N.map(n => hash(n.id));

  let A = null, t0 = performance.now(), level = new Float32Array(N.length), baseCol = [];
  N.forEach(n => baseCol.push(n.cls === 'pn' ? COLORS.pn : n.cls === 'kc' ? COLORS.kc : n.cls === 'apl' ? COLORS.apl : AVOID.has(n.type.split('_')[0].slice(0, 6)) ? COLORS.mbon_av : COLORS.mbon_ap));

  const sm = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  const CYCLE = 7;
  let live = false, livePhase = '';
  function targets(t, now) {      // per-neuron activity 0..1 at time t (s) into the cycle
    const out = new Float32Array(N.length); if (!A) return out;
    const on = new Set(A.kc_pattern || []);
    const kcFrac = (A.kc_pattern || []).length / 400;
    const mb = A.mbon || {}, mbMax = Math.max(1, ...Object.values(mb).map(Math.abs));
    const fade = live ? 1 : 1 - sm(CYCLE - 1.2, CYCLE - 0.2, t);
    N.forEach((n, i) => {
      let a = 0;
      if (n.cls === 'pn') a = (A.glomeruli?.[glomOf(n)] ?? 0) * sm(0.2 + jit[i] * 0.3, 0.9 + jit[i] * 0.3, t);
      else if (n.cls === 'kc') a = on.has(n.slot) ? sm(1.0 + jit[i] * 0.8, 1.5 + jit[i] * 0.8, t) * (0.75 + 0.25 * Math.sin(now * 11 + jit[i] * 40)) : 0;
      else if (n.cls === 'apl') a = Math.min(1, kcFrac * 12) * sm(1.6, 2.4, t);
      else { const v = mb[n.type] ?? mb[n.type.split('_')[0]] ?? 0; a = Math.min(1, (Math.abs(v) / mbMax) ** 2) * sm(2.4 + jit[i] * 0.6, 3.2 + jit[i] * 0.6, t); }
      out[i] = a * fade;
    });
    return out;
  }
  function phase(t) {
    if (!A) return '';
    if (live) return livePhase;
    if (t < 0.9) return 'odour reaches the antennal lobe';
    if (t < 2.0) return `${A.kc_active} Kenyon cells fire · APL clamps the rest`;
    if (t < 3.4) return 'mushroom body output neurons vote';
    return A.valence > 0.05 ? 'decision: approach' : A.valence < -0.05 ? 'decision: avoid' : 'decision: meh';
  }

  // drag to rotate, wheel/pinch to zoom; otherwise it turns slowly by itself
  let yaw = 0, pitch = 0.08, dist = span * 2.0, drag = null, idleAt = 0;
  cv.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY, yaw, pitch]; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => { if (!drag) return; yaw = drag[2] + (e.clientX - drag[0]) * 0.008; pitch = Math.max(-1.2, Math.min(1.2, drag[3] + (e.clientY - drag[1]) * 0.006)); });
  cv.addEventListener('pointerup', () => { drag = null; idleAt = performance.now(); });
  cv.addEventListener('wheel', e => { e.preventDefault(); dist = Math.max(span * 0.6, Math.min(span * 3, dist * (1 + e.deltaY * 0.001))); }, { passive: false });

  const labels = document.getElementById('blabels');
  const LAB = [['AL(R)', 'antennal lobe'], ['CA(R)', 'mushroom body calyx'], ['LH(R)', 'lateral horn'], ['gL(R)', 'MB lobes']];
  labels.innerHTML = LAB.map(([k, s]) => `<span data-k="${k}">${s}</span>`).join('');

  function size() { const r = host.getBoundingClientRect(); renderer.setSize(r.width, r.height, false); camera.aspect = r.width / Math.max(r.height, 1); camera.updateProjectionMatrix(); }
  new ResizeObserver(size).observe(host); size();

  const tmp = new T.Vector3();
  function frame(now) {
    const t = ((now - t0) / 1000) % CYCLE;
    if (!drag && !opts.reduce && now - idleAt > 2500) yaw += 0.0025;
    camera.position.set(Math.sin(yaw) * Math.cos(pitch) * dist, Math.sin(pitch) * dist, Math.cos(yaw) * Math.cos(pitch) * dist);
    camera.lookAt(0, 0, 0);
    const tg = targets(live || opts.reduce ? 4 : t, now / 1000);
    for (let i = 0; i < N.length; i++) level[i] += (tg[i] - level[i]) * 0.25;
    // colour: dim base + glow; the glow travels out along each neuron from its root
    for (let v = 0; v < vNeuron.length; v++) {
      const ni = vNeuron[v], b = baseCol[ni], cls = N[ni].cls;
      const wave = cls === 'apl' ? 1 : sm(vFrac[v] - 0.25, vFrac[v], Math.min(1, level[ni] * 1.4));
      const g = level[ni] * wave, dim = cls === 'kc' ? 0.025 : cls === 'apl' ? 0.02 : cls === 'mbon' ? 0.03 : 0.1;
      const k = dim + g * (cls === 'apl' ? 0.15 : cls === 'mbon' ? 0.38 : cls === 'kc' ? 0.6 : 1.1);
      colors[v * 3] = b.r * k; colors[v * 3 + 1] = b.g * k; colors[v * 3 + 2] = b.b * k;
    }
    lg.attributes.color.needsUpdate = true;
    renderer.render(scene, camera);
    // labels
    const r = host.getBoundingClientRect();
    labels.querySelectorAll('span').forEach(el => {
      const p = npCentres[el.dataset.k]; if (!p) { el.hidden = true; return; }
      tmp.set(...p).applyMatrix4(root.matrixWorld).project(camera);
      el.hidden = tmp.z > 1; el.style.transform = `translate(${(tmp.x * 0.5 + 0.5) * r.width}px, ${(-tmp.y * 0.5 + 0.5) * r.height}px)`;
    });
    document.getElementById('bphase').textContent = phase(t);
    document.getElementById('bclock').textContent = live ? 'live playback' : 't = ' + t.toFixed(2) + ' s';
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  return {
    show(a) { A = a; live = false; t0 = performance.now(); },
    live(a, phaseText) { A = a; live = true; livePhase = phaseText; },
  };
}
