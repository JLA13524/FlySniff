// Fly Pilot core: the fly's real flight-stabilisation subcircuit (MaleCNS v1.0) as a rate
// network, a toy plane, and reward-modulated learning of the muscle -> control mapping.
// Runs in the browser and in Node (tests).
(function (root) {
  const SIGN = { acetylcholine: 1, gaba: -1, glutamate: -1, octopamine: 0.3, unclear: 0.5 };

  // ---------------- network ----------------
  function buildNet(D) {
    const N = D.nodes.length, [pre, post, w] = D.edges;
    const tot = new Float64Array(N);
    for (let e = 0; e < pre.length; e++) tot[post[e]] += w[e];
    const inn = Array.from({ length: N }, () => []);
    for (let e = 0; e < pre.length; e++) if (pre[e] !== post[e]) inn[post[e]].push([pre[e], w[e]]);
    // CSR arrays, weights normalised per target so every neuron sees ~unit total input
    const ptr = new Int32Array(N + 1), src = [], wt = [];
    for (let i = 0; i < N; i++) {
      ptr[i] = src.length;
      for (const [j, c] of inn[i]) { src.push(j); wt.push((SIGN[D.nodes[j].nt] ?? 0.5) * c / Math.max(tot[i], 1)); }
    }
    ptr[N] = src.length;
    const cls = D.nodes.map(n => n.cls);
    const idx = c => cls.map((k, i) => k === c ? i : -1).filter(i => i >= 0);
    return { N, ptr, src: Int32Array.from(src), wt: Float64Array.from(wt), cls, lptc: idx('lptc'), mn: idx('mn'), nodes: D.nodes, axes: lptcAxes(D) };
  }

  // Preferred rotation (roll, pitch, yaw) of each visual neuron, from fly physiology:
  // HS: horizontal front-to-back motion in their own eye (yaw towards the other side);
  // H1/H2: back-to-front (yaw towards their own side); VS: downward motion, so rotations
  // about horizontal axes, frontal cells ~pitch, lateral cells ~roll. The 9 VS cells per
  // side aren't numbered in this release, so they're ordered by where their dendrites sit.
  function lptcAxes(D) {
    const axes = {};
    const cent = n => { const p = n.skel.p; let s = [0, 0, 0]; for (let i = 0; i < p.length; i += 3) { s[0] += p[i]; s[1] += p[i + 1]; s[2] += p[i + 2]; } return s.map(v => v / (p.length / 3)); };
    for (const side of ['L', 'R']) {
      const sg = side === 'L' ? 1 : -1;          // + = roll right / yaw right excites left-side cells
      const vs = D.nodes.map((n, i) => [n, i]).filter(([n]) => n.type === 'VS' && n.side === side)
        .sort((a, b) => cent(a[0])[2] - cent(b[0])[2]);
      vs.forEach(([, i], k) => { const al = (k / Math.max(vs.length - 1, 1)) * Math.PI / 2; axes[i] = [sg * Math.sin(al), Math.cos(al), 0]; });
      D.nodes.forEach((n, i) => {
        if (n.side !== side) return;
        if (n.type === 'HSN') axes[i] = [sg * 0.3, 0, sg];
        if (n.type === 'HSE') axes[i] = [0, 0, sg];
        if (n.type === 'HSS') axes[i] = [-sg * 0.3, 0, sg];
        if (n.type === 'H1' || n.type === 'H2') axes[i] = [0, 0, -sg];
      });
    }
    return axes;
  }

  function makeBrain(net, P = {}) {
    const x = new Float64Array(net.N), r = new Float64Array(net.N), I = new Float64Array(net.N);
    const TAU = P.tau ?? 0.03, BIAS = P.bias ?? 0.35, GAIN = P.gain ?? 1.6, KIN = P.kin ?? 6;
    return {
      x, r,
      step(omega, dt) {            // omega = [p, q, r] body rates (rad/s)
        I.fill(0);
        for (const i of net.lptc) { const a = net.axes[i]; if (a) I[i] = KIN * (a[0] * omega[0] + a[1] * omega[1] + a[2] * omega[2]); }
        const k = dt / TAU;
        for (let i = 0; i < net.N; i++) {
          let s = 0; for (let e = net.ptr[i]; e < net.ptr[i + 1]; e++) s += net.wt[e] * r[net.src[e]];
          x[i] += k * (-x[i] + GAIN * s + I[i] + BIAS);
        }
        for (let i = 0; i < net.N; i++) r[i] = x[i] > 0 ? Math.tanh(x[i]) : 0;
      },
      reset() { x.fill(0); r.fill(0); },
    };
  }

  // ---------------- plane ----------------
  const D2R = Math.PI / 180;
  function makePlane(rng0) {
    let rng = rng0; const s = {};
    const reset = () => Object.assign(s, { phi: (rng() - 0.5) * 6 * D2R, th: (rng() - 0.5) * 4 * D2R, psi: 0, p: 0, q: 0, r: 0, h: 150, gp: 0, gq: 0, gr: 0, t: 0, x: 0 });
    reset();
    const gauss = () => { let u = 0; for (let i = 0; i < 6; i++) u += rng(); return (u - 3) * 1.41; };
    return {
      s, reset, setRng(f) { rng = f; },
      step(u, turb, dt) {           // u = [aileron, elevator, rudder] in [-1, 1]
        const V = 45;
        const c = 4 * turb;     // gust strength; turb 0.5 is a breezy day
        for (const [k, sd] of [['gp', 1], ['gq', 0.45], ['gr', 0.4]]) s[k] += (-s[k] / 0.7) * dt + sd * c * Math.sqrt(dt) * gauss();
        const pd = -1.2 * s.p - 1.0 * Math.sin(s.phi) + 2.6 * u[0] + s.gp;      // dihedral gives a gentle wings-level tendency
        const qd = -1.8 * s.q - 1.6 * s.th + 2.2 * u[1] + s.gq;
        const rd = -1.0 * s.r + 0.25 * s.p + 1.5 * u[2] + s.gr;
        s.p += pd * dt; s.q += qd * dt; s.r += rd * dt;
        s.phi += s.p * dt; s.th += s.q * dt; s.psi += (s.r + 9.81 / V * Math.tan(s.phi)) * dt;
        s.h += (V * Math.sin(s.th) - 22 * (1 / Math.max(Math.cos(s.phi), 0.2) - 1)) * dt;
        s.x += V * dt; s.t += dt;
        return Math.abs(s.phi) > 75 * D2R || Math.abs(s.th) > 45 * D2R || s.h < 0;
      },
    };
  }

  // ---------------- learning ----------------
  // The fly's wing muscles are paired left/right. Controls read the motor neurons the way
  // a fly's steering does: aileron and rudder from left-minus-right activity, elevator
  // from left-plus-right, through a 3 x 15 "muscle map" W that starts random.
  // Learning happens per attempt, like trial and error: fly with W nudged one way, then the
  // same nudge the other way through the same gusts; whichever flew better wins, and W moves
  // towards it in proportion to the difference (the "dopamine"). Paired random search.
  function makeTrainer(net, rng, opt = {}) {
    const types = [...new Set(net.mn.map(i => net.nodes[i].type))];
    const pairs = types.map(t => ['L', 'R'].map(sd => net.mn.filter(i => net.nodes[i].type === t && net.nodes[i].side === sd)));
    const K = types.length, DIM = 3 * K;
    const gauss = () => { let u = 0; for (let i = 0; i < 6; i++) u += rng(); return (u - 3) * 1.41; };
    const T = { types, pairs, K, W: new Float64Array(DIM), d: new Float64Array(DIM), Wtry: new Float64Array(DIM),
      alpha: opt.alpha ?? 0.6, sigma: opt.sigma ?? 0.4, phase: 0, rPlus: 0, sdR: 10, expected: 0, attempt: 0, gustSeed: 1, u: [0, 0, 0] };
    const mean = new Float64Array(net.N);
    let primed = false;
    T.reset = () => { T.W = new Float64Array(DIM).map(() => gauss() * 0.3); T.phase = 0; T.attempt = 0; T.expected = 0; T.sdR = 10; };
    T.reset();
    T.begin = () => {             // returns the gust seed for this attempt (shared by the pair)
      if (T.phase === 0) { T.d = new Float64Array(DIM).map(() => gauss()); T.gustSeed = 1000 + Math.floor(rng() * 1e9); }
      const sg = T.phase === 0 ? 1 : -1;
      for (let i = 0; i < DIM; i++) T.Wtry[i] = T.W[i] + sg * T.sigma * T.d[i];
      T.attempt++;
      return T.gustSeed;
    };
    const avg = (ids, r) => ids.length ? ids.reduce((s, i) => s + r[i] - mean[i], 0) / ids.length : 0;
    T.act = (r, dt) => {
      if (!primed) { mean.set(r); primed = true; }
      for (const side of pairs) for (const ids of side) for (const i of ids) mean[i] += (r[i] - mean[i]) * dt / 2;
      let a = 0, e = 0, ru = 0;
      for (let k = 0; k < K; k++) {
        const L = avg(pairs[k][0], r), R = avg(pairs[k][1], r);
        a += T.Wtry[k] * (L - R) * 8; e += T.Wtry[K + k] * (L + R) * 4; ru += T.Wtry[2 * K + k] * (L - R) * 8;
      }
      T.u = [Math.tanh(a), Math.tanh(e), Math.tanh(ru)];
      return T.u;
    };
    // score = seconds aloft, minus a little for flying with the wings tilted
    T.end = score => {
      const surprise = score - T.expected;          // dopamine: better or worse than it expected?
      T.expected += (score - T.expected) * 0.15;
      if (T.phase === 0) { T.rPlus = score; T.phase = 1; return { dopamine: surprise, updated: false }; }
      const diff = T.rPlus - score;
      T.sdR += (Math.abs(diff) - T.sdR) * 0.1;
      const step = T.alpha * T.sigma * diff / (2 * Math.max(T.sdR, 1));
      for (let i = 0; i < DIM; i++) T.W[i] += step * T.d[i];
      T.phase = 0;
      return { dopamine: surprise, updated: true, winner: diff >= 0 ? '+' : '-' };
    };
    return T;
  }

  function mulberry32(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  const api = { buildNet, makeBrain, makePlane, makeTrainer, mulberry32, D2R };
  if (typeof module !== 'undefined') module.exports = api; else root.FlyPilot = api;
})(this);
