// Fly Duel core: two flying flies with glowing blades, a tiny neural-network policy each (shared
// weights, self-play), trained with PPO. Plain JS so it runs in Node and in the browser.
(function (root) {
  // ---------------- arena ----------------
  // Flies walk or fly (height 0..zMax), turn, swing their blade sideways (a) and tilt it up or
  // down (e), and can dash sideways with a cooldown. Hits and parries are checked in 3D.
  const A = { R: 6, zMax: 3, bodyR: 0.45, blade: 1.7, dt: 0.05, maxSteps: 600, hitsToWin: 3,
              maxV: 2.6, maxTurn: 3.2, swingAcc: 60, swingDamp: 6, alphaMax: 1.9, pitchMax: 0.9,
              climb: 7, dashV: 6, dashCool: 20, OBS: 24, ACT: 6 };

  function makeFly(x, y, h) { return { x, y, z: 0, h, v: 0, vz: 0, a: 0, w: 0, e: 0, er: 0, dx: 0, dy: 0, dash: 0, hits: 0, cool: 0, ccool: 0, danger: null, dodges: 0 }; }
  const body = f => [f.x, f.y, f.z + 0.45];
  function bladeSeg(f) {
    const [cx, cy, cz] = body(f), px = cx + Math.cos(f.h) * A.bodyR * 0.9, py = cy + Math.sin(f.h) * A.bodyR * 0.9, ang = f.h + f.a, ce = Math.cos(f.e);
    return [px, py, cz, px + ce * Math.cos(ang) * A.blade, py + ce * Math.sin(ang) * A.blade, cz + Math.sin(f.e) * A.blade];
  }
  function pointSegDist(p, s) {
    const d = [s[3] - s[0], s[4] - s[1], s[5] - s[2]], L = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] || 1e-9;
    const t = Math.max(0, Math.min(1, ((p[0] - s[0]) * d[0] + (p[1] - s[1]) * d[1] + (p[2] - s[2]) * d[2]) / L));
    return Math.hypot(p[0] - s[0] - t * d[0], p[1] - s[1] - t * d[1], p[2] - s[2] - t * d[2]);
  }
  function segSegDist(P, Q) {    // closest distance between two 3D segments
    const u = [P[3] - P[0], P[4] - P[1], P[5] - P[2]], v = [Q[3] - Q[0], Q[4] - Q[1], Q[5] - Q[2]], w = [P[0] - Q[0], P[1] - Q[1], P[2] - Q[2]];
    const dot = (x, y) => x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    const a = dot(u, u), b = dot(u, v), c = dot(v, v), d = dot(u, w), e = dot(v, w), D = a * c - b * b;
    let sc = D < 1e-9 ? 0 : (b * e - c * d) / D; sc = Math.max(0, Math.min(1, sc));
    let tc = (b * sc + e) / c; if (tc < 0) { tc = 0; sc = Math.max(0, Math.min(1, -d / a)); } else if (tc > 1) { tc = 1; sc = Math.max(0, Math.min(1, (b - d) / a)); }
    return Math.hypot(w[0] + sc * u[0] - tc * v[0], w[1] + sc * u[1] - tc * v[1], w[2] + sc * u[2] - tc * v[2]);
  }

  function makeEnv(rng) {
    const E = { f: [null, null], t: 0, events: [] };
    E.reset = () => {
      const ang = rng() * Math.PI * 2, d = 2.2 + rng() * 1.2;
      E.f[0] = makeFly(Math.cos(ang) * d, Math.sin(ang) * d, ang + Math.PI + (rng() - 0.5) * 0.8);
      E.f[1] = makeFly(-Math.cos(ang) * d, -Math.sin(ang) * d, ang + (rng() - 0.5) * 0.8);
      E.t = 0; return [obs(0), obs(1)];
    };
    const rot = (x, y, h) => [Math.cos(-h) * x - Math.sin(-h) * y, Math.sin(-h) * x + Math.cos(-h) * y];
    function obs(i) {            // egocentric view of the duel: 24 numbers
      const me = E.f[i], op = E.f[1 - i];
      const [rx, ry] = rot(op.x - me.x, op.y - me.y, me.h), rz = op.z - me.z, dist = Math.hypot(rx, ry, rz);
      const ob = bladeSeg(op), [tx, ty] = rot(ob[3] - me.x, ob[4] - me.y, me.h), tz = ob[5] - (me.z + 0.45);
      const [vx, vy] = rot(Math.cos(op.h) * op.v + op.dx, Math.sin(op.h) * op.v + op.dy, me.h);
      const wallD = A.R - Math.hypot(me.x, me.y), [wx, wy] = rot(-me.x, -me.y, me.h);
      return [rx / 4, ry / 4, rz / 3, dist / 4, Math.cos(op.h - me.h), Math.sin(op.h - me.h),
        me.a / A.alphaMax, me.w / 8, me.e / A.pitchMax, me.v / A.maxV, me.z / A.zMax, me.vz / 3, me.dash / A.dashCool,
        op.a / A.alphaMax, op.w / 8, op.e / A.pitchMax, tx / 4, ty / 4, tz / 3, vx / A.maxV, vy / A.maxV, op.vz / 3,
        wallD / A.R, Math.atan2(wy, wx) / Math.PI];
    }
    // actions per fly: [forward, turn, swing, climb, blade tilt, dash (|x| > 0.6: left if +, right if -)]
    E.step = acts => {
      E.events = []; const rew = [0, 0];
      for (let i = 0; i < 2; i++) {
        const f = E.f[i], a = acts[i].map(v => Math.max(-1, Math.min(1, v)));
        const vmax = f.z > 0.1 ? A.maxV * 1.25 : A.maxV * 0.45;     // flies are much quicker in the air than on foot
        f.v += (a[0] * vmax - f.v) * 4 * A.dt; f.h += a[1] * A.maxTurn * A.dt;
        f.w += (a[2] * A.swingAcc - A.swingDamp * f.w) * A.dt; f.a += f.w * A.dt;
        if (Math.abs(f.a) > A.alphaMax) { f.a = Math.sign(f.a) * A.alphaMax; f.w *= -0.3; }
        f.vz += (a[3] * A.climb - 3 * f.vz) * A.dt; f.z += f.vz * A.dt;
        if (f.z < 0) { f.z = 0; f.vz = Math.max(0, f.vz); } if (f.z > A.zMax) { f.z = A.zMax; f.vz = Math.min(0, f.vz); }
        f.er = a[4] * 4; f.e = Math.max(-A.pitchMax, Math.min(A.pitchMax, f.e + f.er * A.dt));
        if (f.dash > 0) f.dash--;
        if (Math.abs(a[5]) > 0.6 && f.dash === 0) {
          const side = Math.sign(a[5]); f.dx = -Math.sin(f.h) * side * A.dashV; f.dy = Math.cos(f.h) * side * A.dashV; f.dash = A.dashCool;
          E.events.push({ type: 'dash', by: i });
        }
        f.x += (Math.cos(f.h) * f.v + f.dx) * A.dt; f.y += (Math.sin(f.h) * f.v + f.dy) * A.dt;
        f.dx *= 0.82; f.dy *= 0.82;
        const r = Math.hypot(f.x, f.y); if (r > A.R - A.bodyR) { f.x *= (A.R - A.bodyR) / r; f.y *= (A.R - A.bodyR) / r; f.v *= 0.3; f.dx = f.dy = 0; rew[i] -= 0.01; }
        if (f.cool > 0) f.cool--; if (f.ccool > 0) f.ccool--;
      }
      // bodies can't overlap
      const [f0, f1] = E.f, b0 = body(f0), b1 = body(f1), dv = [b1[0] - b0[0], b1[1] - b0[1], b1[2] - b0[2]], dd = Math.hypot(...dv) || 1e-6, minD = 2 * A.bodyR;
      if (dd < minD) { const p = (minD - dd) / 2 / dd; f0.x -= dv[0] * p; f0.y -= dv[1] * p; f1.x += dv[0] * p; f1.y += dv[1] * p; }
      const s0 = bladeSeg(f0), s1 = bladeSeg(f1), moving = f => Math.abs(f.w) > 1.5 || Math.abs(f.er) > 2.5;
      let hitThisStep = [false, false];
      if (segSegDist(s0, s1) < 0.1 && (Math.abs(f0.w) + Math.abs(f1.w) + Math.abs(f0.er) + Math.abs(f1.er) > 2) && f0.ccool === 0 && f1.ccool === 0) {
        const c = [0, 1, 2].map(k => (s0[k] + s0[k + 3] + s1[k] + s1[k + 3]) / 4);
        f0.w *= -0.6; f1.w *= -0.6; f0.cool = Math.max(f0.cool, 3); f1.cool = Math.max(f1.cool, 3); f0.ccool = f1.ccool = 6;
        E.events.push({ type: 'clash', x: c[0], y: c[1], z: c[2] });   // a parry is its own reward: it stops the hit
      } else {
        for (let i = 0; i < 2; i++) {      // blade i touches fly 1-i's body while moving: a hit
          const s = i ? s1 : s0, tgt = E.f[1 - i], att = E.f[i], tb = body(tgt);
          if (att.cool === 0 && moving(att) && pointSegDist(tb, s) < A.bodyR) {
            att.hits++; att.cool = 12; rew[i] += 1; rew[1 - i] -= 1; hitThisStep[1 - i] = true; tgt.danger = null;
            tgt.v = -1.5; tgt.x -= Math.cos(tgt.h) * 0.3; tgt.y -= Math.sin(tgt.h) * 0.3; tgt.vz -= 1;
            E.events.push({ type: 'hit', by: i, x: tb[0], y: tb[1], z: tb[2] });
          }
        }
      }
      // dodges: an attacking blade comes within reach and leaves again without landing, while
      // the target got out of the way (changed height, dashed or backed off)
      for (let i = 0; i < 2; i++) {
        const me = E.f[i], op = E.f[1 - i], near = pointSegDist(body(me), i ? s0 : s1) < 0.9 && moving(op);
        if (near && !me.danger && !hitThisStep[i]) me.danger = { x: me.x, y: me.y, z: me.z, t: E.t };
        else if (!near && me.danger) {
          const moved = Math.hypot(me.x - me.danger.x, me.y - me.danger.y) > 0.5 || Math.abs(me.z - me.danger.z) > 0.35;
          // zero-sum: the dodger gains what the swinger loses, so the pair can't farm dodges together
          if (moved && E.t - me.danger.t <= 20) { me.dodges++; rew[i] += 0.15; rew[1 - i] -= 0.15; E.events.push({ type: 'dodge', by: i, x: me.x, y: me.y, z: me.z + 0.45 }); }
          me.danger = null;
        }
      }
      // small nudges so early learning has something to climb: face the opponent and close in
      for (let i = 0; i < 2; i++) {
        const me = E.f[i], op = E.f[1 - i], b = Math.atan2(op.y - me.y, op.x - me.x) - me.h;
        rew[i] += 0.004 * Math.cos(b) - 0.002 * Math.min(4, Math.hypot(op.x - me.x, op.y - me.y, op.z - me.z));
      }
      E.t++;
      const done = E.t >= A.maxSteps || f0.hits >= A.hitsToWin || f1.hits >= A.hitsToWin;
      return { obs: [obs(0), obs(1)], rew, done };
    };
    return E;
  }

  // ---------------- tiny MLP with manual backprop ----------------
  function randn(rng) { let u = 0; for (let i = 0; i < 6; i++) u += rng(); return (u - 3) * 1.41; }
  function makeMLP(sizes, rng, outScale = 1) {
    const L = [];
    for (let k = 0; k < sizes.length - 1; k++) {
      const n = sizes[k], m = sizes[k + 1], sc = (k === sizes.length - 2 ? outScale : 1) * Math.sqrt(1 / n);
      L.push({ W: Float64Array.from({ length: n * m }, () => randn(rng) * sc), b: new Float64Array(m), n, m });
    }
    return L;
  }
  function forward(L, x) {          // returns activations per layer (tanh hidden, linear out)
    const acts = [x];
    for (let k = 0; k < L.length; k++) {
      const { W, b, n, m } = L[k], inp = acts[k], out = new Float64Array(m);
      for (let j = 0; j < m; j++) { let s = b[j]; for (let i = 0; i < n; i++) s += W[j * n + i] * inp[i]; out[j] = k < L.length - 1 ? Math.tanh(s) : s; }
      acts.push(out);
    }
    return acts;
  }
  function backward(L, acts, gOut, grads) {   // accumulate into grads (same shape as L)
    let g = gOut;
    for (let k = L.length - 1; k >= 0; k--) {
      const { W, n, m } = L[k], inp = acts[k], G = grads[k], gIn = new Float64Array(n);
      for (let j = 0; j < m; j++) { const gj = g[j]; if (!gj) continue; G.b[j] += gj; for (let i = 0; i < n; i++) { G.W[j * n + i] += gj * inp[i]; gIn[i] += gj * W[j * n + i]; } }
      if (k > 0) for (let i = 0; i < n; i++) gIn[i] *= 1 - inp[i] * inp[i];
      g = gIn;
    }
  }
  const zeros = L => L.map(l => ({ W: new Float64Array(l.W.length), b: new Float64Array(l.b.length) }));
  function makeAdam(params, lr) {
    const m = params.map(p => ({ W: new Float64Array(p.W.length), b: new Float64Array(p.b.length) })), v = params.map(p => ({ W: new Float64Array(p.W.length), b: new Float64Array(p.b.length) }));
    let t = 0;
    return { lr, step(grads, scale = 1) { t++; const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** t, c2 = 1 - b2 ** t;
      params.forEach((p, k) => { for (const key of ['W', 'b']) { const P = p[key], G = grads[k][key], M = m[k][key], V = v[k][key];
        for (let i = 0; i < P.length; i++) { const g = G[i] * scale; M[i] = b1 * M[i] + (1 - b1) * g; V[i] = b2 * V[i] + (1 - b2) * g * g; P[i] -= this.lr * (M[i] / c1) / (Math.sqrt(V[i] / c2) + 1e-8); } } }); } };
  }

  // ---------------- policy ----------------
  const OBS = A.OBS, ACT = A.ACT, H = 64;
  function makeAgent(rng) {
    return { pi: makeMLP([OBS, H, H, ACT], rng, 0.01), vf: makeMLP([OBS, H, H, 1], rng, 1), logStd: Float64Array.from({ length: ACT }, () => -0.5) };
  }
  function act(agent, o, rng, greedy = false) {
    const mu = forward(agent.pi, o)[3], a = new Float64Array(ACT); let lp = 0;
    for (let k = 0; k < ACT; k++) { const sd = Math.exp(agent.logStd[k]), z = greedy ? 0 : randn(rng); a[k] = mu[k] + sd * z; lp += -0.5 * z * z - agent.logStd[k] - 0.9189385; }
    return { a, lp, mu };
  }

  // ---------------- PPO self-play ----------------
  function makePPO(rng, opt = {}) {
    const P = { agent: makeAgent(rng), nEnv: opt.nEnv ?? 8, T: opt.T ?? 256, gamma: 0.99, lam: 0.95, clip: 0.2, epochs: opt.epochs ?? 4, mb: opt.mb ?? 512,
      lr: opt.lr ?? 3e-4, iter: 0, steps: 0, log: [] };
    const envs = Array.from({ length: P.nEnv }, () => makeEnv(rng));
    let obs = envs.map(e => e.reset());
    const optPi = makeAdam(P.agent.pi, P.lr), optVf = makeAdam(P.agent.vf, P.lr * 3);
    let epRet = envs.map(() => 0), epHits = envs.map(() => 0), finished = [];
    P.iterate = () => {
      const buf = [];      // one trajectory per (env, fly)
      for (let e = 0; e < P.nEnv; e++) for (let i = 0; i < 2; i++) buf.push({ o: [], a: [], lp: [], r: [], v: [], d: [] });
      for (let t = 0; t < P.T; t++) {
        for (let e = 0; e < P.nEnv; e++) {
          const acts = [];
          for (let i = 0; i < 2; i++) {
            const o = obs[e][i], s = act(P.agent, o, rng), B = buf[e * 2 + i];
            B.o.push(o); B.a.push(s.a); B.lp.push(s.lp); B.v.push(forward(P.agent.vf, o)[3][0]); acts.push(s.a);
          }
          const res = envs[e].step(acts);
          for (let i = 0; i < 2; i++) { buf[e * 2 + i].r.push(res.rew[i]); buf[e * 2 + i].d.push(res.done ? 1 : 0); }
          epRet[e] += res.rew[0]; epHits[e] += res.rew[0] > 0.5 || res.rew[1] > 0.5 ? 1 : 0;
          if (res.done) { finished.push({ hits: envs[e].f[0].hits + envs[e].f[1].hits, len: envs[e].t }); obs[e] = envs[e].reset(); epRet[e] = 0; }
          else obs[e] = res.obs;
        }
      }
      // GAE
      const S = [];
      buf.forEach((B, k) => {
        const e = k >> 1, i = k & 1, lastV = forward(P.agent.vf, obs[e][i])[3][0];
        let adv = 0;
        const advs = new Array(P.T);
        for (let t = P.T - 1; t >= 0; t--) {
          const nv = t === P.T - 1 ? lastV : B.v[t + 1], nonterm = 1 - B.d[t];
          const delta = B.r[t] + P.gamma * nv * nonterm - B.v[t];
          adv = delta + P.gamma * P.lam * nonterm * adv; advs[t] = adv;
        }
        for (let t = 0; t < P.T; t++) S.push({ o: B.o[t], a: B.a[t], lp: B.lp[t], adv: advs[t], ret: advs[t] + B.v[t] });
      });
      const mA = S.reduce((s, x) => s + x.adv, 0) / S.length, sA = Math.sqrt(S.reduce((s, x) => s + (x.adv - mA) ** 2, 0) / S.length) + 1e-8;
      S.forEach(x => (x.adv = (x.adv - mA) / sA));
      let kl = 0;
      for (let ep = 0; ep < P.epochs; ep++) {
        for (let i = S.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [S[i], S[j]] = [S[j], S[i]]; }
        for (let s0 = 0; s0 < S.length; s0 += P.mb) {
          const batch = S.slice(s0, s0 + P.mb), gPi = zeros(P.agent.pi), gVf = zeros(P.agent.vf), gLs = new Float64Array(ACT);
          for (const x of batch) {
            const ap = forward(P.agent.pi, x.o), mu = ap[3];
            let lp = 0; const z = new Float64Array(ACT);
            for (let k = 0; k < ACT; k++) { const sd = Math.exp(P.agent.logStd[k]); z[k] = (x.a[k] - mu[k]) / sd; lp += -0.5 * z[k] * z[k] - P.agent.logStd[k] - 0.9189385; }
            const ratio = Math.exp(lp - x.lp); kl += x.lp - lp;
            const clipped = (x.adv > 0 && ratio > 1 + P.clip) || (x.adv < 0 && ratio < 1 - P.clip);
            if (!clipped) {      // d(-ratio*adv)/dlp = -ratio*adv
              const c = -ratio * x.adv / batch.length, gMu = new Float64Array(ACT);
              for (let k = 0; k < ACT; k++) { const sd = Math.exp(P.agent.logStd[k]); gMu[k] = c * z[k] / sd; gLs[k] += c * (z[k] * z[k] - 1); }
              backward(P.agent.pi, ap, gMu, gPi);
            }
            for (let k = 0; k < ACT; k++) gLs[k] -= 0.003 / batch.length;   // entropy bonus
            const av = forward(P.agent.vf, x.o), vv = av[3][0];
            backward(P.agent.vf, av, [(vv - x.ret) / batch.length], gVf);
          }
          optPi.step(gPi); optVf.step(gVf);
          for (let k = 0; k < ACT; k++) P.agent.logStd[k] = Math.max(-2.5, Math.min(0.5, P.agent.logStd[k] - P.lr * 3 * gLs[k]));
        }
      }
      P.iter++; P.steps += P.T * P.nEnv;
      const recent = finished.splice(0);
      const rec = { iter: P.iter, steps: P.steps, episodes: recent.length,
        hitsPerMin: recent.length ? recent.reduce((s, x) => s + x.hits / (x.len * A.dt / 60), 0) / recent.length : null,
        meanRet: buf.reduce((s, B) => s + B.r.reduce((a, b) => a + b, 0), 0) / buf.length, kl: kl / (S.length * P.epochs), std: Array.from(P.agent.logStd, Math.exp) };
      P.log.push(rec); return rec;
    };
    return P;
  }

  const pack = agent => ({ pi: agent.pi.map(l => ({ W: Array.from(l.W, v => +v.toFixed(5)), b: Array.from(l.b, v => +v.toFixed(5)), n: l.n, m: l.m })), logStd: Array.from(agent.logStd) });
  const unpack = p => ({ pi: p.pi.map(l => ({ W: Float64Array.from(l.W), b: Float64Array.from(l.b), n: l.n, m: l.m })), logStd: Float64Array.from(p.logStd) });
  function mulberry32(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  const api = { _backward: backward, _zeros: zeros, A, makeEnv, body, bladeSeg, makeAgent, act, forward, makePPO, pack, unpack, mulberry32 };
  if (typeof module !== 'undefined') module.exports = api; else root.FlyDuel = api;
})(this);
