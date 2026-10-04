// Fly Duel core: two flies with glowing blades, a tiny neural-network policy each (shared
// weights, self-play), trained with PPO. Plain JS so it runs in Node and in the browser.
(function (root) {
  // ---------------- arena ----------------
  const A = { R: 6, bodyR: 0.45, blade: 1.7, dt: 0.05, maxSteps: 600, hitsToWin: 3,
              maxV: 2.6, maxTurn: 3.2, swingAcc: 60, swingDamp: 6, alphaMax: 1.9 };

  function makeFly(x, y, h) { return { x, y, h, v: 0, a: 0, w: 0, hits: 0, cool: 0, ccool: 0 }; }
  function bladeSeg(f) {
    const px = f.x + Math.cos(f.h) * A.bodyR * 0.9, py = f.y + Math.sin(f.h) * A.bodyR * 0.9, ang = f.h + f.a;
    return [px, py, px + Math.cos(ang) * A.blade, py + Math.sin(ang) * A.blade];
  }
  function segSegDist(a, b) {   // min distance between two segments (2D)
    const d = (p, s) => { const [x1, y1, x2, y2] = s, dx = x2 - x1, dy = y2 - y1, L = dx * dx + dy * dy || 1e-9;
      const t = Math.max(0, Math.min(1, ((p[0] - x1) * dx + (p[1] - y1) * dy) / L)); return Math.hypot(p[0] - x1 - t * dx, p[1] - y1 - t * dy); };
    const cross = (o, p, q) => (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0]);
    const A1 = [a[0], a[1]], A2 = [a[2], a[3]], B1 = [b[0], b[1]], B2 = [b[2], b[3]];
    if (cross(A1, A2, B1) * cross(A1, A2, B2) < 0 && cross(B1, B2, A1) * cross(B1, B2, A2) < 0) return 0;
    return Math.min(d(A1, b), d(A2, b), d(B1, a), d(B2, a));
  }
  function pointSegDist(px, py, s) { const [x1, y1, x2, y2] = s, dx = x2 - x1, dy = y2 - y1, L = dx * dx + dy * dy;
    const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / L)); return Math.hypot(px - x1 - t * dx, py - y1 - t * dy); }

  function makeEnv(rng) {
    const E = { f: [null, null], t: 0, events: [] };
    E.reset = () => {
      const ang = rng() * Math.PI * 2, d = 2.2 + rng() * 1.2;
      E.f[0] = makeFly(Math.cos(ang) * d, Math.sin(ang) * d, ang + Math.PI + (rng() - 0.5) * 0.8);
      E.f[1] = makeFly(-Math.cos(ang) * d, -Math.sin(ang) * d, ang + (rng() - 0.5) * 0.8);
      E.t = 0; return [obs(0), obs(1)];
    };
    const rot = (x, y, h) => [Math.cos(-h) * x - Math.sin(-h) * y, Math.sin(-h) * x + Math.cos(-h) * y];
    function obs(i) {            // egocentric view of the duel, ~16 numbers
      const me = E.f[i], op = E.f[1 - i];
      const [rx, ry] = rot(op.x - me.x, op.y - me.y, me.h), dist = Math.hypot(rx, ry);
      const ob = bladeSeg(op), [tx, ty] = rot(ob[2] - me.x, ob[3] - me.y, me.h);
      const [vx, vy] = rot(Math.cos(op.h) * op.v, Math.sin(op.h) * op.v, me.h);
      const wallD = A.R - Math.hypot(me.x, me.y), [wx, wy] = rot(-me.x, -me.y, me.h);
      return [rx / 4, ry / 4, dist / 4, Math.cos(op.h - me.h), Math.sin(op.h - me.h), me.a / A.alphaMax, me.w / 8, me.v / A.maxV,
        op.a / A.alphaMax, op.w / 8, tx / 4, ty / 4, vx / A.maxV, vy / A.maxV, wallD / A.R, Math.atan2(wy, wx) / Math.PI];
    }
    // actions per fly: [forward, turn, swing], each roughly in [-1, 1]
    E.step = acts => {
      E.events = []; const rew = [0, 0];
      for (let i = 0; i < 2; i++) {
        const f = E.f[i], a = acts[i].map(v => Math.max(-1, Math.min(1, v)));
        f.v += (a[0] * A.maxV - f.v) * 4 * A.dt; f.h += a[1] * A.maxTurn * A.dt;
        f.w += (a[2] * A.swingAcc - A.swingDamp * f.w) * A.dt; f.a += f.w * A.dt;
        if (Math.abs(f.a) > A.alphaMax) { f.a = Math.sign(f.a) * A.alphaMax; f.w *= -0.3; }
        f.x += Math.cos(f.h) * f.v * A.dt; f.y += Math.sin(f.h) * f.v * A.dt;
        const r = Math.hypot(f.x, f.y); if (r > A.R - A.bodyR) { f.x *= (A.R - A.bodyR) / r; f.y *= (A.R - A.bodyR) / r; f.v *= 0.3; rew[i] -= 0.01; }
        if (f.cool > 0) f.cool--; if (f.ccool > 0) f.ccool--;
      }
      // bodies can't overlap
      const [f0, f1] = E.f, dx = f1.x - f0.x, dy = f1.y - f0.y, dd = Math.hypot(dx, dy) || 1e-6, minD = 2 * A.bodyR;
      if (dd < minD) { const push = (minD - dd) / 2; f0.x -= dx / dd * push; f0.y -= dy / dd * push; f1.x += dx / dd * push; f1.y += dy / dd * push; }
      const s0 = bladeSeg(f0), s1 = bladeSeg(f1);
      // blade meets blade: a parry. Both blades bounce.
      if (segSegDist(s0, s1) < 0.08 && (Math.abs(f0.w) + Math.abs(f1.w) > 2) && f0.ccool === 0 && f1.ccool === 0) {
        const cx = (s0[2] + s1[2] + s0[0] + s1[0]) / 4, cy = (s0[3] + s1[3] + s0[1] + s1[1]) / 4;
        f0.w *= -0.6; f1.w *= -0.6; f0.cool = Math.max(f0.cool, 3); f1.cool = Math.max(f1.cool, 3); f0.ccool = f1.ccool = 6;
        E.events.push({ type: 'clash', x: cx, y: cy });   // a parry is its own reward: it stops the hit
      } else {
        for (let i = 0; i < 2; i++) {      // blade i touches fly 1-i's body: a hit, if it was swinging
          const s = i ? s1 : s0, tgt = E.f[1 - i], att = E.f[i];
          if (att.cool === 0 && pointSegDist(tgt.x, tgt.y, s) < A.bodyR && Math.abs(att.w) > 1.5) {
            att.hits++; att.cool = 12; rew[i] += 1; rew[1 - i] -= 1;
            tgt.v = -1.5; tgt.x -= Math.cos(tgt.h) * 0.3; tgt.y -= Math.sin(tgt.h) * 0.3;
            E.events.push({ type: 'hit', by: i, x: tgt.x, y: tgt.y });
          }
        }
      }
      // small nudges so early learning has something to climb: face the opponent and close in
      for (let i = 0; i < 2; i++) {
        const me = E.f[i], op = E.f[1 - i], b = Math.atan2(op.y - me.y, op.x - me.x) - me.h;
        rew[i] += 0.004 * Math.cos(b) - 0.002 * Math.min(4, Math.hypot(op.x - me.x, op.y - me.y));
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
  const OBS = 16, ACT = 3, H = 64;
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

  const api = { _backward: backward, _zeros: zeros, A, makeEnv, bladeSeg, makeAgent, act, forward, makePPO, pack, unpack, mulberry32 };
  if (typeof module !== 'undefined') module.exports = api; else root.FlyDuel = api;
})(this);
