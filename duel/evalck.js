const D = require('./core.js'), CK = require('./checkpoints.json');
for (const c of CK.checkpoints) {
  const ag = D.unpack(c.agent), rng = D.mulberry32(11), env = D.makeEnv(rng); let st = { hits: 0, clash: 0, steps: 0, dist: 0, swing: 0, eps: 0 };
  for (let ep = 0; ep < 6; ep++) { let o = env.reset(), done = false;
    while (!done) { const acts = [0, 1].map(i => { const s = D.act(ag, o[i], rng); return Array.from(s.mu, (m, k) => m + (s.a[k] - m) * 0.5); });
      const r = env.step(acts); o = r.obs; done = r.done; st.steps++; st.dist += Math.hypot(env.f[0].x - env.f[1].x, env.f[0].y - env.f[1].y); st.swing += Math.abs(env.f[0].w);
      for (const e of env.events) e.type === 'hit' ? st.hits++ : st.clash++; } st.eps++; }
  const min = st.steps * D.A.dt / 60;
  console.log(`iter ${c.iter}: hits/min ${(st.hits / min).toFixed(1)} clashes/min ${(st.clash / min).toFixed(1)} mean dist ${(st.dist / st.steps).toFixed(2)} mean |swing| ${(st.swing / st.steps).toFixed(1)} bout ${(st.steps / st.eps * D.A.dt).toFixed(1)}s`);
}
