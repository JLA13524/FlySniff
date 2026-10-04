const D = require('./core.js'), CK = require('./checkpoints.json');
for (const c of CK.checkpoints) {
  const ag = D.unpack(c.agent), rng = D.mulberry32(11), env = D.makeEnv(rng); const st = { hit: 0, clash: 0, dodge: 0, dash: 0, steps: 0, z: 0, air: 0, eps: 0 };
  for (let ep = 0; ep < 6; ep++) { let o = env.reset(), done = false;
    while (!done) { const acts = [0, 1].map(i => { const s = D.act(ag, o[i], rng); return Array.from(s.mu, (m, k) => m + (s.a[k] - m) * 0.5); });
      const r = env.step(acts); o = r.obs; done = r.done; st.steps++; st.z += env.f[0].z; st.air += env.f[0].z > 0.1 ? 1 : 0;
      for (const e of env.events) st[e.type]++; } st.eps++; }
  const min = st.steps * D.A.dt / 60;
  console.log(`iter ${c.iter}: hits ${(st.hit / min).toFixed(0)}/min clashes ${(st.clash / min).toFixed(0)} dodges ${(st.dodge / min).toFixed(0)} dashes ${(st.dash / min).toFixed(0)} | airborne ${(100 * st.air / st.steps).toFixed(0)}% mean height ${(st.z / st.steps).toFixed(2)} | bout ${(st.steps / st.eps * D.A.dt).toFixed(1)}s`);
}
