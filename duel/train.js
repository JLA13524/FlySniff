// node duel/train.js <iters> <seed> -> duel/checkpoints.json
const D = require('./core.js'), fs = require('fs');
const N = +(process.argv[2] || 200), seed = +(process.argv[3] || 7), rng = D.mulberry32(seed);
const P = D.makePPO(rng);
const marks = new Set([0, 3, 10, 25, 50, 100, 200, 400, 800, N]);
const ck = []; const t0 = Date.now();
const save = () => fs.writeFileSync(__dirname + '/checkpoints.json', JSON.stringify({ seed, log: P.log, checkpoints: ck }));
ck.push({ iter: 0, steps: 0, agent: D.pack(P.agent) });
for (let i = 1; i <= N; i++) {
  const r = P.iterate();
  if (marks.has(i)) { ck.push({ iter: i, steps: P.steps, agent: D.pack(P.agent) }); save(); }
  if (i % 10 === 0 || i === 1) console.log(`it ${i} steps ${P.steps} ret ${r.meanRet.toFixed(3)} hits/min ${r.hitsPerMin == null ? '-' : r.hitsPerMin.toFixed(1)} std ${r.std.map(v => v.toFixed(2)).join(',')} kl ${r.kl.toFixed(4)} | ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
save();
