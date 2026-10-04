const FP = require('./core.js'), D = require('../data/male-cns-flight.json');
const seed = +(process.argv[2] || 1), N = +(process.argv[3] || 120), turb = +(process.argv[4] || 0.5);
const net = FP.buildNet(D), rng = FP.mulberry32(seed), brain = FP.makeBrain(net), dt = 1 / 120;
const plane = FP.makePlane(rng), T = FP.makeTrainer(net, rng);
for (let i = 0; i < 240; i++) brain.step([0, 0, 0], dt);
const scores = []; let simT = 0;
for (let a = 0; a < N; a++) {
  plane.setRng(FP.mulberry32(T.begin())); plane.reset(); let c = false, lev = 0;
  while (!c && plane.s.t < 90) { brain.step(plane.s.om, dt); c = plane.step(T.act(brain.r, dt), turb, dt); lev += plane.s.phi ** 2 * dt; }
  simT += plane.s.t; const sc = plane.s.t - 5 * lev / Math.max(plane.s.t, 1); T.end(sc); scores.push(plane.s.t);
}
const ch = []; for (let i = 0; i < N; i += 20) ch.push((scores.slice(i, i + 20).reduce((s, v) => s + v, 0) / 20).toFixed(1));
console.log(`seed ${seed} turb ${turb}: mean aloft per 20 attempts ${ch.join(' ')} | sim time ${(simT / 60).toFixed(1)} min`);
