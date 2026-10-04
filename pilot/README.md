# Fly Pilot

A fruit fly's real flight-stabilisation circuit learns to keep a plane level in turbulence.

- **Real:** `data/male-cns-flight.json`, the 264-neuron pathway from MaleCNS v1.0 (via neuPrint):
  28 rotation-detecting visual neurons (HS, VS, H1, H2), 40 brain interneurons, 56 descending
  neurons (DNa02, DNp15…), 110 nerve-cord interneurons and 30 wing-steering motor neurons
  (b1–b3, i1, i2, iii1, iii3, hg1–4, tp1, tp2, tpn, ps1), with ~4,000 connections and each
  neuron's transmitter, plus downsampled skeletons and a brain + nerve cord outline.
- **Simulated:** a rate network on those connections (acetylcholine excites, GABA and glutamate
  inhibit). The plane's roll/pitch/yaw rates drive the visual neurons through each cell type's
  preferred rotation. The 9 VS cells per side aren't numbered in this release, so they're ordered
  by dendrite position.
- **Learned:** a 3 × 15 map from left/right wing motor-neuron activity to aileron, elevator and
  rudder, by paired random search per attempt (same gusts both ways, keep the better). The
  wiring never changes.

`core.js` holds the network, plane and trainer and runs in Node too:

```bash
node pilot/train_test.js 1 120     # seed, attempts: prints mean time aloft per 20 attempts
python pilot/build.py              # -> pilot/fly-pilot.html
```

Typical run at turbulence 0.45 (moderate): ~5–12 s aloft at first, ~20–50 s after 120 attempts. A perfect
rate-damping autopilot on the same plane manages 90 s; the fly's circuit, read out as well as
possible, tops out around 35–45 s, because a lot of rotation information doesn't survive the
trip to the wing motor neurons in this simplified model.
