# flysniff 🪰

Feed your Strava training into the olfactory circuit of a real fruit fly brain
(the MaleCNS v1.0 connectome, Janelia + Google, Sept 2026) and find out what it
thinks of your sessions.

> "the fly hates your Tuesday intervals. it smells like a rival male."

**This is not science.** It's a real wiring diagram doing a silly job.

## How it works

```
Strava activity ─► 6 features ─► 6 glomeruli ─► PNs ─► Kenyon cells ─► MBONs ─► approach / avoid
                   (joke map)                         (APL keeps ~5% firing)
```

| Feature | Glomerulus | What that glomerulus really smells | Innate |
|---|---|---|---|
| speed (vs your other sessions of that sport) | DM1 | fruity ester | 👍 |
| easiness (low HR) | DM4 | vinegar / fermentation | 👍 |
| cadence | VM2 | ripe fruit | 👍 |
| HR drift (Pa:HR decoupling) | V | CO₂ | 👎 |
| climbing (m/km) | DA2 | geosmin (damp soil, mould) | 👎👎 |
| interval-ness (name + HR spikiness) | DA1 | cVA, a male pheromone | 👎 |

The verdict blends two routes, like the real fly:

- **Innate** (60%): hard-wired glomerular valence, lateral-horn style.
- **Learned** (40%): the mushroom body. KC→MBON weights are the real synapse
  counts. If kudos are available, high-kudos sessions act as **dopamine reward**
  and depress KC→avoidance-MBON synapses for the KCs they activated, which is the
  actual MB plasticity rule. So the fly slowly learns to like whatever your mates
  like. MBON approach/avoid signs roughly follow Aso et al. 2014.

## Run it

```bash
pip install -r requirements.txt   # fitdecode is only needed for --fit

# fake triathlon fortnight through the real circuit
python -m flysniff --demo

# .fit / .fit.gz files or whole folders (watch exports, or export/activities from Strava)
python -m flysniff --fit ~/Downloads/export/activities

# your Strava bulk export (Settings › My Account › Download your data)
python -m flysniff --csv ~/Downloads/export/activities.csv

# or straight from the API (kudos + real HR decoupling)
export STRAVA_ACCESS_TOKEN=... # scope activity:read_all
python -m flysniff --api -n 80 --streams
```

The real circuit ships in `data/male-cns-olfactory.json`: the uniglomerular PNs for
the six glomeruli, all Kenyon cells, MBONs and APL from **male-cns v1.0**, with
synapse counts (38 PNs, 4,064 KCs, 97 MBONs, ~73k connections), pulled from
neuprint.janelia.org's public API on 4 Oct 2026. No token needed to run.

To re-pull it yourself (or another version), set `NEUPRINT_TOKEN` and pass
`--dataset male-cns:v1.0`; the result is cached to `circuit.npz`.
`--demo-circuit` swaps in the synthetic stand-in instead.

Data: MaleCNS v1.0 connectome, HHMI Janelia, Google Research and collaborators,
CC BY 4.0 (https://male-cns.janelia.org).

Then bake it into a page with `python ui/build.py results.json my-fly.html`, or drop
`results.json` onto the published **Fly Sniff** page.

### Uploading .fit files on the page

The page can also smell `.fit` / `.fit.gz` files on its own: **Upload .fit files** (or drag
them in). `ui/fitfly.js` decodes them in the browser and runs the same feature and circuit
maths as the Python (a parity check agrees to ~0.001), using the circuit `build.py` baked
into the page. Files never leave the browser. Differences from the CLI: no kudos learning,
and while you have fewer than six files of your own they're z-scored against the demo
sessions so there's something to compare with.

FIT stream metrics: Pa:HR decoupling from moving samples (first vs second half), climbing
from a smoothed altitude trace, and speed variability (intervals spike it).

### The 3D brain

`data/male-cns-brain3d.json` holds real skeletons (downsampled) for all 38 PNs, 97
MBONs, both APLs and the same 400 Kenyon cells the page's grid shows, plus brain and
mushroom-body outlines, all from neuPrint. The page replays each session's computed
response on them: PNs light by glomerulus drive, then the Kenyon cells that actually
fired, then APL, then MBONs by output. Which neurons light and how strongly comes from
the model; the timing of the wave is illustrative.

### Playing a session back

For an uploaded `.fit`, **Play session** slides a one-minute window through the
recording (every 5 s) and asks the fly about each moment: that minute's pace, HR,
cadence, climbing, speed variability over the last 5 minutes, and decoupling against
your first 10 minutes. Each moment is scored against the spread of your sessions, so
reps smell like a PB and the late-session drift smells like CO₂. Click the trace to jump.

## Notes

- In the male fly, DA1 (the cVA pheromone glomerulus) has 15 PNs against DM1's 2,
  so "intervalness" gets far more Kenyon cell input than pace does. That's the real
  anatomy, not a bug.
- Without `--streams`, HR drift falls back to a crude (max − avg HR) proxy.
- Features are z-scored within your own data, so the fly judges you against
  yourself. Everyone gets a favourite and a nemesis.
