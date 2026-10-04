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
pip install -r requirements.txt

# offline, synthetic circuit + fake triathlon fortnight
python -m flysniff --demo

# your Strava bulk export (Settings › My Account › Download your data)
export NEUPRINT_TOKEN=...      # neuprint.janelia.org › Account › Auth token
python -m flysniff --csv ~/Downloads/export/activities.csv

# or straight from the API (kudos + real HR decoupling)
export STRAVA_ACCESS_TOKEN=... # scope activity:read_all
python -m flysniff --api -n 80 --streams
```

The first real run pulls the PN/KC/MBON/APL subcircuit from neuPrint and caches
it to `circuit.npz`. The dataset is auto-detected (anything named `male-cns*`);
override it with `--dataset`. Use `--demo-circuit` to skip neuPrint and use real
activities with the synthetic circuit.

Then open the **Fly Sniff** page and drop `results.json` onto it.

## Notes

- PN types are matched like `DM1_lPN` / `V_l2PN`. If Janelia's naming differs in
  your dataset version, change the regex in `connectome.py`.
- Without `--streams`, HR drift falls back to a crude (max − avg HR) proxy.
- Features are z-scored within your own data, so the fly judges you against
  yourself. Everyone gets a favourite and a nemesis.
