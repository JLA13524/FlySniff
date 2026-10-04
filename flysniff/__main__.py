"""python -m flysniff [--csv activities.csv | --api | --demo] [-o results.json]"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from . import strava
from .connectome import GLOMERULI, get_circuit
from .fly import SMELLS, Fly


def main() -> None:
    p = argparse.ArgumentParser(prog="flysniff", description="Let a fruit fly smell your training.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv", help="Strava bulk-export activities.csv")
    src.add_argument("--api", action="store_true", help="use STRAVA_ACCESS_TOKEN")
    src.add_argument("--fit", nargs="+", metavar="PATH", help=".fit/.fit.gz files or folders (e.g. export/activities)")
    p.add_argument("--demo", action="store_true", help="synthetic circuit + fake activities (offline)")
    p.add_argument("--demo-circuit", action="store_true", help="synthetic circuit, real activities")
    p.add_argument("-n", type=int, default=60, help="activities to pull from the API")
    p.add_argument("--streams", action="store_true", help="fetch HR/speed streams for real decoupling (slower)")
    p.add_argument("--dataset", help="neuPrint dataset, e.g. male-cns:v1.0 (auto-detected)")
    p.add_argument("--cache", default="circuit.npz")
    p.add_argument("--no-learning", action="store_true", help="don't train the mushroom body on kudos")
    p.add_argument("-o", "--out", default="results.json")
    a = p.parse_args()

    if a.csv:
        acts = strava.load_csv(a.csv)
    elif a.api:
        tok = os.environ.get("STRAVA_ACCESS_TOKEN") or p.error("set STRAVA_ACCESS_TOKEN")
        acts = strava.load_api(tok, a.n, a.streams)
    elif a.fit:
        from .fit import load_fit
        acts = load_fit(a.fit)
    elif a.demo:
        acts = strava.demo_activities()
    else:
        p.error("pick --fit, --csv, --api or --demo")
    if acts.empty or "speed_ms" not in acts:
        p.error("no usable activities found")
    acts = acts.dropna(subset=["speed_ms"]).sort_values("date").reset_index(drop=True)
    print(f"{len(acts)} activities")

    circuit = get_circuit(a.cache, a.demo or a.demo_circuit, a.dataset)
    print("circuit:", circuit.describe())

    raw = strava.raw_metrics(acts)
    res = Fly(circuit).sniff(acts, strava.features_from_raw(raw), train_on_kudos=not a.no_learning, raw=raw)
    res["meta"] = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "circuit": circuit.describe(),
        "activities_source": "fit" if a.fit else "csv" if a.csv else "api" if a.api else "demo",
        "glomeruli": [{"name": g, "feature": f, "odour": SMELLS[f][1]}
                      for f in SMELLS for g in [SMELLS[f][0]] if g in GLOMERULI],
    }
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)

    for x in sorted(res["activities"], key=lambda r: r["valence"]):
        print(f"{x['valence']:+.2f}  {x['verdict']:<14} {x['caption']}")
    print(f"\nwrote {a.out} – drop it into the Fly Sniff page")


if __name__ == "__main__":
    main()
