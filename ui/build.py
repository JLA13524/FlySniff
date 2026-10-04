"""Bake results.json (and the circuit it ran on) into a standalone page.

    python ui/build.py [results.json] [fly-sniff.html] [--cache circuit.npz]

The circuit goes in so the page can score .fit files you upload in the browser.
"""
import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flysniff.connectome import Circuit, demo_circuit  # noqa: E402
from flysniff.webexport import circuit_json  # noqa: E402


ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="?", default="results.json")
ap.add_argument("out", nargs="?", default="fly-sniff.html")
ap.add_argument("--cache", default="circuit.npz")
a = ap.parse_args()

res = json.load(open(a.results))
src = res.get("meta", {}).get("circuit", {}).get("source", "demo")
c = demo_circuit() if src == "demo" or not pathlib.Path(a.cache).exists() else Circuit.load(a.cache)
if c.source != src:
    print(f"warning: results came from {src} but embedding {c.source}")

ui = pathlib.Path(__file__).parent
tpl = (ui / "template.html").read_text().replace("/*__FITFLY__*/", (ui / "fitfly.js").read_text())
dump = lambda o: json.dumps(o, separators=(",", ":")).replace("</", "<\\/")
html = tpl.replace("__DATA__", dump(res)).replace("__CIRCUIT__", dump(circuit_json(c)))
pathlib.Path(a.out).write_text(html)
print(f"wrote {a.out} ({len(html) // 1024} KB)")
