"""Bake the flight circuit and core into a standalone page: python pilot/build.py [out.html]"""
import json, pathlib, sys
here = pathlib.Path(__file__).resolve().parent
data = json.load(open(here.parent / "data" / "male-cns-flight.json"))
dump = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
html = (here / "template.html").read_text().replace("/*__CORE__*/", (here / "core.js").read_text()).replace("__DATA__", dump)
out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else here / "fly-pilot.html")
out.write_text(html)
print(f"wrote {out} ({len(html) // 1024} KB)")
