"""Bake trained checkpoints and the core into a page: python duel/build.py [out.html]"""
import json, pathlib, sys
here = pathlib.Path(__file__).resolve().parent
ck = json.load(open(here / "checkpoints.json"))
dump = json.dumps(ck, separators=(",", ":")).replace("</", "<\\/")
html = (here / "template.html").read_text().replace("/*__CORE__*/", (here / "core.js").read_text()).replace("__CK__", dump)
out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else here / "fly-duel.html")
out.write_text(html); print(f"wrote {out} ({len(html) // 1024} KB)")
