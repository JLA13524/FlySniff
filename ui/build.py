"""Bake a results.json into a standalone page: python ui/build.py results.json fly-sniff.html"""
import json, pathlib, sys
src = sys.argv[1] if len(sys.argv) > 1 else "results.json"
out = sys.argv[2] if len(sys.argv) > 2 else "fly-sniff.html"
tpl = (pathlib.Path(__file__).parent / "template.html").read_text()
data = json.dumps(json.load(open(src)), separators=(",", ":")).replace("</", "<\\/")
pathlib.Path(out).write_text(tpl.replace("__DATA__", data))
print("wrote", out)
