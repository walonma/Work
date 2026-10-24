"""Inline viewer/data.json into viewer/template.html -> viewer/microduck_viewer.html."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = (ROOT / "viewer" / "data.json").read_text().replace("</", "<\\/")
html = (ROOT / "viewer" / "template.html").read_text().replace("/*DATA*/", data)
out = ROOT / "viewer" / "microduck_viewer.html"
out.write_text(html)
print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.2f} MB)")
