"""Inline viewer/parts.json into viewer/template.html -> viewer/phone_stand_viewer.html."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
data = (ROOT / "viewer" / "parts.json").read_text().replace("</", "<\\/")
html = (ROOT / "viewer" / "template.html").read_text().replace("/*DATA*/", data)
out = ROOT / "viewer" / "phone_stand_viewer.html"
out.write_text(html)
print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.2f} MB)")
