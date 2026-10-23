"""Merge the per-part 3MF files into one A1 plate, one object per part.

Run after the printed-part models: reads 3MF/<part>.3mf (print orientation,
centred on XY, resting on Z=0) and writes 3MF/A1_plate_all_parts.3mf with
each part as its own object placed per `a1_plate.LAYOUT`.
"""
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from print_plate.a1_plate import BED, LAYOUT  # noqa: E402

NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"


def mesh_xml(path):
    model = zipfile.ZipFile(path).read("3D/3dmodel.model").decode()
    return re.search(r"<mesh>.*</mesh>", model, re.S).group(0)


def main():
    objects, items = [], []
    for i, (label, _make, _rot, (x, y)) in enumerate(LAYOUT, start=1):
        objects.append(f'<object id="{i}" type="model" name="{label}">{mesh_xml(ROOT / "3MF" / f"{label}.3mf")}</object>')
        items.append(f'<item objectid="{i}" transform="1 0 0 0 1 0 0 0 1 {x + BED / 2:.3f} {y + BED / 2:.3f} 0"/>')
    model = (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" xmlns="{NS}">'
             f'<resources>{"".join(objects)}</resources><build>{"".join(items)}</build></model>')
    types = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    out = ROOT / "3MF" / "A1_plate_all_parts.3mf"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model)
    print(f"wrote {out.relative_to(ROOT)} ({len(LAYOUT)} objects)")


if __name__ == "__main__":
    main()
