"""A1 plate layout: every part on the bed, 5 mm apart."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cadgen import build123d as bd

from lib.orient import on_bed
from print_plate.a1_plate import BED, LAYOUT

boxes = []
for label, make, rot, (x, y) in LAYOUT:
    bb = (bd.Pos(x + BED / 2, y + BED / 2, 0) * on_bed(make(), rot)).bounding_box()
    boxes.append((label, bb))
    inside = bb.min.X >= 5 and bb.min.Y >= 5 and bb.max.X <= BED - 5 and bb.max.Y <= BED - 5 and bb.max.Z <= BED - 6
    print(("PASS " if inside else "FAIL ") + f"{label}: x {bb.min.X:.0f}-{bb.max.X:.0f} y {bb.min.Y:.0f}-{bb.max.Y:.0f} h {bb.max.Z:.0f}")
ok = True
for (a, p), (b, q) in itertools.combinations(boxes, 2):
    gap = max(q.min.X - p.max.X, p.min.X - q.max.X, q.min.Y - p.max.Y, p.min.Y - q.max.Y)
    if gap < 5:
        ok = False
        print(f"FAIL {a} / {b} gap {gap:.1f} mm")
print("PLATE OK" if ok else "PLATE OVERLAP")
