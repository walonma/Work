"""Place a part for manufacture: optional rotation, then bbox min on the origin plane."""
from __future__ import annotations

from cadgen import build123d as bd


def on_bed(shape, rot=(0, 0, 0)):
    s = bd.Rot(*rot) * shape
    bb = s.bounding_box()
    return bd.Pos(-(bb.min.X + bb.max.X) / 2, -(bb.min.Y + bb.max.Y) / 2, -bb.min.Z) * s
