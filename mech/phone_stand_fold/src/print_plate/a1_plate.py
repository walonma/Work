"""Bambu Lab A1 (256 x 256) plate layout for the five printed parts (see combine_3mf.py)."""
from cadgen import build123d as bd

from lib import parts
from lib.orient import on_bed

BED = 256.0
# (label, factory, print rotation, plate centre XY)
LAYOUT = [
    ("cable_reel", lambda: parts.cable_reel(), (0, 0, 0), (-88.0, 45.0)),
    ("cradle_body", lambda: parts.cradle_body(), (0, 0, 0), (-10.0, 85.0)),
    ("head", lambda: parts.head(), (180, 0, 0), (50.0, 70.0)),
    ("jaw_left", lambda: parts.jaw(-1), (0, -90, 0), (-55.0, -70.0)),
    ("jaw_right", lambda: parts.jaw(+1), (0, 90, 0), (25.0, -70.0)),
]


def a1_plate():
    """Preview of the whole plate as one shape (the 3MF is written by combine_3mf.py)."""
    placed = []
    for label, make, rot, (x, y) in LAYOUT:
        p = bd.Pos(x + BED / 2, y + BED / 2, 0) * on_bed(make(), rot)
        p.label = label
        placed.append(p)
    return bd.Compound(children=placed, label="a1_plate")
