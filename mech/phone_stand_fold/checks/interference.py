"""Interference check: folded (as packed) and a sweep of usage poses."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cadgen.geometry import overlap_volume

from lib.assembly_tree import build
from lib.dims import ELBOW_RANGE, HEAD_RANGE, JAW_MAX, JAW_MIN, SHOULDER_RANGE

TOL = 1.0   # mm^3
PIVOTS = {
    "shoulder_pivot": {"base_plate", "lower_arm"},
    "elbow_pivot": {"lower_arm", "upper_arm"},  # washers bear on both walls
    "head_pivot": {"upper_arm", "head"},
}


def leaves(shape, out):
    if shape.label and not shape.children:
        out[shape.label] = shape
    for c in shape.children:
        leaves(c, out)
    return out


def clashes(**pose):
    parts = leaves(build(refs=False, **pose), {})
    bad = []
    for (a, sa), (b, sb) in itertools.combinations(parts.items(), 2):
        if b in PIVOTS.get(a, ()) or a in PIVOTS.get(b, ()):
            continue
        v = sum(overlap_volume(x, y) for x in sa.solids() for y in sb.solids())
        if v > TOL:
            bad.append((a, b, round(v, 1)))
    return bad


def poses():
    yield "folded", dict(shoulder=0.0, elbow=0.0, head=0.0)
    for sh, psi, tilt in itertools.product((20, 55, 95), (-60, -20, 20, 60), (0, 30, 60)):
        el = sh - psi                      # upper arm direction psi from the front horizontal
        hd = 90 - tilt - psi               # phone faces the user, leaned back by `tilt`
        if not (SHOULDER_RANGE[0] <= sh <= SHOULDER_RANGE[1] and ELBOW_RANGE[0] <= el <= ELBOW_RANGE[1]
                and HEAD_RANGE[0] <= hd <= HEAD_RANGE[1]):
            continue
        for rot, g in ((0, JAW_MAX), (90, JAW_MIN)):
            yield f"sh={sh} psi={psi} tilt={tilt} rot={rot} jaw={g}", dict(
                shoulder=sh, elbow=el, head=hd, rotate=rot, opening=g)


if __name__ == "__main__":
    results = [(n, clashes(**p)) for n, p in poses()]
    for n, bad in results:
        print(n, "OK" if not bad else bad)
    nbad = sum(bool(b) for _, b in results)
    print(f"{len(results)} poses, {nbad} with interference")
