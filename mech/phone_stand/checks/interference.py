"""Interference check across the motion range (height, tilt, rotate, jaw)."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cadgen import build123d as bd
from cadgen.geometry import overlap_volume

from lib import parts
from lib.dims import *  # noqa: F403

TOL = 1.0  # mm^3


def solid(s):
    sol = s.solids()
    return sol[0] if len(sol) == 1 else bd.Solid.fuse(*sol) if hasattr(bd.Solid, "fuse") else sol[0]


fixed = {
    "base_plate": parts.base_plate(), "weight_plate": parts.weight_plate(),
    "column_bracket": parts.column_bracket(), "tube_foot": parts.tube_foot(), "outer_tube": parts.outer_tube(),
    "column_collar": parts.column_collar(), "fan_bracket": parts.fan_bracket(),
    "fan_pad": parts.fan_pad(), "cable_reel": parts.cable_reel(),
    "collar_knob": parts.star_knob(+1, (CX + 10.5, CY, COLLAR_KNOB_Z), 12.0),
}
fixed.update({f"foot{i}": parts.rubber_foot(x, y) for i, (x, y) in enumerate(FOOT_XY)})


def head(height, tilt, rot, opening):
    H = bd.Pos(CX, CY, AXIS_Z_MIN + height)
    T_ = bd.Rot(-tilt, 0, 0)
    R_ = bd.Rot(0, rot, 0)
    out = {
        "inner_tube": parts.inner_tube(INNER_TOP_MIN + height),
        "head_yoke": H * parts.head_yoke(),
        "tilt_knob": H * parts.star_knob(+1, (17.2, 0, 0), 4.0),
        "tilt_hub": H * T_ * parts.tilt_hub(),
    }
    for k, s in (("cradle_body", parts.cradle_body()), ("jaw_right", parts.jaw(+1, opening)),
                 ("jaw_left", parts.jaw(-1, opening))):
        out[k] = H * T_ * R_ * s
    return out


def pairs_overlap(shapes, allowed):
    bad = []
    for (a, sa), (b, sb) in itertools.combinations(shapes.items(), 2):
        if {a, b} in allowed:
            continue
        v = sum(overlap_volume(x, y) for x in sa.solids() for y in sb.solids())
        if v > TOL:
            bad.append((a, b, round(v, 1)))
    return bad


# Intended contacts that are modelled as touching/overlapping fasteners
ALLOWED = [{"collar_knob", "column_collar"}, {"tilt_knob", "head_yoke"}, {"tilt_knob", "tilt_hub"},
           {"collar_knob", "inner_tube"}]

worst = 0
for h, t, r, g in itertools.product((0, TRAVEL), (TILT_LIMITS[0], 0, 35, TILT_LIMITS[1]), (0, 90, -90), (JAW_MIN, JAW_MAX)):
    shapes = dict(fixed)
    shapes.update(head(h, t, r, g))
    bad = pairs_overlap(shapes, ALLOWED)
    tag = f"h={h:>4} tilt={t:>5} rot={r:>4} jaw={g}"
    print(tag, "OK" if not bad else bad)
    worst += bool(bad)
print("FAIL" if worst else "PASS", worst, "poses with interference")
