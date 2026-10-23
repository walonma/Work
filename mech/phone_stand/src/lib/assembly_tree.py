"""Builds the labelled assembly tree (shared by the assembly and exploded views)."""
from __future__ import annotations

from cadgen import build123d as bd, srgb

from lib import parts
from lib.dims import *  # noqa: F403

STEEL = srgb("#3B4046")     # powder-coated SPCC
RAW = srgb("#80868D")       # ballast plate / tubes
PRINT = srgb("#F08A24")     # PETG
BLACK = srgb("#1E1E1E")
REF = srgb("#9DB4C8", 0.35)


def _p(shape, label, color):
    shape.label = label
    shape.color = color
    return shape


def head_offset():
    return bd.Pos(CX, CY, AXIS_Z_MIN)


def build(explode=0.0):
    """explode > 0 spreads parts apart along their assembly direction."""
    e = explode
    H = head_offset()
    feet = [_p(bd.Pos(0, 0, -1.6 * e) * parts.rubber_foot(x, y), f"rubber_foot:{i}", BLACK)
            for i, (x, y) in enumerate(FOOT_XY)]
    fixed = [
        _p(bd.Pos(0, 0, -1.0 * e) * parts.weight_plate(), "weight_plate", RAW),
        _p(parts.base_plate(), "base_plate", STEEL),
        _p(bd.Pos(0, 0, 0.6 * e) * parts.column_bracket(), "column_bracket", STEEL),
        _p(bd.Pos(0, 0, 0.9 * e) * parts.tube_foot(), "tube_foot", PRINT),
        _p(bd.Pos(0, 0, 1.2 * e) * parts.outer_tube(), "outer_tube", RAW),
        _p(bd.Pos(0, 0, 1.8 * e) * parts.column_collar(), "column_collar", PRINT),
        _p(bd.Pos(0.8 * e, 0, 1.8 * e) * parts.star_knob(+1, (CX + 10.5, CY, COLLAR_KNOB_Z), 12.0), "collar_knob", BLACK),
        _p(bd.Pos(-0.6 * e, 0, 0.6 * e) * parts.fan_bracket(), "fan_bracket", STEEL),
        _p(bd.Pos(-1.2 * e, 0, 1.4 * e) * parts.fan_pad(), "fan_pad", PRINT),
        _p(bd.Pos(0.6 * e, 0, 0.6 * e) * parts.cable_reel(), "cable_reel", PRINT),
    ]
    jaw_r = _p(bd.Pos(0.6 * e, -0.6 * e, 0) * parts.jaw(+1), "jaw_right", PRINT)
    jaw_l = _p(bd.Pos(-0.6 * e, -0.6 * e, 0) * parts.jaw(-1), "jaw_left", PRINT)
    cradle = _p(bd.Pos(0, -0.9 * e, 0) * parts.cradle_body(), "cradle_body", PRINT)
    rotate_children = [cradle, jaw_l, jaw_r]
    if e == 0:
        rotate_children.append(_p(parts.phone_ref(), "phone_ref", REF))
    rotate_stage = bd.Compound(children=[H * c for c in rotate_children], label="rotate_stage")
    hub = _p(H * (bd.Pos(0, -0.3 * e, 0) * parts.tilt_hub()), "tilt_hub", PRINT)
    tilt_stage = bd.Compound(children=[hub, rotate_stage], label="tilt_stage")
    yoke = _p(H * (bd.Pos(0, 0, -0.3 * e) * parts.head_yoke()), "head_yoke", PRINT)
    tilt_knob = _p(H * (bd.Pos(0.8 * e, 0, 0) * parts.star_knob(+1, (12.2 + 5.0, 0, 0), 4.0)), "tilt_knob", BLACK)
    inner = _p(bd.Pos(0, 0, 0.6 * e) * parts.inner_tube(), "inner_tube", RAW)
    height_stage = bd.Compound(children=[inner, yoke, tilt_knob, tilt_stage],
                               label="height_stage")
    if e:
        height_stage = bd.Pos(0, 0, 2.4 * e) * height_stage
    children = feet + fixed + [height_stage]
    if e == 0:
        children.append(_p(parts.fan_ref(), "fan_ref", REF))
    return bd.Compound(children=children, label="phone_stand")
