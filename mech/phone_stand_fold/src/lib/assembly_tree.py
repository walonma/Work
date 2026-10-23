"""Labelled assembly tree at any joint pose (absolute angles, see lib.dims)."""
from __future__ import annotations

from cadgen import build123d as bd, srgb

from lib import parts
from lib.dims import *  # noqa: F403

STEEL = srgb("#2F3338")     # powder-coated SPCC
ARM = srgb("#9AA1A8")       # arms, satin silver powder coat
PRINT = srgb("#F08A24")     # PETG
BLACK = srgb("#1E1E1E")
SILICONE = srgb("#3C3C3C")
REF = srgb("#9DB4C8", 0.35)


def _p(shape, label, color):
    shape.label = label
    shape.color = color
    return shape


def link_locations(shoulder=SHOULDER, elbow=ELBOW, head=HEAD):
    c = chain(shoulder, elbow, head)
    return (bd.Pos(*c["h1"]) * bd.Rot(c["lower"], 0, 0),
            bd.Pos(*c["h2"]) * bd.Rot(c["upper"], 0, 0),
            bd.Pos(*c["h3"]) * bd.Rot(c["head"], 0, 0))


def build(shoulder=SHOULDER, elbow=ELBOW, head=HEAD, rotate=0.0, opening=JAW_AS_BUILT,
          explode=0.0, refs=True):
    e = explode
    lo, up, hd = link_locations(shoulder, elbow, head)
    cp = parts.cradle_place() * bd.Rot(0, rotate, 0)
    rot_children = [
        _p(cp * (bd.Pos(0, -0.9 * e, 0) * parts.cradle_body()), "cradle_body", PRINT),
        _p(cp * (bd.Pos(-0.6 * e, -0.6 * e, 0) * parts.jaw(-1, opening)), "jaw_left", PRINT),
        _p(cp * (bd.Pos(0.6 * e, -0.6 * e, 0) * parts.jaw(+1, opening)), "jaw_right", PRINT),
    ]
    if refs:
        rot_children.append(_p(cp * parts.phone_ref(), "phone_ref", REF))
    rotate_stage = bd.Compound(children=[hd * c for c in rot_children], label="rotate_stage")
    head_part = _p(hd * (bd.Pos(0, 0, 0.5 * e) * parts.head()), "head", PRINT)
    head_stage = bd.Compound(children=[head_part, rotate_stage], label="head_stage")
    upper_stage = bd.Compound(children=[
        _p(up * parts.upper_arm(), "upper_arm", ARM),
        _p(hd * parts.pivot_bolt(UP_W), "head_pivot", BLACK),
        bd.Pos(0, 0, 0.6 * e) * head_stage], label="upper_stage")
    lower_stage = bd.Compound(children=[
        _p(lo * parts.lower_arm(), "lower_arm", ARM),
        _p(up * parts.pivot_bolt(LOW_W, spacer=(UP_W / 2, LOW_W / 2 - T)), "elbow_pivot", BLACK),
        bd.Pos(0, 0, 0.6 * e) * upper_stage], label="lower_stage")
    children = [
        _p(bd.Pos(0, 0, -0.8 * e) * parts.silicone_pad(), "silicone_pad", SILICONE),
        _p(parts.base_plate(), "base_plate", STEEL),
        _p(bd.Pos(0.6 * e, 0, 0.6 * e) * parts.cable_reel(), "cable_reel", PRINT),
        _p(lo * parts.pivot_bolt(LOW_W + 2 * T + 0.4), "shoulder_pivot", BLACK),
        bd.Pos(0, 0, 0.8 * e) * lower_stage,
    ]
    if refs:
        children.append(_p(parts.fan_ref(), "fan_ref", REF))
    return bd.Compound(children=children, label="phone_stand_fold")
