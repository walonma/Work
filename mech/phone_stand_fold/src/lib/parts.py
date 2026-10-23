"""Part factories for the folding phone stand.

Base parts are built in world position. Arm parts are built in their own
link frame (origin on the pivot, folded orientation) and placed by
`lib.assembly_tree` with the joint angles.
"""
from __future__ import annotations

import math

from cadgen import build123d as bd

from lib.dims import *  # noqa: F403 - shared dimension table
from lib.geo import bent_xz, box, cyl_x, cyl_y, cyl_z, stadium, stadium_face

M3, M4, M5 = 3.4, 4.5, 5.5
PIVOT_HOLE = PIVOT_D + 0.2
PEM_M4 = 5.4                      # press-in nut hole
INSERT_M5 = 6.4                   # heat-set insert hole


def _cone_csk(x, y, z_top, d_hole, d_head, axis="z"):
    depth = (d_head - d_hole) / 2
    return bd.Pos(x, y, z_top - depth) * bd.Cone(d_hole / 2, d_head / 2, depth,
                                                 align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))


# ---------------------------------------------------------------- sheet metal

def base_plate():
    hw, hd = BASE_W / 2, BASE_D / 2
    z0 = BASE_TOP - T
    zb = PAD_T                       # flange bottom (hem underside)
    body = box(-hw, hw, -hd, hd, z0, BASE_TOP)
    g = 0.5
    # down flanges with inward hems carrying the silicone pads (front, back, right)
    body += box(-hw + g, hw - g, -hd, -hd + T, zb, z0)
    body += box(-hw + g, hw - g, hd - T, hd, zb, z0)
    body += box(hw - T, hw, -hd + T + g, hd - T - g, zb, z0)
    body += box(-hw + g + T, hw - g - T, -hd + T, -hd + T + HEM_W, zb, zb + T)
    body += box(-hw + g + T, hw - g - T, hd - T - HEM_W, hd - T, zb, zb + T)
    body += box(hw - T - HEM_W, hw - T, -hd + T + HEM_W + g, hd - T - HEM_W - g, zb, zb + T)
    # left edge: down, out along the desk, up into the fan plate
    xf = -hw + T / 2                 # tray flange centreline
    xp = -hw - TROUGH_GAP - T / 2    # fan plate centreline
    pts = [(-hw + 12.0, BASE_TOP - T / 2), (xf, BASE_TOP - T / 2), (xf, zb + T / 2),
           (xp, zb + T / 2), (xp, FAN_PLATE_TOP - T / 2)]
    body += bent_xz(pts, -hd + T + g, hd - T - g)
    # closed hem folded outward on the plate top edge (no sharp edge)
    xo = -hw - TROUGH_GAP - T
    hem = box(xo - T, xo, -hd + T + g, hd - T - g, FAN_PLATE_TOP - HEM_DROP, FAN_PLATE_TOP - T)
    hem += cyl_y(2 * T, -hd + T + g, hd - T - g, xo, FAN_PLATE_TOP - T)
    body += hem
    # shoulder ears, bent up out of the plate
    ex0 = LOW_W / 2 + 0.2
    hy = H1[1]
    for s in (-1, 1):
        xa, xb = sorted((s * ex0, s * (ex0 + T)))
        ear = box(xa, xb, hy - EAR_W / 2, hy + EAR_W / 2, z0, H1[2]) + cyl_x(EAR_W, xa, xb, hy, H1[2])
        body += ear
        body -= cyl_x(PIVOT_HOLE, xa - 1, xb + 1, hy, H1[2])
        wa, wb = sorted((s * (ex0 + T), s * (ex0 + T + H1[2] - z0 + EAR_W / 2)))
        body -= box(wa, wb, hy - EAR_W / 2 - 0.5, hy + EAR_W / 2 + 0.5, z0 - 1, BASE_TOP + 1)
    # PEM nuts for the cable reel
    body -= [cyl_z(PEM_M4, z0 - 1, BASE_TOP + 1, x, y) for x, y in REEL_HOLES]
    return body


def lower_arm():
    """Link frame: origin on the shoulder pivot, along +Y, open to +Z (folded)."""
    c = LOW_W / 2 - T / 2
    h = LOW_H / 2
    prof = bent_xz([(-c, h), (-c, -h + T / 2), (c, -h + T / 2), (c, h)], -h, L1 + h)
    ends = box(-LOW_W, LOW_W, 0, L1, -h, h) + cyl_x(LOW_H, -LOW_W, LOW_W) + cyl_x(LOW_H, -LOW_W, LOW_W, L1)
    arm = prof & ends
    return arm - [cyl_x(PIVOT_HOLE, -LOW_W, LOW_W, y) for y in (0.0, L1)]


def upper_arm():
    """Link frame: origin on the elbow pivot, along -Y to the head pivot, open to +Z."""
    c = UP_W / 2 - T / 2
    h = UP_H / 2
    prof = bent_xz([(-c, h), (-c, -h + T / 2), (c, -h + T / 2), (c, h)], -L2 - h, h)
    ends = box(-UP_W, UP_W, -L2, 0, -h, h) + cyl_x(UP_H, -UP_W, UP_W) + cyl_x(UP_H, -UP_W, UP_W, -L2)
    arm = prof & ends
    # web relief at the head end so the head tongue can swing
    wi = UP_W / 2 - T - 0.01                       # inner wall face, removes the bend radius too
    arm -= box(-wi, wi, -L2 - h - 1, -L2 + 12.0, -h - 1, -h + T + BEND_R + 0.2)
    return arm - [cyl_x(PIVOT_HOLE, -UP_W, UP_W, y) for y in (0.0, -L2)]


# ---------------------------------------------------------------- printed parts

def head():
    """Link frame: origin on the head pivot (folded orientation)."""
    tw = UP_W / 2 - T - 0.2
    z0, z1 = HEAD_PLATE_Z
    tongue = cyl_x(12.0, -tw, tw) + box(-tw, tw, -6.0, 6.0, 0.0, z0 + 0.5)
    plate = stadium(0.0, HEAD_AXIS_OFF / 2, HEAD_AXIS_OFF, HEAD_W / 2, z0, z1)
    body = tongue + plate
    body -= cyl_x(PIVOT_HOLE, -tw - 1, tw + 1)
    body -= cyl_z(INSERT_M5, z0 - 1, z1 + 1, 0.0, HEAD_AXIS_OFF)
    return body


def cradle_place():
    """Head-frame placement of the cradle (face up when folded, phone top toward +Y)."""
    return bd.Pos(0.0, HEAD_AXIS_OFF, HEAD_PLATE_Z[1] - 16.0) * bd.Rot(-90, 0, 0)


def cradle_body():
    """Cradle frame: phone face at y=FACE_Y (normal -Y), rotation axis along Y."""
    hw = PLATE_HW
    yb = FACE_Y + 6.0
    yh = FACE_Y + 20.0
    body = bd.extrude(bd.Plane.XZ.offset(-FACE_Y) * (bd.Pos(0, (CRADLE_ZB + 4 + CRADLE_ZT) / 2) *
                      bd.RectangleRounded(2 * hw, CRADLE_ZT - CRADLE_ZB - 4, 6.0)), amount=-6.0)
    bb = body.bounding_box()
    body = bd.Pos(0, FACE_Y - bb.min.Y, 0) * body
    ly = FACE_Y - PHONE_MAX_T - 2.0
    body += box(-hw, hw, ly, FACE_Y + 0.5, CRADLE_ZB, CRADLE_ZB + 4.0)         # ledge
    body += box(-hw, hw, ly, ly + 3.0, CRADLE_ZB, CRADLE_ZB + 8.0)             # front lip
    body -= box(-8.0, 8.0, ly - 1, FACE_Y - 4.0, CRADLE_ZB - 1, CRADLE_ZB + 9)  # charging-cable notch
    body += box(-hw, hw, yb - 0.5, yh, -18.0, 18.0)                            # channel housing
    body += cyl_y(36.0, yh - 0.5, yh + 6.0)                                    # rotation boss
    for z0, z1 in (CH_UPPER_Z, CH_LOWER_Z):
        body -= box(-hw - 1, hw + 1, CH_Y[0], CH_Y[1], z0, z1)
    yc = (CH_Y[0] + CH_Y[1]) / 2
    body -= cyl_z(SPRING_PIN_D, 0.0, 19.0, -SPRING_PIN_X, yc)
    body -= cyl_z(SPRING_PIN_D, -19.0, 0.0, SPRING_PIN_X, yc)
    # M5 countersunk screw from the phone face into the head insert
    body -= cyl_y(M5, FACE_Y - 1, yh + 7)
    body -= bd.Pos(0, FACE_Y, 0) * bd.Cone(5.4, 2.75, 2.7, rotation=(-90, 0, 0),
                                          align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
    return body


def jaw(side, opening=JAW_AS_BUILT):
    """Cradle frame. side=+1 right jaw (upper channel), -1 left jaw (lower channel)."""
    g = opening / 2
    ly = FACE_Y - PHONE_MAX_T - 4.0
    wall = box(g, g + 4.0, ly, FACE_Y + 20.0, -40.0, 20.0)
    lip = box(g - 3.0, g, ly, ly + 4.0, -40.0, 20.0)
    z0, z1 = CH_UPPER_Z if side > 0 else CH_LOWER_Z
    c = (CH - BAR) / 2
    bar = box(g - BAR_LEN, g + 0.5, CH_Y[0] + c, CH_Y[1] - c, z0 + c, z1 - c)
    bar -= cyl_z(2.2, z0 - 1, z1 + 1, g - BAR_LEN + 3.0, (CH_Y[0] + CH_Y[1]) / 2)
    part = wall + lip + bar
    part = part.fillet(1.0, part.edges().filter_by(bd.Axis.Z).filter_by_position(bd.Axis.X, g + 3.9, g + 4.1))
    return part if side > 0 else bd.mirror(part, about=bd.Plane.YZ)


def cable_reel():
    x0, x1 = REEL_X
    y0, y1 = REEL_Y
    z0 = BASE_TOP
    zp = z0 + REEL_PLATE_T
    plate = bd.Pos((x0 + x1) / 2, (y0 + y1) / 2, z0) * bd.extrude(bd.RectangleRounded(x1 - x0, y1 - y0, 8.0), amount=REEL_PLATE_T)
    bx, by = BOBBIN_C
    core_r, flange_r, straight = 12.0, 22.0, 30.0
    z_core = zp + 35.0
    z_fl = z_core + 10.0                 # 45 deg skirt under the flange: no supports
    body = plate + stadium(bx, by, straight, core_r, zp - 0.5, z_core + 0.1)
    body += bd.loft([stadium_face(bx, by, straight, core_r, z_core), stadium_face(bx, by, straight, flange_r, z_fl)])
    body += stadium(bx, by, straight, flange_r, z_fl - 0.01, z_fl + 3.0)
    for ys in (-1, 1):
        yn = by + ys * (straight / 2 + core_r)
        body -= box(bx - 2.3, bx + 2.3, min(yn, yn + ys * 20), max(yn, yn + ys * 20), z_core - 1, z_fl + 4)
    zd = z_fl + 3.0
    body += bd.Pos(bx, by, zd - 0.01) * bd.extrude(bd.RectangleRounded(16.0, 24.0, 3.0), amount=12.0)
    body -= box(bx - 1.5, bx + 1.5, by - 4.4, by + 4.4, zd + 12.0 - 8.0, zd + 13.0)   # USB-C dummy port
    body -= box(bx - 4.0, bx + 4.0, by - 7.0, by + 7.0, zd + 12.0 - 1.5, zd + 13.0)   # overmold seat
    cx, cy = CHARGER_C
    ix, iy, depth = CHARGER_IN
    w = 2.4
    pocket = bd.Pos(cx, cy, zp - 0.5) * bd.extrude(bd.RectangleRounded(ix + 2 * w, iy + 2 * w, 4.0), amount=depth + 0.5)
    pocket -= bd.Pos(cx, cy, zp) * bd.extrude(bd.RectangleRounded(ix, iy, 2.0), amount=depth + 1)
    pocket -= box(cx - 6.0, cx + 6.0, cy - iy / 2 - w - 1, cy - iy / 2 + 1, zp + depth - 18.0, zp + depth + 1)
    body += pocket
    for x, y in REEL_HOLES:
        body -= cyl_z(M4, z0 - 1, zp + 1, x, y)
        body -= _cone_csk(x, y, zp + 0.01, M4, 8.6)
    return body


# ---------------------------------------------------------------- purchased / reference

def silicone_pad():
    """Die-cut 2 mm silicone strips under the hems and the fan trough."""
    hw, hd = BASE_W / 2, BASE_D / 2
    g = 0.5
    pads = [box(-hw + g + T, hw - g - T, -hd + T, -hd + T + HEM_W, 0.0, PAD_T),
            box(-hw + g + T, hw - g - T, hd - T - HEM_W, hd - T, 0.0, PAD_T),
            box(hw - T - HEM_W, hw - T, -hd + T + HEM_W + g, hd - T - HEM_W - g, 0.0, PAD_T),
            box(-hw - TROUGH_GAP, -hw, -hd + T + g, hd - T - g, 0.0, PAD_T)]
    return bd.Compound(children=pads)


def pivot_bolt(width, spacer=None):
    """Low-head M5 friction pivot (sex bolt + disc springs) along X, centred on the pivot.

    `spacer=(x_in, x_out)` adds the friction washer stacks filling |x| in that range.
    """
    w = width / 2
    bolt = cyl_x(PIVOT_D, -w, w) + cyl_x(9.0, -w - 3.0, -w) + cyl_x(9.0, w, w + 3.0)
    if spacer:
        a, b = spacer
        bolt += cyl_x(10.0, a, b) + cyl_x(10.0, -b, -a)
    return bolt


def phone_ref():
    return box(-38.0, 38.0, FACE_Y - 9.0, FACE_Y, CRADLE_ZB + 4.0, CRADLE_ZB + 4.0 + 160.0)


def fan_ref():
    """DIKE DUF201 (approximate): spring clip slid down over the fan plate, head on top."""
    xp0 = -BASE_W / 2 - TROUGH_GAP - T       # plate inner face
    xp1 = xp0 - T                            # plate outer face (hem side is thicker)
    jl, jw, jd = 60.0, 64.0, FAN_JAW_T       # jaw length (Z), width (Y), thickness (X)
    inner = box(xp0, xp0 + jd, -jw / 2, jw / 2, PAD_T + T, PAD_T + T + jl)
    outer = box(xp1 - 2 * T - jd, xp1 - 2 * T, -jw / 2, jw / 2, 0.0, jl)
    hinge = box(xp1 - 2 * T - jd, xp0 + jd, -jw / 2, jw / 2, jl, jl + 18.0)
    neck = cyl_z(20, jl + 18.0, jl + 40.0, (xp0 + xp1) / 2)
    fc = ((xp0 + xp1) / 2, 0.0, jl + 40.0 + 75.0)
    ring = cyl_y(150, -25, 25, fc[0], fc[2]) - cyl_y(138, -26, 26, fc[0], fc[2])
    hubp = cyl_y(40, -10, 20, fc[0], fc[2])
    return inner + outer + hinge + neck + ring + hubp
