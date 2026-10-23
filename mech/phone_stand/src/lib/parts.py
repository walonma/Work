"""Part factories. Each returns the part in assembly position unless noted.

Head parts (yoke, hub, cradle, jaws) are built in the head frame: origin on
the tilt axis, axes parallel to the world frame. The assembly places them.
"""
from __future__ import annotations

import math

from cadgen import build123d as bd

from lib.dims import *  # noqa: F403 - shared dimension table
from lib.geo import bent_xz, box, cyl_x, cyl_y, cyl_z, stadium, stadium_face

M4, M5, M6 = 4.5, 5.5, 6.6        # clearance holes
M4_TAP, M5_TAP = 3.3, 4.2         # tap drills in the ballast plate
INSERT_M4, INSERT_M6 = 5.6, 8.0   # heat-set insert holes


# ---------------------------------------------------------------- sheet metal

def base_plate():
    hw, hd = BASE_W / 2, BASE_D / 2
    z0 = BASE_TOP - T
    body = box(-hw, hw, -hd, hd, z0, BASE_TOP)
    fz0 = BASE_TOP - BASE_FLANGE
    gap = 0.5   # corner relief between adjacent flanges
    body += box(-hw + gap, hw - gap, -hd, -hd + T, fz0, z0)
    body += box(-hw + gap, hw - gap, hd - T, hd, fz0, z0)
    body += box(-hw, -hw + T, -hd + T + gap, hd - T - gap, fz0, z0)
    body += box(hw - T, hw, -hd + T + gap, hd - T - gap, fz0, z0)
    holes = [cyl_z(M5, z0 - 1, BASE_TOP + 1, x, y) for x, y in COLUMN_HOLES + FAN_HOLES]
    holes += [cyl_z(M4, z0 - 1, BASE_TOP + 1, x, y) for x, y in REEL_HOLES]
    for x, y in BASE_SCREWS:   # 90 deg countersink for M5 flat head
        holes.append(cyl_z(M5, z0 - 1, BASE_TOP + 1, x, y))
        holes.append(bd.Pos(x, y, BASE_TOP - 2.0) * bd.Cone(M5 / 2, M5 / 2 + 2.0, 2.0,
                     align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)))
    return body - holes


def weight_plate():
    hw, hd = WEIGHT_W / 2, WEIGHT_D / 2
    body = box(-hw, hw, -hd, hd, WEIGHT_Z0, WEIGHT_Z0 + WEIGHT_T)
    z0, z1 = WEIGHT_Z0 - 1, WEIGHT_Z0 + WEIGHT_T + 1
    holes = [cyl_z(M5_TAP, z0, z1, x, y) for x, y in COLUMN_HOLES + FAN_HOLES + BASE_SCREWS]
    holes += [cyl_z(M4_TAP, z0, z1, x, y) for x, y in REEL_HOLES]
    return body - holes


def column_bracket():
    xi = OUTER_A / 2 + 0.3 + T / 2       # wall centreline from the column axis
    z_web = BASE_TOP + T / 2
    pts = [(CX - xi, BASE_TOP + BRK_WALL_H), (CX - xi, z_web), (CX + xi, z_web), (CX + xi, BASE_TOP + BRK_WALL_H)]
    body = bent_xz(pts, CY - BRK_WALL_D / 2, CY + BRK_WALL_D / 2)
    xo = xi + T / 2
    body += box(CX - xo + 3, CX + xo - 3, CY - BRK_WEB_D / 2, CY + BRK_WEB_D / 2, BASE_TOP, BASE_TOP + T)
    for y0, y1 in ((CY - BRK_WEB_D / 2, CY - BRK_WALL_D / 2), (CY + BRK_WALL_D / 2, CY + BRK_WEB_D / 2)):
        body += box(CX - xo, CX + xo, y0, y1, BASE_TOP, BASE_TOP + T)
    holes = [cyl_z(M5, BASE_TOP - 1, BASE_TOP + T + 1, x, y) for x, y in COLUMN_HOLES]
    holes += [cyl_x(M5, CX - xo - 1, CX + xo + 1, CY, z) for z in BRK_BOLT_Z]
    return body - holes


def fan_bracket():
    zf = BASE_TOP + T / 2
    zt = FAN_SHELF_TOP - T / 2
    pts = [(FAN_FLANGE_X_IN, zf), (FAN_WEB_X, zf), (FAN_WEB_X, zt), (FAN_SHELF_X_OUT, zt)]
    hw = FAN_W / 2
    body = bent_xz(pts, -hw, hw)
    holes = [cyl_z(M5, BASE_TOP - 1, BASE_TOP + T + 1, x, y) for x, y in FAN_HOLES]
    holes += [cyl_z(M4, FAN_SHELF_TOP - T - 1, FAN_SHELF_TOP + 1, x, y) for x in (-135.0, -182.0) for y in (-44.0, 44.0)]
    holes += [cyl_x(M4, FAN_WEB_X - 2, FAN_WEB_X + 2, y, 65.0) for y in (-45.0, 45.0)]
    return body - holes


def square_tube(a, wall, x, y, z0, length, r=None):
    r = wall * 1.5 if r is None else r
    outer = bd.Pos(x, y, z0) * bd.extrude(bd.RectangleRounded(a, a, r), amount=length)
    inner = bd.Pos(x, y, z0 - 1) * bd.extrude(bd.RectangleRounded(a - 2 * wall, a - 2 * wall, max(r - wall, 0.3)), amount=length + 2)
    return outer - inner


def outer_tube():
    tube = square_tube(OUTER_A, OUTER_WALL, CX, CY, OUTER_Z0, OUTER_LEN)
    return tube - [cyl_x(M5, CX - 20, CX + 20, CY, z) for z in BRK_BOLT_Z]


def inner_tube(z_top=INNER_TOP_MIN):
    tube = square_tube(INNER_A, INNER_WALL, CX, CY, z_top - INNER_LEN, INNER_LEN)
    return tube - cyl_y(M4, CY - 15, CY + 15, CX, z_top - 12.0)


# ---------------------------------------------------------------- printed parts

def tube_foot():
    """Plug under the outer tube; its flange is narrower than the tube so it clears the bends."""
    z0 = BASE_TOP + T
    flange = bd.Pos(CX, CY, z0) * bd.extrude(bd.RectangleRounded(21.0, 21.0, 2.0), amount=FOOT_PLUG_T)
    plug_a = OUTER_A - 2 * OUTER_WALL - 0.4
    plug = bd.Pos(CX, CY, z0 + FOOT_PLUG_T - 0.01) * bd.extrude(bd.RectangleRounded(plug_a, plug_a, 1.0), amount=5.0)
    return flange + plug - cyl_z(8.0, z0 + 1.2, z0 + 20, CX, CY)


def column_collar():
    c = COLLAR_SIDE
    body = bd.Pos(CX, CY, COLLAR_Z0) * bd.extrude(bd.RectangleRounded(c, c, 5.0), amount=COLLAR_H)
    sock = OUTER_A + 0.4
    body -= bd.Pos(CX, CY, COLLAR_Z0 - 1) * bd.extrude(bd.RectangleRounded(sock, sock, 2.4), amount=OUTER_TOP - COLLAR_Z0 + 1)
    bore = INNER_A + 0.6
    body -= box(CX - bore / 2, CX + bore / 2, CY - bore / 2, CY + bore / 2, COLLAR_Z0, COLLAR_TOP + 1)
    # M6 locking screw from +X, captive nut dropped in from the top
    body -= cyl_x(M6, CX, CX + c, CY, COLLAR_KNOB_Z)
    wall_mid = CX + (bore / 2 + c / 2) / 2
    body -= box(wall_mid - 2.8, wall_mid + 2.8, CY - 5.2, CY + 5.2, COLLAR_KNOB_Z - 6.0, COLLAR_TOP + 1)
    return body


def head_yoke():
    """Head frame. Socket over the inner tube, two ears carrying the tilt bolt."""
    sx, sy = 17.2, 14.0
    top = -AXIS_ABOVE_TUBE
    body = box(-sx, sx, -sy, sy, top - 25.0, -20.0)
    body -= box(-(INNER_A + 0.6) / 2, (INNER_A + 0.6) / 2, -(INNER_A + 0.6) / 2, (INNER_A + 0.6) / 2, top - 26.0, top)
    for x0, x1 in ((-sx, -12.2), (12.2, sx)):
        ear = box(x0, x1, -sy, sy, -20.0, 0.0) + cyl_x(2 * sy, x0, x1)
        body += ear
    body -= cyl_x(6.4, -sx - 1, sx + 1)
    body -= cyl_y(M4, -sy - 1, sy + 1, 0, top - 12.0)
    return body


def tilt_hub():
    body = cyl_x(30.0, -12.0, 12.0) + cyl_y(20.0, FACE_Y + 26.0, 0.0)
    body -= cyl_x(6.4, -13.0, 13.0)
    body -= cyl_y(INSERT_M6, FACE_Y + 25.0, FACE_Y + 38.0)    # M6 heat-set insert, 12 deep
    return body


def cradle_body():
    hw = PLATE_HW
    yb = FACE_Y + 6.0            # back of the plate
    yh = FACE_Y + 20.0           # back of the channel housing
    body = bd.extrude(bd.Plane.XZ.offset(-FACE_Y) * (bd.Pos(0, (CRADLE_ZB + 4 + CRADLE_ZT) / 2) *
                      bd.RectangleRounded(2 * hw, CRADLE_ZT - CRADLE_ZB - 4, 6.0)), amount=-6.0)
    body = bd.Pos(0, 0, 0) * body
    bb = body.bounding_box()
    body = bd.Pos(0, FACE_Y - bb.min.Y, 0) * body
    ly = FACE_Y - PHONE_MAX_T - 2.0
    body += box(-hw, hw, ly, FACE_Y + 0.5, CRADLE_ZB, CRADLE_ZB + 4.0)        # ledge
    body += box(-hw, hw, ly, ly + 3.0, CRADLE_ZB, CRADLE_ZB + 8.0)            # front lip
    body -= box(-8.0, 8.0, ly - 1, FACE_Y - 4.0, CRADLE_ZB - 1, CRADLE_ZB + 9)  # charging-cable notch
    body += box(-hw, hw, yb - 0.5, yh, -18.0, 18.0)                           # channel housing
    body += cyl_y(36.0, yh - 0.5, yh + 6.0)                                   # rotation boss
    for z0, z1 in (CH_UPPER_Z, CH_LOWER_Z):
        body -= box(-hw - 1, hw + 1, CH_Y[0], CH_Y[1], z0, z1)
    # Ø2 dowel pins anchoring the jaw return springs at the far end of each channel
    yc = (CH_Y[0] + CH_Y[1]) / 2
    body -= cyl_z(SPRING_PIN_D, 0.0, 19.0, -SPRING_PIN_X, yc)
    body -= cyl_z(SPRING_PIN_D, -19.0, 0.0, SPRING_PIN_X, yc)
    # M6 countersunk screw from the phone face into the hub insert
    body -= cyl_y(6.4, FACE_Y - 1, yh + 7)
    body -= bd.Pos(0, FACE_Y, 0) * bd.Cone(6.2, 3.2, 3.0, rotation=(-90, 0, 0),
                                          align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
    return body


def jaw(side, opening=JAW_AS_BUILT):
    """side=+1 right jaw (upper channel), -1 left jaw (lower channel)."""
    g = opening / 2
    ly = FACE_Y - PHONE_MAX_T - 4.0
    wall = box(g, g + 4.0, ly, FACE_Y + 20.0, -40.0, 20.0)
    lip = box(g - 3.0, g, ly, ly + 4.0, -40.0, 20.0)
    z0, z1 = CH_UPPER_Z if side > 0 else CH_LOWER_Z
    c = (CH - BAR) / 2
    bar = box(g - BAR_LEN, g + 0.5, CH_Y[0] + c, CH_Y[1] - c, z0 + c, z1 - c)
    # hook hole for the return spring at the inner end of the bar
    bar -= cyl_z(2.2, z0 - 1, z1 + 1, g - BAR_LEN + 3.0, (CH_Y[0] + CH_Y[1]) / 2)
    part = wall + lip + bar
    part = part.fillet(1.0, part.edges().filter_by(bd.Axis.Z).filter_by_position(bd.Axis.X, g + 3.9, g + 4.1))
    return part if side > 0 else bd.mirror(part, about=bd.Plane.YZ)


def fan_pad():
    x0, x1 = FAN_SHELF_X_OUT, FAN_WEB_X - T / 2
    zt = FAN_SHELF_TOP - T
    zb = FAN_SHELF_TOP - FAN_SHELF_THICK
    hw = FAN_W / 2
    body = box(x0, x1, -hw, hw, zb, zt)
    for y0, y1 in ((-hw, -hw + 10.0), (hw - 10.0, hw)):
        rib = bd.Plane.XZ * bd.Polygon((x1, zb + 0.5), (x1, 45.0), (x1 - 43.0, zb + 0.5), align=None)
        rib = bd.extrude(rib, amount=y1 - y0)
        bb = rib.bounding_box()
        body += bd.Pos(0, y0 - bb.min.Y, 0) * rib
    body -= box(x1 - 3.0, x1 + 1, -hw - 1, hw + 1, zt - 3.0, zt + 1)   # clears the inside bend radius
    body -= [cyl_z(INSERT_M4, zt - 8.0, zt + 1, x, y) for x in (-135.0, -182.0) for y in (-44.0, 44.0)]
    body -= [cyl_x(INSERT_M4, x1 - 8.0, x1 + 1, y, 65.0) for y in (-45.0, 45.0)]
    return body


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
    # cable notches at both ends of the flange
    for ys in (-1, 1):
        yn = by + ys * (straight / 2 + core_r)
        body -= box(bx - 2.3, bx + 2.3, min(yn, yn + ys * 20), max(yn, yn + ys * 20), z_core - 1, z_fl + 4)
    # USB-C parking dock on top of the bobbin
    zd = z_fl + 3.0
    body += bd.Pos(bx, by, zd - 0.01) * bd.extrude(bd.RectangleRounded(16.0, 24.0, 3.0), amount=12.0)
    body -= box(bx - 1.5, bx + 1.5, by - 4.4, by + 4.4, zd + 12.0 - 8.0, zd + 13.0)
    body -= box(bx - 4.0, bx + 4.0, by - 7.0, by + 7.0, zd + 12.0 - 1.5, zd + 13.0)   # overmold seat
    # charger pocket
    cx, cy = CHARGER_C
    ix, iy, depth = CHARGER_IN
    w = 2.4
    pocket = bd.Pos(cx, cy, zp - 0.5) * bd.extrude(bd.RectangleRounded(ix + 2 * w, iy + 2 * w, 4.0), amount=depth + 0.5)
    pocket -= bd.Pos(cx, cy, zp) * bd.extrude(bd.RectangleRounded(ix, iy, 2.0), amount=depth + 1)
    pocket -= box(cx - 6.0, cx + 6.0, cy - iy / 2 - w - 1, cy - iy / 2 + 1, zp + depth - 18.0, zp + depth + 1)
    body += pocket
    # M4 countersunk screws into the ballast plate
    for x, y in REEL_HOLES:
        body -= cyl_z(M4, z0 - 1, zp + 1, x, y)
        body -= bd.Pos(x, y, zp - 2.2) * bd.Cone(M4 / 2, M4 / 2 + 2.3, 2.3,
                                                 align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
    return body


# ---------------------------------------------------------------- purchased / reference

def star_knob(axis_dir, base, stem_len, head_d=30.0, head_t=10.0):
    """Simplified star knob along +X (axis_dir=+1) or -X from point `base`."""
    x, y, z = base
    s = axis_dir
    stem = cyl_x(6.0, min(x, x + s * stem_len), max(x, x + s * stem_len), y, z)
    hx0 = x + s * stem_len
    head = cyl_x(head_d, min(hx0, hx0 + s * head_t), max(hx0, hx0 + s * head_t), y, z)
    for i in range(5):
        a = math.radians(72 * i)
        head += cyl_x(10.0, min(hx0, hx0 + s * head_t), max(hx0, hx0 + s * head_t),
                      y + head_d / 2 * math.cos(a), z + head_d / 2 * math.sin(a))
    return stem + head


def rubber_foot(x, y):
    return cyl_z(FOOT_D, 0.0, FOOT_H, x, y)


def phone_ref():
    return box(-38.0, 38.0, FACE_Y - 9.0, FACE_Y, CRADLE_ZB + 4.0, CRADLE_ZB + 4.0 + 160.0)


def fan_ref():
    """Clip fan, 250 mm dia x 80 mm, clamped on the shelf (reference only)."""
    xs = FAN_SHELF_X_OUT
    clamp = box(xs - 16, xs - 4, -15, 15, FAN_SHELF_TOP - FAN_SHELF_THICK - 22, FAN_SHELF_TOP + 8)
    clamp += box(xs - 4, xs + 36, -15, 15, FAN_SHELF_TOP, FAN_SHELF_TOP + 8)
    clamp += box(xs - 4, xs + 30, -15, 15, FAN_SHELF_TOP - FAN_SHELF_THICK - 22, FAN_SHELF_TOP - FAN_SHELF_THICK - 14)
    clamp += cyl_z(10, FAN_SHELF_TOP - FAN_SHELF_THICK - 14, FAN_SHELF_TOP - FAN_SHELF_THICK, xs + 18)
    neck = cyl_z(16, FAN_SHELF_TOP + 8, FAN_SHELF_TOP + 40, xs + 11)
    fc = (xs + 11, 0.0, FAN_SHELF_TOP + 40 + 125)
    ring = cyl_y(250, -40, 40, fc[0], fc[2]) - cyl_y(236, -41, 41, fc[0], fc[2])
    hubp = cyl_y(70, -10, 30, fc[0], fc[2])
    return clamp + neck + ring + hubp
