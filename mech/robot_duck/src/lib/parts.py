"""Part factories for the robot duck concept, all in world coordinates at the zero pose."""
from __future__ import annotations

import math

from cadgen import build123d as bd

from lib.dims import *  # noqa: F403 - shared dimension table


def box(x0, x1, y0, y1, z0, z1):
    return bd.Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * bd.Box(x1 - x0, y1 - y0, z1 - z0)


def cyl(d, p0, p1):
    """Cylinder of diameter d from point p0 to p1."""
    v = bd.Vector(*p1) - bd.Vector(*p0)
    c = bd.Cylinder(d / 2, v.length, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
    return bd.Location(bd.Plane(origin=p0, z_dir=v)) * c


def ellipsoid(c, r):
    return bd.Pos(*c) * bd.scale(bd.Sphere(1.0), by=r)


def _servo_loc(name):
    _parent, kind, origin, axis, body_dir, _lim = JOINTS[name]
    return bd.Location(bd.Plane(origin=origin, x_dir=body_dir, z_dir=axis))


def _spec(name):
    return LEG_SERVO if JOINTS[name][1] == "leg" else MICRO_SERVO


def servo_body(name):
    s = _spec(name)
    body = box(-s["shaft"], s["L"] - s["shaft"], -s["W"] / 2, s["W"] / 2, -s["H"], 0.0)
    body = body.fillet(1.0, body.edges().filter_by(bd.Axis.Z))
    boss = cyl(s["horn_d"] * 0.45, (0, 0, -0.01), (0, 0, 0.6))
    return _servo_loc(name) * (body + boss)


def servo_horn(name):
    s = _spec(name)
    return _servo_loc(name) * cyl(s["horn_d"], (0, 0, 0.6), (0, 0, HORN_T))


def joint_axis(name):
    """Rotation axis. Right-side joints turn about the negated horn axis so equal
    angles give mirror-symmetric motion on both legs."""
    _p, _k, origin, axis, _b, lim = JOINTS[name]
    if name.startswith("r_"):
        axis = tuple(-v for v in axis)
    return origin, axis, lim


# ---------------------------------------------------------------- body (root)

def _body_outer():
    return ellipsoid(BODY_C, BODY_R)


def body_shell(upper):
    cx, cy, cz = BODY_C
    a, b, c = BODY_R
    shell = _body_outer() - ellipsoid(BODY_C, (a - SHELL_T, b - SHELL_T, c - SHELL_T))
    half = box(-a - 5, a + 5, -b - 5, b + 5, cz, cz + c + 5) if upper else box(-a - 5, a + 5, -b - 5, b + 5, cz - c - 5, cz)
    shell = shell & half
    if upper:
        shell -= box(4.0, a + 5, -22.0, 22.0, 160.0, cz + c + 5)            # neck opening
    else:
        for s in (-1, 1):                                                      # hip openings
            for hx in (-14.0, 8.0):
                shell -= cyl(58.0, (hx, s * HIP_Y, cz - c - 5), (hx, s * HIP_Y, cz - 14.0))
        shell -= box(48.0, 58.0, -7.0, 7.0, 146.0, 154.0)                    # ToF window
        shell -= cyl(18.0, (-a - 5, 0, 150.0), (-a + 8, 0, 150.0))            # speaker grille
    return shell


def wing(side):
    return ellipsoid((-8.0, side * (BODY_R[1] - 1.0), 160.0), (30.0, 4.0, 15.0))


def tail():
    return cyl(14.0, (-56.0, 0.0, 168.0), (-70.0, 0.0, 178.0)) + ellipsoid((-70.0, 0.0, 178.0), (5.0, 7.0, 5.0))


def pelvis_frame():
    frame = box(-30.0, 12.0, -34.0, 34.0, 166.0, 169.0)                  # top plate over the hip yaw servos
    frame += box(12.0, 42.0, -13.0, 13.0, 166.0, 169.0)                  # arm to the neck servo
    frame += box(22.0, 42.0, -13.0, 13.0, 169.0, 172.0)                  # neck servo seat
    frame += box(-27.0, 27.0, -14.0, 14.0, 138.0, 139.0)                 # battery tray
    frame += box(-28.0, -26.0, -12.0, 12.0, 138.0, 166.0)                # spine
    return frame


def battery():
    return box(*BATTERY)


def board():
    b = box(*BOARD)
    return b + box(-10.0, -2.0, -6.0, 6.0, BOARD[5], BOARD[5] + 2.0)    # IMU / heat spreader


def speaker():
    return cyl(20.0, (-50.0, 0.0, 150.0), (-44.0, 0.0, 150.0))


def tof_sensor():
    return box(50.0, 54.0, -6.0, 6.0, 146.0, 154.0)


# ---------------------------------------------------------------- legs (s = +1 left, -1 right)

def _m(s, shape):
    return shape if s > 0 else bd.mirror(shape, about=bd.Plane.XZ)


def yaw_bracket(s):
    return _m(s, box(-27.0, -1.0, HIP_Y - 10, HIP_Y + 10, 132.0, 138.0))


def hip_bracket(s):
    return _m(s, box(1.0, 4.0, HIP_Y - 10, HIP_Y + 10, 98.0, 130.0) + box(1.0, LEG_X + 12, HIP_Y - 12, HIP_Y + 14, 120.0, 123.0))


def thigh(s):
    y = HIP_Y + 16
    x = LEG_X
    return _m(s, box(x - 10, x + 10, y, y + 3, 80.0, 118.0) + box(x - 10, x + 24, HIP_Y - 12, y + 3, 80.0, 83.0))


def shin(s):
    y = HIP_Y + 16
    x = LEG_X
    knee_z = 70.0
    plate = box(x - 10, x + 10, y, y + 3, 20.0, knee_z) + cyl(20.0, (x, y, knee_z), (x, y + 3, knee_z))
    return _m(s, plate + box(x - 10, x + 10, HIP_Y + 12, y, 20.0, 54.0))


def foot(s):
    y = HIP_Y
    pts = [(-10, y - 10), (30, y - 20), (50, y - 14), (44, y - 5), (55, y), (44, y + 5), (50, y + 14), (30, y + 20), (-10, y + 10)]
    plate = bd.extrude(bd.Polygon(*pts, align=None), amount=5.0)
    plate = plate.fillet(1.5, plate.edges().filter_by(bd.Axis.Z))
    bracket = box(LEG_X - 8, LEG_X + 8, HIP_Y - 19, HIP_Y - 16, 5.0, 36.0)
    return _m(s, plate + bracket)


def foot_pad(s):
    y = HIP_Y
    pts = [(-8, y - 8), (30, y - 17), (47, y - 12), (42, y - 4), (51, y), (42, y + 4), (47, y + 12), (30, y + 17), (-8, y + 8)]
    return _m(s, bd.Pos(0, 0, -1.5) * bd.extrude(bd.Polygon(*pts, align=None), amount=1.5))


# ---------------------------------------------------------------- neck and head

def neck_link():
    return box(22.0, 56.0, 15.0, 18.0, 172.0, 192.0) + box(50.0, 73.0, -6.0, 18.0, 177.0, 181.0)


def neck_cover():
    return cyl(32.0, (62.0, 0.0, 182.0), (62.0, 0.0, 198.0)) - cyl(28.0, (62.0, 0.0, 181.0), (62.0, 0.0, 199.0))


def pitch_bracket():
    return box(53.0, 76.0, -12.0, 12.0, 207.0, 209.0)


def roll_bracket():
    return box(52.0, 73.0, 14.0, 17.0, 212.0, 242.0) + box(49.0, 73.0, 6.0, 14.0, 228.0, 240.0)


def head_frame():
    return box(75.0, 78.0, -14.0, 14.0, 224.0, 248.0)


def _cam_dir():
    return bd.Vector(0.8, 0.0, 0.6).normalized()


def _eye_dir(side):
    d = bd.Vector(0.55, side * 0.62, 0.30)
    return d.normalized()


def head_shell(front):
    shell = ellipsoid(HEAD_C, (HEAD_R,) * 3) - ellipsoid(HEAD_C, (HEAD_R - SHELL_T,) * 3)
    hx = HEAD_C[0]
    half = box(hx, hx + 40, -40, 40, 180, 270) if front else box(hx - 40, hx, -40, 40, 180, 270)
    shell = shell & half
    shell -= cyl(60.0, (62.0, 0, 180.0), (62.0, 0, 205.0))                 # neck opening
    if front:
        c = bd.Vector(*HEAD_C)
        for s in (-1, 1):
            d = _eye_dir(s)
            shell -= cyl(22.0, tuple(c + d * (HEAD_R - 8)), tuple(c + d * (HEAD_R + 4)))   # eye windows
        cam = _cam_dir()
        shell -= cyl(6.0, tuple(c + cam * (HEAD_R - 8)), tuple(c + cam * (HEAD_R + 4)))  # camera window
        shell -= box(84.0, 108.0, -17.0, 17.0, 188.0, 213.0)                 # lower beak / linkage opening
    return shell


def eye_lcd(side):
    """0.71 in round LCD (GC9A01 class), behind a clear dome."""
    c = bd.Vector(*HEAD_C)
    d = _eye_dir(side)
    return cyl(22.0, tuple(c + d * (HEAD_R - 5.0)), tuple(c + d * (HEAD_R - 2.5)))


def eye_lens(side):
    c = bd.Vector(*HEAD_C)
    d = _eye_dir(side)
    return cyl(21.6, tuple(c + d * (HEAD_R - 2.0)), tuple(c + d * (HEAD_R + 0.5)))


def camera():
    c = bd.Vector(*HEAD_C)
    d = _cam_dir()
    p = c + d * (HEAD_R - 6.0)
    return cyl(9.0, tuple(p - d * 3), tuple(p + d * 3)) + bd.Pos(*tuple(p - d * 4)) * bd.Box(4, 10, 10)


def mics():
    c = bd.Vector(*HEAD_C)
    out = []
    for s in (-1, 1):
        d = bd.Vector(0.0, s * 0.55, 0.83).normalized()
        out.append(cyl(4.0, tuple(c + d * (HEAD_R - 4.5)), tuple(c + d * (HEAD_R - 2.5))))
    return out[0] + out[1]


def beak_upper():
    return ellipsoid((108.0, 0.0, 217.0), (18.0, 13.0, 5.0)) - ellipsoid(HEAD_C, (HEAD_R - SHELL_T,) * 3)


def beak_lower():
    jaw = ellipsoid((108.0, 0.0, 208.0), (15.0, 10.0, 3.5))
    arm = box(80.0, 92.0, 14.0, 16.0, 204.0, 220.0) + box(88.0, 96.0, 8.0, 16.0, 204.0, 208.0)
    return jaw + arm
