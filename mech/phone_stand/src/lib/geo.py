"""Small geometry helpers: holes and bent sheet-metal profiles."""
from __future__ import annotations

from cadgen import build123d as bd

from lib.dims import BEND_R, T


def cyl_z(d, z0, z1, x=0.0, y=0.0):
    return bd.Pos(x, y, z0) * bd.Cylinder(d / 2, z1 - z0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))


def cyl_x(d, x0, x1, y=0.0, z=0.0):
    c = bd.Cylinder(d / 2, x1 - x0, rotation=(0, 90, 0))
    return bd.Pos((x0 + x1) / 2, y, z) * c


def cyl_y(d, y0, y1, x=0.0, z=0.0):
    c = bd.Cylinder(d / 2, y1 - y0, rotation=(90, 0, 0))
    return bd.Pos(x, (y0 + y1) / 2, z) * c


def box(x0, x1, y0, y1, z0, z1):
    return bd.Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * bd.Box(x1 - x0, y1 - y0, z1 - z0)


def bent_xz(points, y0, y1, t=T, r=BEND_R):
    """Sheet bent along Y: `points` is the XZ centreline polyline."""
    line = bd.FilletPolyline(*points, radius=r + t / 2)
    face = bd.Plane.XZ * bd.trace(line, line_width=t)
    solid = bd.extrude(face, amount=y1 - y0)
    bb = solid.bounding_box()
    return bd.Pos(0, y0 - bb.min.Y, 0) * solid


def stadium(cx, cy, straight_y, r, z0, z1):
    """Obround: straight part along Y, rounded ends of radius r."""
    s = bd.Pos(cx, cy) * bd.SlotCenterToCenter(straight_y, 2 * r, rotation=90)
    return bd.Pos(0, 0, z0) * bd.extrude(s, amount=z1 - z0)


def stadium_face(cx, cy, straight_y, r, z):
    return bd.Plane.XY.offset(z) * (bd.Pos(cx, cy) * bd.SlotCenterToCenter(straight_y, 2 * r, rotation=90))
