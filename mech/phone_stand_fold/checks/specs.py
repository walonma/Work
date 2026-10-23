"""Design checks: folded envelope, reach, mass, stability without the fan, pivot torque, print fit."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cadgen import build123d as bd

from lib import parts
from lib.assembly_tree import build, link_locations
from lib.dims import *  # noqa: F403
from lib.orient import on_bed

STEEL, PETG, FILL = 7.85e-3, 1.27e-3, 0.45
PHONE_G = 230.0
BED = 256.0
ok = True


def check(name, cond, detail):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ": " + detail)


folded = build(0.0, 0.0, 0.0, refs=False).bounding_box()
print(f"INFO folded envelope {folded.size.X:.0f} x {folded.size.Y:.0f} x {folded.size.Z:.0f} mm "
      f"(box inside ~{folded.size.X + 10:.0f} x {folded.size.Y + 10:.0f} x {folded.size.Z + 10:.0f})")
check("folded arm lower than fan shelf", folded.max.Z <= SHELF_TOP + 3.0, f"top {folded.max.Z:.1f} vs shelf {SHELF_TOP}")
check("fan clamp fits shelf", SHELF_THICK <= 30 - 5, f"{SHELF_THICK} mm in a 30 mm clamp")
check("clamp lower jaw reaches the desk", 50.0 <= SHELF_BOTTOM <= 66.0, f"shelf underside {SHELF_BOTTOM} mm for a 50-60 mm jaw")

# masses (part frames are fine for mass; CoM taken from the placed assembly)
lo, up, hd = link_locations()
cp = hd * parts.cradle_place()
items = {
    "base_plate": (parts.base_plate(), STEEL), "lower_arm": (lo * parts.lower_arm(), STEEL),
    "upper_arm": (up * parts.upper_arm(), STEEL), "head": (hd * parts.head(), PETG * FILL),
    "cradle_body": (cp * parts.cradle_body(), PETG * FILL), "jaw_left": (cp * parts.jaw(-1), PETG * FILL),
    "jaw_right": (cp * parts.jaw(+1), PETG * FILL), "cable_reel": (parts.cable_reel(), PETG * FILL),
}
mass = {k: s.volume * rho for k, (s, rho) in items.items()}
total = sum(mass.values()) + 40.0           # pivots, standoffs, pads
for k, m in mass.items():
    print(f"INFO {k}: {m:.0f} g")
print(f"INFO stand total ~{total:.0f} g (incl. ~40 g hardware)")


def phone_point(sh, el, h):
    c = chain(sh, el, h)
    loc = bd.Pos(*c["h3"]) * bd.Rot(c["head"], 0, 0) * parts.cradle_place()
    return (loc * parts.phone_ref()).center()


fixed_m = sum(m for k, m in mass.items() if k in ("base_plate", "cable_reel")) + 40.0
fixed_c = (sum((items[k][0].center(bd.CenterOf.MASS) * mass[k] for k in ("base_plate", "cable_reel")), bd.Vector())
           / (fixed_m - 40.0))
front = -BASE_D / 2
worst = None
heights = []
for sh, psi, tilt in itertools.product(range(int(SHOULDER_RANGE[0]), int(SHOULDER_RANGE[1]) + 1, 10), range(-60, 61, 15), (0, 30, 60)):
    el, h = sh - psi, 90 - tilt - psi
    if not (ELBOW_RANGE[0] <= el <= ELBOW_RANGE[1] and HEAD_RANGE[0] <= h <= HEAD_RANGE[1]):
        continue
    p = phone_point(sh, el, h)
    heights.append(p.Z)
    # moving parts + phone CoM, crude: arm + head mass lumped at the phone for the worst case
    mov = sum(m for k, m in mass.items() if k not in ("base_plate", "cable_reel")) + PHONE_G
    com_y = (fixed_c.Y * fixed_m + p.Y * mov) / (fixed_m + mov)
    if worst is None or com_y < worst[0]:
        worst = (com_y, sh, psi, tilt, p)
print(f"INFO phone centre height {min(heights):.0f} .. {max(heights):.0f} mm over the pose sweep")
com_y, sh, psi, tilt, p = worst
check("no forward tip, no fan, worst reach", com_y > front + 5,
      f"CoM y {com_y:.1f} vs front edge {front} (shoulder {sh}, upper {psi}, tilt {tilt}, phone y {p.Y:.0f})")

# Pivot torque to hold the default pose with a 230 g phone and 2 N of finger tap at the phone
g = 9.81e-3
p0 = phone_point(SHOULDER, ELBOW, HEAD)
c0 = chain()
tap = 2.0
for name, piv in (("shoulder", c0["h1"]), ("elbow", c0["h2"]), ("head", c0["h3"])):
    arm_mm = abs(p0.Y - piv[1])
    lever = ((p0.Y - piv[1]) ** 2 + (p0.Z - piv[2]) ** 2) ** 0.5
    need = (PHONE_G * g * arm_mm + tap * lever) / 1000.0
    print(f"INFO {name} pivot: hold torque >= {need:.2f} N*m (phone arm {arm_mm:.0f} mm)")

for n, (s, rot) in {"head": (parts.head(), (180, 0, 0)), "cradle_body": (parts.cradle_body(), (0, 0, 0)),
                    "jaw_left": (parts.jaw(-1), (0, -90, 0)), "jaw_right": (parts.jaw(+1), (0, 90, 0)),
                    "cable_reel": (parts.cable_reel(), (0, 0, 0))}.items():
    bb = on_bed(s, rot).bounding_box()
    check(f"print fit {n}", max(bb.size.X, bb.size.Y, bb.size.Z) <= BED - 6,
          f"{bb.size.X:.0f} x {bb.size.Y:.0f} x {bb.size.Z:.0f}, ~{s.volume * PETG * FILL:.0f} g")
print("ALL PASS" if ok else "SOME CHECKS FAILED")
