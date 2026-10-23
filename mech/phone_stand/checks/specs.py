"""Design checks: travel, clamp fit, print-bed fit, mass and tipping margin."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cadgen import build123d as bd

from lib import parts
from lib.dims import *  # noqa: F403
from lib.orient import on_bed

STEEL, PETG = 7.85e-3, 1.27e-3          # g/mm^3
PETG_FILL = 0.45                        # effective density factor (walls + 30 % infill)
BED = 256.0                             # Bambu Lab A1
FAN_G, PHONE_G = 1500.0, 250.0          # assumed masses
ok = True


def check(name, cond, detail):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ": " + detail)


print("--- geometry")
inner_bot_max = INNER_MIN_BOTTOM + TRAVEL
check("inner tube engagement at max height", COLLAR_TOP - inner_bot_max >= 35, f"{COLLAR_TOP - inner_bot_max:.1f} mm in collar/outer tube")
check("inner tube clears cross bolts", INNER_MIN_BOTTOM > max(BRK_BOLT_Z) + 2.75, f"bottom {INNER_MIN_BOTTOM} vs bolt top {max(BRK_BOLT_Z) + 2.75:.2f}")
yoke_bot = INNER_TOP_MIN - 25.0
check("yoke above collar at min height", yoke_bot > COLLAR_TOP, f"gap {yoke_bot - COLLAR_TOP:.1f} mm")
check("fan clamp fits shelf", FAN_SHELF_THICK <= 30.0 - 5.0, f"shelf {FAN_SHELF_THICK} mm, clamp opening 30 mm")
print(f"INFO tilt axis height {AXIS_Z_MIN:.1f} .. {AXIS_Z_MIN + TRAVEL:.1f} mm; "
      f"phone (160 mm) centre at 0 deg {AXIS_Z_MIN + CRADLE_ZB + 4 + 80:.1f} .. {AXIS_Z_MIN + TRAVEL + CRADLE_ZB + 4 + 80:.1f} mm")

print("--- print bed (A1 256 mm)")
printed = {
    "tube_foot": (parts.tube_foot(), (0, 0, 0)), "column_collar": (parts.column_collar(), (0, 0, 0)),
    "head_yoke": (parts.head_yoke(), (0, 0, 0)), "tilt_hub": (parts.tilt_hub(), (0, 90, 0)),
    "cradle_body": (parts.cradle_body(), (0, 0, 0)), "jaw_right": (parts.jaw(+1), (0, 90, 0)),
    "jaw_left": (parts.jaw(-1), (0, -90, 0)), "fan_pad": (parts.fan_pad(), (180, 0, 0)),
    "cable_reel": (parts.cable_reel(), (0, 0, 0)),
}
petg_total = 0.0
for n, (s, rot) in printed.items():
    bb = on_bed(s, rot).bounding_box()
    g = s.volume * PETG * PETG_FILL
    petg_total += g
    check(n, max(bb.size.X, bb.size.Y, bb.size.Z) <= BED - 6, f"{bb.size.X:.0f} x {bb.size.Y:.0f} x {bb.size.Z:.0f} mm, ~{g:.0f} g")

print("--- mass and tipping")
metal = {"base_plate": parts.base_plate(), "weight_plate": parts.weight_plate(), "column_bracket": parts.column_bracket(),
         "fan_bracket": parts.fan_bracket(), "outer_tube": parts.outer_tube(), "inner_tube": parts.inner_tube()}
items = []
for n, s in metal.items():
    m = s.volume * STEEL
    items.append((m, s.center(bd.CenterOf.MASS)))
    print(f"INFO {n}: {m:.0f} g")
for n, (s, _) in printed.items():
    items.append((s.volume * PETG * PETG_FILL, s.center(bd.CenterOf.MASS)))
stand = sum(m for m, _ in items)
print(f"INFO stand mass {stand:.0f} g (steel {sum(s.volume for s in metal.values()) * STEEL:.0f} g, PETG ~{petg_total:.0f} g)")

# Fan centre of mass: 15 mm outboard of the clamp, worst case; phone at max height, offset forward
fan = (FAN_G, bd.Vector(FAN_SHELF_X_OUT + 15.0, 0, FAN_SHELF_TOP + 120))
phone = (PHONE_G, bd.Vector(CX, CY + FACE_Y - 5, AXIS_Z_MIN + TRAVEL))
allm = items + [fan, phone]
M = sum(m for m, _ in allm)
com = sum((p * m for m, p in allm), bd.Vector()) / M
fx = [x for x, _ in FOOT_XY]
fy = [y for _, y in FOOT_XY]
margin_left = com.X - (min(fx) - FOOT_D / 2)
margin_front = com.Y - (min(fy) - FOOT_D / 2)
print(f"INFO total with fan {FAN_G:.0f} g + phone {PHONE_G:.0f} g: {M:.0f} g, CoM ({com.X:.1f}, {com.Y:.1f}, {com.Z:.1f})")
check("CoM inside feet (left, fan side)", margin_left > 20, f"{margin_left:.1f} mm from the left feet")
check("CoM inside feet (front)", margin_front > 20, f"{margin_front:.1f} mm from the front feet")
# Side push at the fan head needed to tip it over the left feet
lever = com.X - (min(fx) - FOOT_D / 2)
push_h = fan[1].Z
print(f"INFO side force at fan centre to tip: {M * 9.81e-3 * lever / push_h:.1f} N")
print("ALL PASS" if ok else "SOME CHECKS FAILED")
