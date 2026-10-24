"""Mass budget and centre of mass vs the support polygon (zero pose)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cadgen import build123d as bd

from lib.assembly_tree import stage_parts
from lib.dims import JOINTS, LEG_SERVO, MICRO_SERVO

PETG = 1.27e-3
# label prefix -> (mass in g, or density in g/mm^3 x fill)
FIXED = {"battery": 45.0, "rk3566_board": 20.0, "speaker": 5.0, "tof_sensor": 1.5, "camera": 3.0, "mics": 1.0,
         "eye_lcd_l": 3.0, "eye_lcd_r": 3.0}
DENSITY = {"shell": PETG, "wing": PETG * 0.6, "tail": PETG * 0.6, "neck_cover": PETG, "beak": PETG * 0.6,
           "eye_lens": 1.19e-3, "foot_pad": 1.2e-3, "foot_": PETG * 0.6, "horn": 1.2e-3}
FRAME_FILL = PETG * 0.5   # brackets, links, frames


def mass_of(label, shape):
    if label.startswith("servo_"):
        return (LEG_SERVO if JOINTS[label[6:]][1] == "leg" else MICRO_SERVO)["mass"]
    if label in FIXED:
        return FIXED[label]
    for k, rho in DENSITY.items():
        if k in label:
            return shape.volume * rho
    return shape.volume * FRAME_FILL


P = stage_parts()
rows, M, mom = [], 0.0, bd.Vector()
for stage, items in P.items():
    for s in items:
        m = mass_of(s.label, s)
        c = s.center(bd.CenterOf.MASS) if s.volume > 0 else s.center()
        rows.append((s.label, m))
        M += m
        mom += c * m
com = mom / M
groups = {}
for label, m in rows:
    key = ("servos" if label.startswith("servo_") else "shells & cosmetics" if any(k in label for k in ("shell", "wing", "tail", "beak", "cover", "lens"))
           else "electronics" if label in FIXED else "frames & links")
    groups[key] = groups.get(key, 0) + m
for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
    print(f"INFO {k:20s} {v:6.0f} g")
print(f"INFO total {M:.0f} g, CoM ({com.X:.1f}, {com.Y:.1f}, {com.Z:.1f}) mm")
bb = bd.Compound(children=[s for v in P.values() for s in v]).bounding_box()
print(f"INFO envelope {bb.size.X:.0f} x {bb.size.Y:.0f} x {bb.size.Z:.0f} mm (height {bb.max.Z:.0f})")
ok = True
for name, cond, detail in [
    ("mass <= 800 g", M <= 800, f"{M:.0f} g"),
    ("CoM over the feet (x)", -5 <= com.X <= 45, f"x {com.X:.1f} within foot -10..55"),
    ("CoM centred (y)", abs(com.Y) < 2, f"y {com.Y:.1f}"),
    ("height ~25 cm", 240 <= bb.max.Z <= 265, f"{bb.max.Z:.0f} mm"),
]:
    ok &= cond
    print(("PASS " if cond else "FAIL ") + name + ": " + detail)
print("ALL PASS" if ok else "SOME CHECKS FAILED")
