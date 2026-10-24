"""Pairwise interference of the duck at the zero pose and at test poses."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cadgen.geometry import overlap_volume

from lib.assembly_tree import build, stage_parts

TOL = 1.0
# Designed overlaps: glued/fused cosmetic parts
ALLOWED = [{"wing_l", "body_shell_upper"}, {"wing_l", "body_shell_lower"}, {"wing_r", "body_shell_upper"},
           {"wing_r", "body_shell_lower"}, {"tail", "body_shell_upper"}, {"tail", "body_shell_lower"},
           {"beak_upper", "head_shell_front"}, {"eye_lens_l", "head_shell_front"}, {"eye_lens_r", "head_shell_front"},
           {"camera", "head_shell_front"}, {"mics", "head_shell_front"}, {"mics", "head_shell_back"}]

POSES = {
    "stand": {},
    "crouch": {"l_hip_pitch": -35, "r_hip_pitch": -35, "l_knee": 70, "r_knee": 70, "l_ankle": -35, "r_ankle": -35},
    "look_left": {"head_yaw": 60, "head_pitch": 12, "neck_pitch": 15},
    "look_right_down": {"head_yaw": -60, "head_pitch": -12, "neck_pitch": 25, "head_roll": 15},
    "quack": {"beak": 30, "head_pitch": -12, "neck_pitch": -20},
    "step_left": {"l_hip_pitch": -30, "l_knee": 50, "l_ankle": -20, "r_hip_pitch": 15, "l_hip_roll": 10, "r_hip_roll": 10,
                  "l_hip_yaw": 20},
    "nap": {"neck_pitch": 25, "head_pitch": 12, "head_roll": -15, "l_hip_pitch": -55, "r_hip_pitch": -55,
            "l_knee": 85, "r_knee": 85, "l_ankle": -40, "r_ankle": -40},
}


def leaves(shape, out):
    if shape.label and not shape.children:
        out[shape.label] = shape
    for c in shape.children:
        leaves(c, out)
    return out


def clashes(pose, P):
    parts = leaves(build(pose or {"_": 0}, P), {}) if pose else leaves(build(None, P), {})
    bad = []
    for (a, sa), (b, sb) in itertools.combinations(parts.items(), 2):
        if {a, b} in ALLOWED:
            continue
        bb1, bb2 = sa.bounding_box(), sb.bounding_box()
        if (bb1.min.X > bb2.max.X or bb2.min.X > bb1.max.X or bb1.min.Y > bb2.max.Y or bb2.min.Y > bb1.max.Y
                or bb1.min.Z > bb2.max.Z or bb2.min.Z > bb1.max.Z):
            continue
        v = sum(overlap_volume(x, y) for x in sa.solids() for y in sb.solids())
        if v > TOL:
            bad.append((a, b, round(v, 1)))
    return bad


if __name__ == "__main__":
    names = sys.argv[1:] or list(POSES)
    P = stage_parts()
    total = 0
    for n in names:
        bad = clashes(POSES[n], P)
        total += len(bad)
        print(n, "OK" if not bad else bad)
    print("PASS" if not total else f"FAIL {total} clashes")
