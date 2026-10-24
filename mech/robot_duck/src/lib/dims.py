"""Robot duck concept (Microduck class): dimensions, servo placement, joints. mm / deg.

World frame (URDF convention): X forward (beak), Y left, Z up, desk Z=0.
Zero pose = standing, legs straight. Every servo is placed by its output
(horn) face centre, the axis the horn faces, and the direction its body runs.
"""

# ---------------------------------------------------------------- actuators
# Leg/neck servo: XL330 / Feetech HD-1910 class bus servo envelope (34 x 20 x 26, ~18 g)
LEG_SERVO = dict(L=34.0, W=20.0, H=26.0, shaft=10.0, horn_d=20.0, mass=18.0)
# Head/beak servo: Feetech SCS0009 class micro bus servo (23 x 12 x 24, ~9 g)
MICRO_SERVO = dict(L=23.0, W=12.0, H=24.0, shaft=6.0, horn_d=14.0, mass=9.0)
HORN_T = 2.0

HIP_Y = 26.0               # leg centreline from the body midplane
LEG_X = 15.0               # hip pitch / knee / ankle axes, ahead of the hip yaw axis

# ---------------------------------------------------------------- shells
BODY_C = (0.0, 0.0, 156.0)
BODY_R = (60.0, 46.0, 36.0)          # ellipsoid semi-axes
HEAD_C = (68.0, 0.0, 224.0)
HEAD_R = 34.0              # big head: the cute part, and room for 4 micro servos
SHELL_T = 2.0

# ---------------------------------------------------------------- joints
# name: (parent joint or None, servo kind, horn origin, horn axis, body direction, limits)
_LEFT = {
    "hip_yaw":   (None,        "leg", (-15.0, HIP_Y, 140.0), (0, 0, -1), (1, 0, 0), (-30, 30)),
    "hip_roll":  ("hip_yaw",   "leg", (-1.0, HIP_Y, 122.0), (1, 0, 0), (0, 0, -1), (-20, 20)),
    "hip_pitch": ("hip_roll",  "leg", (LEG_X, HIP_Y + 14.0, 110.0), (0, 1, 0), (0, 0, -1), (-60, 60)),
    "knee":      ("hip_pitch", "leg", (LEG_X, HIP_Y + 14.0, 70.0), (0, 1, 0), (1, 0, 0), (-10, 90)),
    "ankle":     ("knee",      "leg", (LEG_X, HIP_Y - 14.0, 30.0), (0, -1, 0), (0, 0, 1), (-45, 45)),
}


def _mirror(v):
    return (v[0], -v[1], v[2])


JOINTS = {}
for side, m in (("l", lambda v: v), ("r", _mirror)):
    for name, (parent, kind, o, a, b, lim) in _LEFT.items():
        JOINTS[f"{side}_{name}"] = (f"{side}_{parent}" if parent else None, kind, m(o), m(a), m(b), lim)
JOINTS.update({
    "neck_pitch": (None,         "leg",   (32.0, 13.0, 182.0), (0, 1, 0), (-1, 0, 0), (-30, 25)),
    "head_yaw":   ("neck_pitch", "micro", (56.0, 0.0, 205.0), (0, 0, 1), (1, 0, 0), (-90, 90)),
    "head_pitch": ("head_yaw",   "micro", (58.0, 12.0, 215.0), (0, 1, 0), (1, 0, 0), (-12, 12)),
    "head_roll":  ("head_pitch", "micro", (73.0, 0.0, 234.0), (1, 0, 0), (0, -1, 0), (-15, 15)),
    "beak":       ("head_roll",  "micro", (86.0, 12.0, 214.0), (0, 1, 0), (0, 0, 1), (-5, 35)),
})

# ---------------------------------------------------------------- electronics (placeholders)
BATTERY = (-25.0, 25.0, -13.0, 13.0, 139.0, 152.0)   # 2S 800 mAh flat LiPo 50 x 26 x 13, ~45 g
BOARD = (-24.0, 41.0, -15.0, 15.0, 154.0, 162.0)     # RK3566 SBC 65 x 30 (e.g. Radxa Zero 3W), ~20 g
