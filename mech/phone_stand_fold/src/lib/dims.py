"""Dimensions for the folding phone stand (mm, degrees).

World frame: X to the right (fan shelf on -X, cable reel on +X), Y away from
the user, Z up from the desk (Z=0). The stand ships fully assembled and folded;
the user lifts the arm to use it.
"""
import math

T = 1.6                    # SPCC sheet thickness
BEND_R = 1.6               # inner bend radius

# Base tray (SPCC 1.6, one blank: tray + fan shelf + hinge ears)
BASE_W = 230.0             # X (tray only, the fan trough is outboard)
BASE_D = 140.0             # Y
BASE_TOP = 14.0            # top surface
PAD_T = 2.0                # silicone pad under the hems
HEM_W = 8.0                # inward hem under the flanges, carries the pad

# Fan plate for a DIKE DUF201 clip fan (spring clip, opens 46 mm, 226 g).
# The left edge folds down, runs out along the desk and back up as a vertical
# plate. The clip slides down over the plate: the outer jaw tip lands on the
# desk, the inner jaw drops into the trough, so the fan props that side.
FAN_PLATE_TOP = 40.0       # plate top (hemmed edge) above the desk
TROUGH_GAP = 20.0          # room for the inner jaw between tray flange and plate
HEM_DROP = 10.0            # closed hem on the plate top edge
FAN_JAW_T = 12.0           # assumed DUF201 jaw thickness (reference model only)

# Arm: three friction pivots (M5 shoulder bolt + disc springs, set at the factory)
ARM_X = 0.0
H1 = (0.0, -48.0, 26.0)    # shoulder pivot on the base ears
L1 = 110.0                 # lower arm, pivot to pivot
L2 = 95.0                  # upper arm, pivot to pivot
UP_W = 32.4                # upper arm channel outer width
UP_H = 9.0
PIVOT_GAP = 4.0            # each side of the upper arm inside the lower arm: friction
                           # washers at the elbow, room for the head pivot heads when folded
LOW_W = UP_W + 2 * PIVOT_GAP + 2 * T   # lower arm channel outer width (upper nests inside)
LOW_H = 16.0               # lower arm wall height (pivot at mid wall)
EAR_W = 16.0               # base ear width (Y)
PIVOT_D = 5.0              # M5 shoulder bolts

# Head: tongue in the upper arm end, plate carrying the cradle's rotation axis
HEAD_AXIS_OFF = 50.0       # rotation axis offset along the head (+Y when folded)
HEAD_PLATE_Z = (LOW_H / 2 + 0.5, LOW_H / 2 + 7.5)
HEAD_W = 36.0

# Authored pose (deployed, typical use). Absolute angles:
#   shoulder: lower arm from +Y (folded flat) rotating up about +X
#   elbow:    upper arm unfolded from the lower arm
#   head:     head unfolded from the upper arm
SHOULDER = 60.0
ELBOW = 105.0
HEAD = 120.0
SHOULDER_RANGE = (0.0, 70.0)
ELBOW_RANGE = (0.0, 165.0)
HEAD_RANGE = (0.0, 170.0)

# Phone cradle (reused from v1, local frame: face at y=-42, rotation axis along Y)
FACE_Y = -42.0
PLATE_HW = 28.0
CRADLE_ZB = -85.0
CRADLE_ZT = 25.0
PHONE_MAX_T = 14.0
JAW_MIN = 60.0
JAW_MAX = 95.0
JAW_AS_BUILT = 76.0
BAR = 6.4
BAR_LEN = 45.0
CH = 7.0
CH_Y = (-33.0, -26.0)
CH_UPPER_Z = (5.0, 12.0)
CH_LOWER_Z = (-12.0, -5.0)
SPRING_PIN_X = 26.0
SPRING_PIN_D = 2.0

# Cable organiser (USB-C charger + 90 cm cable), screwed to PEM nuts in the base
REEL_X = (50.0, 114.0)
REEL_Y = (-67.0, 67.0)
REEL_PLATE_T = 4.0
BOBBIN_C = (82.0, -24.0)
REEL_HOLES = [(56.0, -60.0), (108.0, -60.0), (56.0, 18.0), (108.0, 18.0)]
CHARGER_IN = (54.0, 34.0, 40.0)
CHARGER_C = (82.0, 42.0)


def rot_x(v, deg):
    a = math.radians(deg)
    x, y, z = v
    return (x, y * math.cos(a) - z * math.sin(a), y * math.sin(a) + z * math.cos(a))


def add(a, b):
    return tuple(p + q for p, q in zip(a, b))


def chain(shoulder=SHOULDER, elbow=ELBOW, head=HEAD):
    """World pivots and link rotations (deg about +X) for absolute joint angles."""
    psi = shoulder - elbow
    h2 = add(H1, rot_x((0.0, L1, 0.0), shoulder))
    h3 = add(h2, rot_x((0.0, -L2, 0.0), psi))
    return {"h1": H1, "h2": h2, "h3": h3, "lower": shoulder, "upper": psi, "head": psi + head}
