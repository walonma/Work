"""Shared dimensions for the phone stand (mm).

World frame: X to the right (fan on -X, cable reel on +X), Y away from the
user (phone faces -Y), Z up from the desk. Z=0 is the desk surface.
"""

# Sheet metal (SPCC)
T = 2.0                    # sheet thickness
BEND_R = 2.0               # inner bend radius

# Base plate (SPCC 2.0, flanges folded down)
BASE_W = 300.0             # X
BASE_D = 160.0             # Y
BASE_FLANGE = 10.0         # flange height below the top plate
BASE_TOP = 13.0            # top surface Z

# Ballast plate (steel 6 t, laser cut) under the base, also acts as nut plate
WEIGHT_W = 284.0
WEIGHT_D = 130.0
WEIGHT_T = 6.0
WEIGHT_Z0 = BASE_TOP - T - WEIGHT_T

# Rubber feet (purchased, adhesive) under the ballast plate
FOOT_D = 20.0
FOOT_H = WEIGHT_Z0
FOOT_XY = [(sx * 120.0, sy * 52.0) for sx in (-1, 1) for sy in (-1, 1)]

# Column position
CX = 30.0
CY = 20.0

# Square steel tubes
OUTER_A = 25.0             # outer tube 25x25x1.5
OUTER_WALL = 1.5
INNER_A = 20.0             # inner tube 20x20x1.5
INNER_WALL = 1.5

# Column bracket (SPCC 2.0 U)
BRK_WALL_H = 32.0          # wall height above base top
BRK_WALL_D = 30.0          # wall depth (Y)
BRK_WEB_D = 70.0           # web depth (Y)

FOOT_PLUG_T = 2.5          # printed foot lifts the tube clear of the bend radius
OUTER_Z0 = BASE_TOP + T + FOOT_PLUG_T
BRK_BOLT_Z = (OUTER_Z0 + 10.0, OUTER_Z0 + 22.0)   # cross bolts M5
OUTER_LEN = 126.0
OUTER_TOP = OUTER_Z0 + OUTER_LEN

COLLAR_SIDE = 39.0
COLLAR_H = 40.0
COLLAR_Z0 = OUTER_TOP - 20.0           # 20 mm socket over the outer tube
COLLAR_TOP = COLLAR_Z0 + COLLAR_H
COLLAR_KNOB_Z = COLLAR_TOP - 10.0

INNER_LEN = 147.0
INNER_MIN_BOTTOM = 44.0                # clears the bracket cross bolts
TRAVEL = 80.0                          # height adjustment
INNER_TOP_MIN = INNER_MIN_BOTTOM + INNER_LEN

# Head: tilt axis is 24 mm above the inner tube top
AXIS_ABOVE_TUBE = 24.0
AXIS_Z_MIN = INNER_TOP_MIN + AXIS_ABOVE_TUBE
TILT_LIMITS = (0.0, 70.0)

# Phone cradle (local to the tilt axis; y toward the user is negative)
FACE_Y = -42.0             # phone back rests on this plane
PLATE_HW = 28.0            # half width of the back plate
CRADLE_ZB = -85.0          # bottom of the ledge
CRADLE_ZT = 25.0           # top of the back plate
PHONE_MAX_T = 14.0         # phone + case thickness
JAW_MIN = 60.0             # jaw opening range
JAW_MAX = 95.0
JAW_AS_BUILT = 76.0        # opening the assembly is written at
BAR = 6.4                  # sliding bar section
BAR_LEN = 45.0
CH = 7.0                   # channel section
CH_Y = (-33.0, -26.0)
CH_UPPER_Z = (5.0, 12.0)   # right jaw
CH_LOWER_Z = (-12.0, -5.0) # left jaw
SPRING_PIN_X = 26.0        # spring anchor pins, opposite side to each jaw
SPRING_PIN_D = 2.0

# Fan mount (the fan's own clamp grips this 18 mm thick shelf)
FAN_SHELF_TOP = 110.0
FAN_SHELF_THICK = 18.0     # fan clamp opens to 30 mm
FAN_WEB_X = -116.0         # bracket web centerline
FAN_SHELF_X_OUT = -196.0   # outer edge of the shelf
FAN_FLANGE_X_IN = -60.0
FAN_W = 100.0              # bracket width (Y)

# Cable organiser (USB-C charger + 90 cm cable)
REEL_X = (62.0, 146.0)
REEL_Y = (-76.0, 62.0)
REEL_PLATE_T = 4.0
BOBBIN_C = (104.0, -32.0)
REEL_HOLES = [(68.0, -60.0), (136.0, -60.0), (68.0, 50.0), (136.0, 50.0)]
CHARGER_IN = (54.0, 34.0, 40.0)        # pocket inner X, Y, depth (65 W GaN ~50x30x50)
CHARGER_C = (104.0, 38.0)

# Base holes
COLUMN_HOLES = [(CX, CY - 25.0), (CX, CY + 25.0)]          # M5
FAN_HOLES = [(-85.0, -30.0), (-85.0, 30.0)]                # M5
BASE_SCREWS = [(-125.0, -55.0), (-125.0, 55.0), (0.0, -55.0), (0.0, 55.0)]  # M5 csk
