"""手機支架總組立 (phone stand assembly) with viewer kinematics."""
import cadgen
from cadgen import glb, step

from lib.assembly_tree import build
from lib.dims import AXIS_Z_MIN, CX, CY, JAW_AS_BUILT, JAW_MAX, JAW_MIN, TILT_LIMITS, TRAVEL

_J = ((JAW_MIN - JAW_AS_BUILT) / 2, (JAW_MAX - JAW_AS_BUILT) / 2)

KINEMATICS = {
    "mates": [
        cadgen.slider("height", parent="#outer_tube", child="#height_stage",
                      origin=(CX, CY, 0), direction=(0, 0, 1), limits=(0, TRAVEL)),
        cadgen.revolute("tilt", parent="#head_yoke", child="#tilt_stage",
                        origin=(CX, CY, AXIS_Z_MIN), direction=(-1, 0, 0), limits=TILT_LIMITS),
        cadgen.revolute("rotate", parent="#tilt_stage", child="#rotate_stage",
                        origin=(CX, CY, AXIS_Z_MIN), direction=(0, 1, 0), limits=(-90, 90)),
        cadgen.slider("jaw_right", parent="#rotate_stage", child="#jaw_right",
                      origin=(CX, CY, AXIS_Z_MIN), direction=(1, 0, 0), limits=_J),
        cadgen.slider("jaw_left", parent="#rotate_stage", child="#jaw_left",
                      origin=(CX, CY, AXIS_Z_MIN), direction=(-1, 0, 0), limits=_J),
    ],
    "couplings": [cadgen.couple("jaw_opening", {"jaw_right": 0.5, "jaw_left": 0.5},
                                limits=(JAW_MIN - JAW_AS_BUILT, JAW_MAX - JAW_AS_BUILT))],
    "poses": {
        "desk_low": {"height": 0, "tilt": 20},
        "eye_level": {"height": TRAVEL, "tilt": 10},
        "landscape": {"height": 40, "tilt": 15, "rotate": 90},
    },
}


@step(out="../STEP/phone_stand.step", kinematics=KINEMATICS)
@glb(out="../GLB/phone_stand.glb", kinematics=KINEMATICS)
def phone_stand():
    return build()


if __name__ == "__main__":
    phone_stand()
