"""折疊式手機支架總組立 (deployed pose) with viewer kinematics."""
import cadgen
from cadgen import glb, step

from lib.assembly_tree import build
from lib.dims import (ELBOW, ELBOW_RANGE, HEAD, HEAD_AXIS_OFF, HEAD_PLATE_Z, HEAD_RANGE, JAW_AS_BUILT,
                      JAW_MAX, JAW_MIN, SHOULDER, SHOULDER_RANGE, add, chain, rot_x)


def _rel(rng, at):
    return (rng[0] - at, rng[1] - at)


_C = chain()
_ROT_O = add(_C["h3"], rot_x((0.0, HEAD_AXIS_OFF, HEAD_PLATE_Z[1]), _C["head"]))
_ROT_D = rot_x((0.0, 0.0, 1.0), _C["head"])
_J = ((JAW_MIN - JAW_AS_BUILT) / 2, (JAW_MAX - JAW_AS_BUILT) / 2)

FOLDED = {"shoulder": -SHOULDER, "elbow": -ELBOW, "head": -HEAD}

KINEMATICS = {
    "mates": [
        cadgen.revolute("shoulder", parent="#base_plate", child="#lower_stage",
                        origin=_C["h1"], direction=(1, 0, 0), limits=_rel(SHOULDER_RANGE, SHOULDER)),
        cadgen.revolute("elbow", parent="#lower_stage", child="#upper_stage",
                        origin=_C["h2"], direction=(-1, 0, 0), limits=_rel(ELBOW_RANGE, ELBOW)),
        cadgen.revolute("head", parent="#upper_stage", child="#head_stage",
                        origin=_C["h3"], direction=(1, 0, 0), limits=_rel(HEAD_RANGE, HEAD)),
        cadgen.revolute("rotate", parent="#head_stage", child="#rotate_stage",
                        origin=_ROT_O, direction=_ROT_D, limits=(-90, 90)),
        cadgen.slider("jaw_right", parent="#rotate_stage", child="#jaw_right",
                      origin=_ROT_O, direction=(1, 0, 0), limits=_J),
        cadgen.slider("jaw_left", parent="#rotate_stage", child="#jaw_left",
                      origin=_ROT_O, direction=(-1, 0, 0), limits=_J),
    ],
    "couplings": [cadgen.couple("jaw_opening", {"jaw_right": 0.5, "jaw_left": 0.5},
                                limits=(JAW_MIN - JAW_AS_BUILT, JAW_MAX - JAW_AS_BUILT))],
    "poses": {
        "folded": FOLDED,
        "use": {},
        "low": {"shoulder": -30, "elbow": -45, "head": 15},
        "high": {"shoulder": 10, "elbow": -5, "head": -15},
        "landscape": {"rotate": 90},
    },
}


@step(out="../STEP/phone_stand_fold.step", kinematics=KINEMATICS)
@glb(out="../GLB/phone_stand_fold.glb", kinematics=KINEMATICS)
def phone_stand_fold():
    return build()


if __name__ == "__main__":
    phone_stand_fold()
