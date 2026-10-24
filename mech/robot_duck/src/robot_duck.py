"""機器小鴨概念模型 (Microduck class, 15 joints) with viewer kinematics."""
import cadgen
from cadgen import glb, step

from lib import parts
from lib.assembly_tree import build
from lib.dims import JOINTS


def _mates():
    out = []
    for name, (parent, *_rest) in JOINTS.items():
        origin, axis, lim = parts.joint_axis(name)
        out.append(cadgen.revolute(name, parent=f"#{parent}_stage" if parent else "#pelvis_frame",
                                   child=f"#{name}_stage", origin=origin, direction=axis, limits=lim))
    return out


KINEMATICS = {
    "mates": _mates(),
    "poses": {
        "stand": {},
        "crouch": {"l_hip_pitch": -35, "r_hip_pitch": -35, "l_knee": 70, "r_knee": 70, "l_ankle": -35, "r_ankle": -35},
        "look_left": {"head_yaw": 60, "head_pitch": 12, "neck_pitch": 15},
        "quack": {"beak": 30, "head_pitch": -12, "neck_pitch": -20},
        "step_left": {"l_hip_pitch": -30, "l_knee": 50, "l_ankle": -20, "r_hip_pitch": 15,
                      "l_hip_roll": 10, "r_hip_roll": 10, "l_hip_yaw": 20},
        "nap": {"neck_pitch": 25, "head_pitch": 12, "head_roll": -15, "l_hip_pitch": -55, "r_hip_pitch": -55,
                "l_knee": 85, "r_knee": 85, "l_ankle": -40, "r_ankle": -40},
    },
}


@step(out="../STEP/robot_duck.step", kinematics=KINEMATICS)
@glb(out="../GLB/robot_duck.glb", kinematics=KINEMATICS)
def robot_duck():
    return build()


if __name__ == "__main__":
    robot_duck()
