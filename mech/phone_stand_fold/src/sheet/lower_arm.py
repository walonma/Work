"""下臂 SPCC 1.6 U 型. Exported in its own orientation."""
from cadgen import step

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/sheet/lower_arm.step")
def lower_arm():
    part = on_bed(parts.lower_arm(), rot=(0, 0, 0))
    part.label = "lower_arm"
    return part


if __name__ == "__main__":
    lower_arm()
