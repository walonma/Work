"""上臂 SPCC 1.6 U 型. Exported in its own orientation."""
from cadgen import step

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/sheet/upper_arm.step")
def upper_arm():
    part = on_bed(parts.upper_arm(), rot=(0, 0, 0))
    part.label = "upper_arm"
    return part


if __name__ == "__main__":
    upper_arm()
