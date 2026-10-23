"""底座 SPCC 2.0. Exported in assembly orientation."""
from cadgen import step

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/metal/base_plate.step")
def base_plate():
    part = on_bed(parts.base_plate(), rot=(0, 0, 0))
    part.label = "base_plate"
    return part


if __name__ == "__main__":
    base_plate()
