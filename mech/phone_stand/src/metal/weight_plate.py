"""配重板 鋼板 6t. Exported in assembly orientation."""
from cadgen import step

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/metal/weight_plate.step")
def weight_plate():
    part = on_bed(parts.weight_plate(), rot=(0, 0, 0))
    part.label = "weight_plate"
    return part


if __name__ == "__main__":
    weight_plate()
