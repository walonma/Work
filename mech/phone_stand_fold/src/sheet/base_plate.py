"""底座（含電扇夾板、轉軸耳） SPCC 1.6. Exported in its own orientation."""
from cadgen import step

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/sheet/base_plate.step")
def base_plate():
    part = on_bed(parts.base_plate(), rot=(0, 0, 0))
    part.label = "base_plate"
    return part


if __name__ == "__main__":
    base_plate()
