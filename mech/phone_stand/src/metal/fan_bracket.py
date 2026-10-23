"""電扇夾板支架 SPCC 2.0. Exported in assembly orientation."""
from cadgen import step

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/metal/fan_bracket.step")
def fan_bracket():
    part = on_bed(parts.fan_bracket(), rot=(0, 0, 0))
    part.label = "fan_bracket"
    return part


if __name__ == "__main__":
    fan_bracket()
