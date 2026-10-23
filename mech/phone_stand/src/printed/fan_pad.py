"""電扇夾板墊塊 (PETG). Exported in print orientation."""
from cadgen import step, stl, threemf

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/printed/fan_pad.step")
@stl(out="../../STL/fan_pad.stl")
@threemf(out="../../3MF/fan_pad.3mf")
def fan_pad():
    part = on_bed(parts.fan_pad(), rot=(180, 0, 0))
    part.label = "fan_pad"
    return part


if __name__ == "__main__":
    fan_pad()
