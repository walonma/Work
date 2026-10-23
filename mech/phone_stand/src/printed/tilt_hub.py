"""仰角轉軸 (PETG). Exported in print orientation."""
from cadgen import step, stl, threemf

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/printed/tilt_hub.step")
@stl(out="../../STL/tilt_hub.stl")
@threemf(out="../../3MF/tilt_hub.3mf")
def tilt_hub():
    part = on_bed(parts.tilt_hub(), rot=(0, 90, 0))
    part.label = "tilt_hub"
    return part


if __name__ == "__main__":
    tilt_hub()
