"""左夾爪 (PETG). Exported in print orientation."""
from cadgen import step, stl, threemf

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/printed/jaw_left.step")
@stl(out="../../STL/jaw_left.stl")
@threemf(out="../../3MF/jaw_left.3mf")
def jaw_left():
    part = on_bed(parts.jaw(-1), rot=(0, -90, 0))
    part.label = "jaw_left"
    return part


if __name__ == "__main__":
    jaw_left()
