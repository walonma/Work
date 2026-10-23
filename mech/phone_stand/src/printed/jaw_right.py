"""右夾爪 (PETG). Exported in print orientation."""
from cadgen import step, stl, threemf

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/printed/jaw_right.step")
@stl(out="../../STL/jaw_right.stl")
@threemf(out="../../3MF/jaw_right.3mf")
def jaw_right():
    part = on_bed(parts.jaw(+1), rot=(0, 90, 0))
    part.label = "jaw_right"
    return part


if __name__ == "__main__":
    jaw_right()
