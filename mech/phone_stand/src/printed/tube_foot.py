"""外管底塞 (PETG). Exported in print orientation."""
from cadgen import step, stl, threemf

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/printed/tube_foot.step")
@stl(out="../../STL/tube_foot.stl")
@threemf(out="../../3MF/tube_foot.3mf")
def tube_foot():
    part = on_bed(parts.tube_foot(), rot=(0, 0, 0))
    part.label = "tube_foot"
    return part


if __name__ == "__main__":
    tube_foot()
