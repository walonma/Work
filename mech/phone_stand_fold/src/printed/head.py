"""手機頭座 (PETG). Exported in print orientation."""
from cadgen import step, stl, threemf

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/printed/head.step")
@stl(out="../../STL/head.stl")
@threemf(out="../../3MF/head.3mf")
def head():
    part = on_bed(parts.head(), rot=(180, 0, 0))
    part.label = "head"
    return part


if __name__ == "__main__":
    head()
