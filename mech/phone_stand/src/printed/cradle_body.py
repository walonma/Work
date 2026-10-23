"""手機托架本體 (PETG). Exported in print orientation."""
from cadgen import step, stl, threemf

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/printed/cradle_body.step")
@stl(out="../../STL/cradle_body.stl")
@threemf(out="../../3MF/cradle_body.3mf")
def cradle_body():
    part = on_bed(parts.cradle_body(), rot=(0, 0, 0))
    part.label = "cradle_body"
    return part


if __name__ == "__main__":
    cradle_body()
