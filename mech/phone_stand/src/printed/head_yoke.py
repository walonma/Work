"""轉軸叉座 (PETG). Exported in print orientation."""
from cadgen import step, stl, threemf

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/printed/head_yoke.step")
@stl(out="../../STL/head_yoke.stl")
@threemf(out="../../3MF/head_yoke.3mf")
def head_yoke():
    part = on_bed(parts.head_yoke(), rot=(0, 0, 0))
    part.label = "head_yoke"
    return part


if __name__ == "__main__":
    head_yoke()
