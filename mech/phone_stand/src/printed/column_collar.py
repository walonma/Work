"""升降鎖緊套環 (PETG). Exported in print orientation."""
from cadgen import step, stl, threemf

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/printed/column_collar.step")
@stl(out="../../STL/column_collar.stl")
@threemf(out="../../3MF/column_collar.3mf")
def column_collar():
    part = on_bed(parts.column_collar(), rot=(0, 0, 0))
    part.label = "column_collar"
    return part


if __name__ == "__main__":
    column_collar()
