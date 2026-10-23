"""外管 25x25x1.5 方管. Exported in assembly orientation."""
from cadgen import step

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/metal/outer_tube.step")
def outer_tube():
    part = on_bed(parts.outer_tube(), rot=(0, 0, 0))
    part.label = "outer_tube"
    return part


if __name__ == "__main__":
    outer_tube()
