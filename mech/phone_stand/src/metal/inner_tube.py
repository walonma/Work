"""內管 20x20x1.5 方管. Exported in assembly orientation."""
from cadgen import step

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/metal/inner_tube.step")
def inner_tube():
    part = on_bed(parts.inner_tube(), rot=(0, 0, 0))
    part.label = "inner_tube"
    return part


if __name__ == "__main__":
    inner_tube()
