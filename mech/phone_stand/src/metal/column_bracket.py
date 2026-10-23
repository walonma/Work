"""立柱座 SPCC 2.0. Exported in assembly orientation."""
from cadgen import step

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/metal/column_bracket.step")
def column_bracket():
    part = on_bed(parts.column_bracket(), rot=(0, 0, 0))
    part.label = "column_bracket"
    return part


if __name__ == "__main__":
    column_bracket()
