"""捲線器＋線頭座＋充電頭槽 (PETG). Exported in print orientation."""
from cadgen import step, stl, threemf

from lib import parts
from lib.orient import on_bed


@step(out="../../STEP/printed/cable_reel.step")
@stl(out="../../STL/cable_reel.stl")
@threemf(out="../../3MF/cable_reel.3mf")
def cable_reel():
    part = on_bed(parts.cable_reel(), rot=(0, 0, 0))
    part.label = "cable_reel"
    return part


if __name__ == "__main__":
    cable_reel()
