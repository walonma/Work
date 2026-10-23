"""出貨狀態（折平裝盒） - folded, as packed in the box."""
from cadgen import step

from lib.assembly_tree import build
from lib.dims import JAW_MIN


@step(out="../STEP/phone_stand_folded.step")
def phone_stand_folded():
    return build(0.0, 0.0, 0.0, opening=JAW_MIN, refs=False)   # jaws spring shut when empty


if __name__ == "__main__":
    phone_stand_folded()
