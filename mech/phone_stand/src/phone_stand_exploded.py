"""手機支架爆炸圖 (exploded view for part breakdown)."""
from cadgen import step

from lib.assembly_tree import build


@step(out="../STEP/phone_stand_exploded.step")
def phone_stand_exploded():
    return build(explode=60.0)


if __name__ == "__main__":
    phone_stand_exploded()
