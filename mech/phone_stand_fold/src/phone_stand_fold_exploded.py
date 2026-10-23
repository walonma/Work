"""爆炸圖（拆件） - exploded view."""
from cadgen import step

from lib.assembly_tree import build


@step(out="../STEP/phone_stand_fold_exploded.step")
def phone_stand_fold_exploded():
    return build(explode=50.0, refs=False)


if __name__ == "__main__":
    phone_stand_fold_exploded()
