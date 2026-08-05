"""產線風速風場模擬觀測器 — 物理核心。

模擬熱發泡 (TIJ) 噴碼設備在移動箱子上方 3~5mm 處噴印時，
間隙內的氣流場以及 300 顆墨滴的飄移方向。
"""

from .config import SimParams, AirProps, InkProps
from .belt import BeltKinematics
from .windfield import WindField
from .droplet import DropletSwarm
from .simulate import fire_burst, sweep_over_cycle, BurstResult

__all__ = [
    "SimParams",
    "AirProps",
    "InkProps",
    "BeltKinematics",
    "WindField",
    "DropletSwarm",
    "fire_burst",
    "sweep_over_cycle",
    "BurstResult",
]
