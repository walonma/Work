"""模擬參數與物理常數。

內部一律使用 SI 單位 (m, s, kg, Pa)。使用者介面單位 (m/min, mm, pL)
在 SimParams 的建構與衍生屬性中換算。

座標系統 (對齊補充圖說的箱子 X/Y/Z)::

      Z (向上)
      ↑
      │   ┌──────────┐  噴嘴板 (z = gap, 靜止)
      │   │▪▪▪▪▪▪▪▪▪▪│  300 噴嘴沿 Y 排列
      │   └──────────┘
      │      ↓↓↓ 墨滴
      │   - - - 3~5mm 間隙 - - -
      │  ▓▓▓▓▓ 箱頂面 (z = 0, 以 U_box 移動) ▓▓▓▓
      └──────────────────────────→ X (箱子前進方向)

X 為箱子前進方向 (實際產線右→左, 繪圖時鏡射)。
Y 為橫向, 即 300 噴嘴的排列方向, 原點在陣列中心。
Z 垂直向上, z=0 為箱頂面, z=gap 為噴嘴板。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

INCH = 0.0254  # m


@dataclass(frozen=True)
class AirProps:
    """空氣物性 (20°C, 1 atm)。"""

    density: float = 1.204  # kg/m^3
    viscosity: float = 1.813e-5  # Pa*s (動力黏度 mu)

    @property
    def kinematic_viscosity(self) -> float:
        """運動黏度 nu = mu / rho。"""
        return self.viscosity / self.density


@dataclass(frozen=True)
class InkProps:
    """墨水物性 (水性 TIJ 墨水典型值)。"""

    density: float = 1050.0  # kg/m^3


GRAVITY = 9.80665  # m/s^2


def _check(name: str, value: float, low: float, high: float, unit: str = "") -> None:
    if not (low <= value <= high):
        raise ValueError(
            f"{name} = {value}{unit} 超出允許範圍 [{low}, {high}]{unit}"
        )


@dataclass
class SimParams:
    """模擬器的完整參數集。

    規格範圍 (依需求書):
      - 產線速度 0 ~ 350 m/min
      - 箱重 5 ~ 50 kg
      - 噴嘴間隙 3 ~ 5 mm
      - 墨滴體積 10 ~ 50 pL
      - 300 個噴嘴
      - 箱間距 30 ~ 50 cm (箱子連續不間斷)
    """

    # --- 使用者可調的主要參數 ---
    belt_speed_m_min: float = 350.0
    box_mass_kg: float = 20.0
    gap_mm: float = 4.0
    droplet_pl: float = 30.0
    box_spacing_cm: float = 40.0

    # --- 噴頭幾何 ---
    n_nozzles: int = 300
    print_height_in: float = 1.0
    """噴印帶高度 (吋)。1.0 -> 300 dpi (點距 84.67um);
    0.5 -> 600 dpi (點距 42.3um), 同樣飄移量的嚴重度加倍。"""
    plate_x_mm: float = 15.0
    """噴嘴板在 X 方向 (沿產線) 的寬度。指墨匣底部正對箱子、
    隔著間隙的整片平面 —— 不是噴孔直徑 (噴孔僅 20~30um,
    對氣流拘束毫無作用)。決定擠壓膜的拘束範圍與前緣脈衝強度。"""
    head_y_offset_mm: float = 0.0
    """噴嘴陣列中心相對箱子中線的 Y 偏移。0 = 對準中線。"""
    eject_velocity: float = 8.0  # m/s, TIJ 典型 3~10
    fire_frequency_hz: float = 15000.0

    # --- 箱體幾何 ---
    box_length_m: float = 0.5
    box_width_m: float = 0.5
    box_height_m: float = 0.5

    # --- 風場校正係數 (可由實測反推校正) ---
    c_leading: float = 0.30
    """前緣壓力脈衝耦合係數。"""
    c_trailing: float = 0.15
    """後緣尾流回吸係數 (負向, 較弱較寬)。"""
    sigma_edge_mm: float = 15.0
    """前緣脈衝的高斯包絡長度尺度。"""
    c_lateral: float = 0.35
    """橫向擠壓膜逸流係數 (脈衝驅動)。"""
    c_lateral_steady: float = 0.05
    """穩態橫向逸流係數 (Couette 泵送驅動)。"""
    p_lateral: float = 1.0
    """橫向逸流的 y 次方。1.0 = 均勻源擠壓膜的古典線性解。"""
    c_vertical: float = 0.15
    """前緣處的向上抬升係數 (空氣被推上箱頂)。"""
    c_wake: float = 0.25
    """箱間開放區的尾流回流強度。"""
    wake_length_m: float = 0.15
    """尾流衰減長度。30cm 間距時前一箱尾流尚未消散,
    下一箱前緣即抵達 -> 兩者干涉。"""

    # --- 皮帶傳動 / 質量耦合 ---
    drive_force_max_n: float = 300.0
    """皮帶驅動的最大推力。箱越重, 整線加速越慢。"""
    belt_inertia_kg: float = 20.0
    """皮帶與滾輪的等效慣量。"""
    friction_mu: float = 0.60
    """箱子與皮帶間的摩擦係數 (紙箱 vs 橡膠帶)。
    箱子最大可承受加速度 = mu * g; 超過即打滑。"""
    bounce_amp_ref_mm: float = 0.15
    """5 kg 參考箱重下的跳動振幅。振幅 ∝ 1/m。"""
    bounce_mass_ref_kg: float = 5.0
    bounce_stiffness_n_m: float = 6.16e5
    """皮帶支撐的等效剛度, 決定跳動頻率 f = sqrt(k/m)/2pi。"""

    # --- 物性 ---
    air: AirProps = field(default_factory=AirProps)
    ink: InkProps = field(default_factory=InkProps)

    # --- 數值 ---
    dt_droplet: float = 1e-6  # s
    max_flight_steps: int = 4000

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check("產線速度", self.belt_speed_m_min, 0.0, 350.0, " m/min")
        _check("箱重", self.box_mass_kg, 5.0, 50.0, " kg")
        _check("噴嘴間隙", self.gap_mm, 3.0, 5.0, " mm")
        _check("墨滴體積", self.droplet_pl, 10.0, 50.0, " pL")
        _check("箱間距", self.box_spacing_cm, 30.0, 50.0, " cm")
        if self.n_nozzles < 1:
            raise ValueError(f"噴嘴數必須 >= 1, 得到 {self.n_nozzles}")
        if self.plate_x_mm <= 0:
            raise ValueError("噴嘴板寬度必須為正")
        if self.box_length_m <= 0 or self.box_width_m <= 0 or self.box_height_m <= 0:
            raise ValueError("箱體尺寸必須為正")

    def replace(self, **kwargs) -> "SimParams":
        """回傳一份改了部分欄位的副本 (會重新驗證)。"""
        return replace(self, **kwargs)

    # ------------------------------------------------------------------
    # 衍生量 — 產線 / 幾何
    # ------------------------------------------------------------------
    @property
    def belt_speed(self) -> float:
        """產線速度 (m/s)。"""
        return self.belt_speed_m_min / 60.0

    @property
    def gap(self) -> float:
        """標稱噴嘴間隙 (m)。"""
        return self.gap_mm / 1000.0

    @property
    def box_spacing(self) -> float:
        """箱間距 (m)。"""
        return self.box_spacing_cm / 100.0

    @property
    def train_pitch(self) -> float:
        """箱列的空間週期 = 箱長 + 箱間距 (m)。"""
        return self.box_length_m + self.box_spacing

    @property
    def cycle_period(self) -> float:
        """箱列循環的時間週期 (s)。速度為零時回傳 inf。"""
        if self.belt_speed <= 0:
            return math.inf
        return self.train_pitch / self.belt_speed

    @property
    def cycle_frequency_hz(self) -> float:
        """擾動頻率 (Hz)。350 m/min + 30~50cm 間距 -> 5.8~7.3 Hz。"""
        period = self.cycle_period
        return 0.0 if math.isinf(period) else 1.0 / period

    @property
    def plate_x(self) -> float:
        """噴嘴板 X 向寬度 (m)。"""
        return self.plate_x_mm / 1000.0

    @property
    def sigma_edge(self) -> float:
        """前緣脈衝包絡長度 (m)。"""
        return self.sigma_edge_mm / 1000.0

    # ------------------------------------------------------------------
    # 衍生量 — 噴嘴陣列
    # ------------------------------------------------------------------
    @property
    def print_height(self) -> float:
        """噴印帶高度 (m)。"""
        return self.print_height_in * INCH

    @property
    def nozzle_pitch(self) -> float:
        """噴嘴間距 (m)。1 吋 / 300 = 84.67um (300 dpi)。"""
        return self.print_height / self.n_nozzles

    @property
    def dpi(self) -> float:
        """Y 方向解析度 (dpi)。"""
        return self.n_nozzles / self.print_height_in

    @property
    def plate_half_y(self) -> float:
        """噴嘴板的 Y 向半長 (m) —— 橫向擠壓膜逸流的特徵長度。

        關鍵: 這是 12.7mm (噴嘴板半長), 不是 250mm (箱半寬)。
        噴嘴板是塊小平板, 空氣從底下往四面八方逸散, 側向只需走 12.7mm。
        """
        return self.print_height / 2.0

    def nozzle_y(self):
        """回傳 300 個噴嘴的 Y 座標 (m), 以陣列中心為原點。"""
        import numpy as np

        idx = np.arange(self.n_nozzles)
        centred = (idx - (self.n_nozzles - 1) / 2.0) * self.nozzle_pitch
        return centred + self.head_y_offset_mm / 1000.0

    # ------------------------------------------------------------------
    # 衍生量 — 墨滴
    # ------------------------------------------------------------------
    @property
    def droplet_volume(self) -> float:
        """墨滴體積 (m^3)。1 pL = 1e-15 m^3。"""
        return self.droplet_pl * 1e-15

    @property
    def droplet_diameter(self) -> float:
        """墨滴直徑 (m)。10 pL -> 26.7um; 50 pL -> 45.7um。"""
        return (6.0 * self.droplet_volume / math.pi) ** (1.0 / 3.0)

    @property
    def droplet_mass(self) -> float:
        """單顆墨滴質量 (kg)。"""
        return self.ink.density * self.droplet_volume

    @property
    def relaxation_time(self) -> float:
        """墨滴的速度鬆弛時間 tau = rho_p * d^2 / (18 mu) (s)。

        30um 墨滴約 2.9 ms, 遠大於 ~0.55ms 的飛行時間, 因此墨滴
        來不及跟隨氣流 (Stokes 數 ~5), 必須用完整彈道+阻力模型。
        """
        d = self.droplet_diameter
        return self.ink.density * d * d / (18.0 * self.air.viscosity)

    # ------------------------------------------------------------------
    # 衍生量 — 無因次數與診斷
    # ------------------------------------------------------------------
    @property
    def gap_reynolds(self) -> float:
        """間隙 Reynolds 數 Re_h = U*h/nu。350 m/min + 4mm -> ~1550。"""
        return self.belt_speed * self.gap / self.air.kinematic_viscosity

    def estimated_flight_time(self) -> float:
        """粗估飛行時間 (s) —— 純用於 Stokes 數診斷與時間軸配置。"""
        if self.eject_velocity <= 0:
            return math.sqrt(2.0 * self.gap / GRAVITY)
        return self.gap / self.eject_velocity

    @property
    def stokes_number(self) -> float:
        """Stokes 數 St = tau / t_flight。>1 表示墨滴以彈道為主。"""
        t_f = self.estimated_flight_time()
        return self.relaxation_time / t_f if t_f > 0 else math.inf

    @property
    def box_travel_during_flight(self) -> float:
        """飛行期間箱子的位移 (m)。

        350 m/min 下約 3.2mm, 遠大於風場飄移的 ~0.15mm。實務上由噴發
        時序補償, 因此飄移誤差必須以「已補償的標稱落點」為基準來計算。
        """
        return self.belt_speed * self.estimated_flight_time()

    @property
    def x_dot_pitch(self) -> float:
        """X 方向的實際點距 (m) = 速度 / 噴發頻率。"""
        if self.fire_frequency_hz <= 0:
            return math.inf
        return self.belt_speed / self.fire_frequency_hz

    @property
    def x_dpi(self) -> float:
        """X 方向的實際解析度 (dpi)。"""
        pitch = self.x_dot_pitch
        return 0.0 if (math.isinf(pitch) or pitch <= 0) else INCH / pitch

    @property
    def required_fire_frequency_hz(self) -> float:
        """要讓 X 向解析度等於 Y 向所需的噴發頻率 (Hz)。

        350 m/min + 300 dpi 需要 68.9 kHz, 遠超過 TIJ 的 12~15 kHz 上限。
        """
        return self.belt_speed / self.nozzle_pitch

    def summary(self) -> str:
        """人可讀的參數與診斷摘要。"""
        lines = [
            "=== 產線風場模擬參數 ===",
            f"  產線速度      : {self.belt_speed_m_min:.1f} m/min ({self.belt_speed:.3f} m/s)",
            f"  箱重          : {self.box_mass_kg:.1f} kg",
            f"  噴嘴間隙      : {self.gap_mm:.2f} mm",
            f"  墨滴體積      : {self.droplet_pl:.1f} pL (直徑 {self.droplet_diameter*1e6:.1f} um)",
            f"  箱間距        : {self.box_spacing_cm:.1f} cm",
            f"  箱體          : {self.box_length_m*100:.0f} x {self.box_width_m*100:.0f} x {self.box_height_m*100:.0f} cm",
            "--- 噴頭 ---",
            f"  噴嘴數        : {self.n_nozzles} @ {self.nozzle_pitch*1e6:.2f} um 間距 ({self.dpi:.0f} dpi)",
            f"  噴印帶高度    : {self.print_height_in} 吋 ({self.print_height*1000:.1f} mm)",
            f"  噴嘴板 X 寬   : {self.plate_x_mm:.1f} mm",
            f"  噴發初速      : {self.eject_velocity:.1f} m/s",
            "--- 診斷 ---",
            f"  間隙 Reynolds : {self.gap_reynolds:.0f}",
            f"  間隙平均風速  : {self.belt_speed/2:.2f} m/s",
            f"  墨滴鬆弛時間  : {self.relaxation_time*1e3:.2f} ms",
            f"  估計飛行時間  : {self.estimated_flight_time()*1e3:.3f} ms",
            f"  Stokes 數     : {self.stokes_number:.2f} ({'彈道主導' if self.stokes_number > 1 else '跟隨氣流'})",
            f"  飛行期間箱移  : {self.box_travel_during_flight*1000:.2f} mm",
            f"  循環週期      : {self.cycle_period*1e3:.1f} ms ({self.cycle_frequency_hz:.2f} Hz)",
            f"  X 向點距      : {self.x_dot_pitch*1e6:.0f} um ({self.x_dpi:.0f} dpi)"
            f" @ {self.fire_frequency_hz/1000:.0f} kHz",
            f"  X 向需要頻率  : {self.required_fire_frequency_hz/1000:.1f} kHz 才能達到 {self.dpi:.0f} dpi",
        ]
        return "\n".join(lines)
