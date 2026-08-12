"""導風擋板 / 擾流板 —— 幾何參數與密封效果的解析求解。

設計目的是在噴孔與箱面之間製造「無風區」。但這裡有一個物理硬限制必須
講清楚：

    箱頂面是以 U 移動的實體壁面, no-slip 條件強制緊貼箱面的空氣必須
    跟著跑 U。只要箱子在動, 間隙裡就一定有剪切流 —— 任何擋板都擋不掉。

擋板真正能做到的是**把單向貫流變成零淨流量的循環流**。若上下游都封起來,
間隙成為封閉腔體, 質量守恆強制回流, 速度剖面從線性 Couette 變成::

    u(ζ) = U(1−ζ) − 3·U·σ·ζ(1−ζ)          ζ = z/h

σ 為密封有效度。σ=0 是完全開放 (原本的 Couette); σ=1 時淨流量精確為零::

    ζ=0   (箱面)   -> +U        仍是滿速, 這是 no-slip 的必然
    ζ=1/3          ->  0
    ζ=2/3          -> −U/3      逆向流
    ζ=1   (噴嘴板) ->  0
    ∫u dζ = 0

墨滴從噴嘴板往下飛, 先穿過上方 2/3 的逆向流 (往後推), 再穿過下方 1/3
的正向流 (往前推), 兩者互相抵消 —— 這才是擋板降低飄移的真正機制。

因此存在一個**最佳密封程度使淨飄移為零**: 密封不足殘留正向飄移,
密封過度變成負向飄移。sim.simulate.find_zero_drift_clearance() 可解出來。


σ 怎麼算 —— 一維潤滑理論的串聯網路
=====================================

把間隙沿 X 切成數段, 第 i 段高度 hᵢ、長度 Lᵢ。每段的體積流率 (單位 Y 寬)::

    Q = U·hᵢ/2 − (hᵢ³/12μ)·(dp/dx)ᵢ        Couette 拖曳 + 壓差驅動

連續方程要求各段 Q 相同。兩端皆通大氣, 故沿路徑的總壓力變化為零::

    Σ (dp/dx)ᵢ·Lᵢ = 0

解出::

    Q = (U/2) · [Σ Lᵢ/hᵢ²] / [Σ Lᵢ/hᵢ³]  ≡  U·δ_eff/2
    σ = 1 − δ_eff/h

無擋板時只有一段, δ_eff = h, σ = 0 —— 精確退回原本的 Couette 解。
下緣間隙 δ→0 時 δ_eff→δ, σ→1。

同一個網路也給出壓差驅動流 (前後緣脈衝) 的衰減倍率::

    pulse_factor = (L_cav/h³) / [Σ Lᵢ/hᵢ³]

無擋板時為 1。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

INCH = 0.0254

# --- 需要實測校正的經驗係數 (與 windfield 的 c_leading 等同一性質) ---
K_VENT_LEAK = 0.5
"""洩壓孔造成的旁通洩漏, 直接扣減 σ。"""
K_VENT_RELIEF = 0.8
"""洩壓孔對前緣壓力脈衝的釋放效果。"""
K_ANGLE_DEFLECT = 0.5
"""斜角把來流導開的效果 (乘上 sin θ)。"""
K_SEPARATION = 0.08
"""銳利前緣脫落渦流造成的亂流強度 (乘上 U)。"""


@dataclass
class BaffleParams:
    """擋板 / 擾流板的設計參數。

    座標與主模型一致: X 沿產線, Y 沿噴嘴陣列, Z 垂直。
    擋板從噴頭往下延伸, 下緣與箱面留 clearance 的間隙。
    """

    enabled: bool = False

    # --- 配置 (可獨立切換以便比較) ---
    upstream: bool = True
    """上游側擋板 (箱子先遇到的那一側)。"""
    downstream: bool = True
    """下游側擋板。要形成封閉腔體、產生回流, 上下游都必須有。"""
    y_end_plates: bool = True
    """兩側 Y 端板。抑制橫向擠壓膜逸流 (外八圖樣)。"""

    # --- 主要幾何 ---
    clearance_mm: float = 1.5
    """下緣與箱面的間隙。密封效果的主導參數 (∝ δ³), 但也是撞擊風險來源。"""
    thickness_mm: float = 1.0
    """單片擋板厚度。"""
    angle_deg: float = 20.0
    """擋板相對垂直方向的傾角。>0 表示把來流往上導開。"""
    width_in: float = 1.25
    """擋板在 Y 方向的總寬度 (吋), 規格範圍 1~1.5。
    比噴印帶 (1 吋) 寬時, 橫向逸流出口外移, 陣列上的橫向梯度變小。"""
    x_extent_mm: float = 25.0
    """擋板組件在 X 方向的內腔長度, 即上下游擋板內側面的距離。"""

    # --- 擾流板設計參數 ---
    n_stages: int = 2
    """每側的串列片數 (迷宮式密封)。"""
    stage_pitch_mm: float = 2.5
    """串列片之間的間距 (級間腔室長度)。"""
    edge_radius_mm: float = 0.3
    """前緣圓角半徑。銳利邊緣會脫落渦流, 若分離剪切層落在噴印區
    會產生非穩態亂流 —— 比原本穩定的剪切更難處理。"""
    vent_ratio: float = 0.0
    """排氣/洩壓孔的開孔面積比 (0~0.3)。讓前緣壓力脈衝有路可洩,
    但同時造成旁通洩漏而降低密封效果。"""

    # --- 機械餘裕 ---
    box_height_tolerance_mm: float = 0.5
    """箱高公差 / 箱頂面不平度。與箱體跳動一起決定撞擊風險。"""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not 1.0 <= self.width_in <= 1.5:
            raise ValueError(
                f"擋板寬度 = {self.width_in} 吋 超出允許範圍 [1.0, 1.5] 吋"
            )
        if not 0.3 <= self.clearance_mm <= 3.0:
            raise ValueError(
                f"下緣間隙 = {self.clearance_mm} mm 超出允許範圍 [0.3, 3.0] mm"
            )
        if self.thickness_mm <= 0:
            raise ValueError("擋板厚度必須為正")
        if not 0.0 <= self.angle_deg <= 60.0:
            raise ValueError(f"斜角 = {self.angle_deg}° 超出允許範圍 [0, 60]°")
        if self.n_stages < 1:
            raise ValueError("串列片數必須 >= 1")
        if self.stage_pitch_mm <= 0:
            raise ValueError("片間距必須為正")
        if self.edge_radius_mm < 0:
            raise ValueError("前緣圓角半徑不可為負")
        if self.edge_radius_mm > self.thickness_mm / 2.0 + 1e-9:
            # 幾何約束: 半徑最大只能等於半厚 (完全圓化的鼻端)
            raise ValueError(
                f"前緣圓角半徑 {self.edge_radius_mm} mm 不可超過厚度的一半"
                f" ({self.thickness_mm/2:.3f} mm) —— 幾何上做不出來。"
                f"要更大的圓角必須先加厚擋板。"
            )
        if not 0.0 <= self.vent_ratio <= 0.3:
            raise ValueError(f"洩壓孔面積比 = {self.vent_ratio} 超出允許範圍 [0, 0.3]")
        if self.x_extent_mm <= 0:
            raise ValueError("X 方向內腔長度必須為正")

    def replace(self, **kwargs) -> "BaffleParams":
        return replace(self, **kwargs)

    # ------------------------------------------------------------------
    # 衍生幾何
    # ------------------------------------------------------------------
    @property
    def clearance(self) -> float:
        return self.clearance_mm / 1000.0

    @property
    def thickness(self) -> float:
        return self.thickness_mm / 1000.0

    @property
    def angle(self) -> float:
        return math.radians(self.angle_deg)

    @property
    def width(self) -> float:
        """Y 方向總寬度 (m)。"""
        return self.width_in * INCH

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    @property
    def x_extent(self) -> float:
        return self.x_extent_mm / 1000.0

    @property
    def stage_pitch(self) -> float:
        return self.stage_pitch_mm / 1000.0

    @property
    def edge_radius(self) -> float:
        return self.edge_radius_mm / 1000.0

    @property
    def n_sides(self) -> int:
        """有擋板的 X 側數 (0/1/2)。要形成回流需要 2。"""
        return int(self.upstream) + int(self.downstream)

    @property
    def active(self) -> bool:
        """是否有任何擋板生效。"""
        return self.enabled and (self.n_sides > 0 or self.y_end_plates)

    @property
    def constriction_length(self) -> float:
        """單片擋板在 X 方向的收縮段長度 (m)。

        傾角使投影長度變長: L_b = t / cos θ。
        """
        return self.thickness / math.cos(self.angle)

    @property
    def total_x_footprint(self) -> float:
        """整組擋板在 X 方向的總佔用長度 (m) —— 供拘束因子與繪圖使用。"""
        if not self.active:
            return 0.0
        per_side = (
            self.n_stages * self.constriction_length
            + max(self.n_stages - 1, 0) * self.stage_pitch
        )
        return self.x_extent + self.n_sides * per_side


# ----------------------------------------------------------------------
# 密封求解
# ----------------------------------------------------------------------
@dataclass
class SealingResult:
    """給定間隙下的擋板氣動效果。"""

    sigma: float
    """密封有效度 ∈ [0, 1]。0 = 完全開放的 Couette; 1 = 零淨流量循環流。"""
    pulse_factor: float
    """前/後緣壓差脈衝的衰減倍率 ∈ (0, 1]。無擋板時為 1。"""
    delta_eff: float
    """等效通流高度 (m)。淨流量 Q = U·δ_eff/2。"""
    sigma_y: float
    """Y 方向 (橫向逸流) 的密封有效度。"""
    lateral_half_length: float
    """橫向逸流的特徵半長 (m) —— 取最寬的正對箱面實體表面。"""
    turbulence_intensity: float
    """銳利前緣脫落渦流造成的亂流強度 (相對 U 的比例)。"""

    @property
    def net_flux_fraction(self) -> float:
        """淨流量相對無擋板時的比例 = 1 − σ。"""
        return 1.0 - self.sigma


def _series_network(segments):
    """一維潤滑串聯網路。

    segments: [(長度, 高度), ...]
    回傳 (Σ L/h², Σ L/h³)。
    """
    s2 = 0.0
    s3 = 0.0
    for length, height in segments:
        s2 += length / (height * height)
        s3 += length / (height * height * height)
    return s2, s3


def solve_sealing(baffle: BaffleParams, gap: float, plate_half_y: float) -> SealingResult:
    """求解給定瞬時間隙下的擋板密封效果。

    gap          : 噴嘴板到箱面的間隙 h (m)
    plate_half_y : 無擋板時的橫向逸流特徵半長 (m), 即噴嘴板半長
    """
    h = np.asarray(gap, dtype=float)

    if not baffle.active:
        # 精確退回無擋板的 Couette 解
        zero = np.zeros_like(h)
        return SealingResult(
            sigma=zero,
            pulse_factor=np.ones_like(h),
            delta_eff=h,
            sigma_y=zero,
            lateral_half_length=plate_half_y,
            turbulence_intensity=zero,
        )

    # 下緣間隙不可能大於間隙本身 (擋板不會伸到箱面以下)
    delta = np.minimum(baffle.clearance, h)
    l_b = baffle.constriction_length
    n = baffle.n_stages

    # --- X 方向的串聯網路 ---
    cavity = [(baffle.x_extent, h)]
    per_side = [(l_b, delta)] * n + [(baffle.stage_pitch, h)] * max(n - 1, 0)
    segments = cavity + per_side * baffle.n_sides

    s2, s3 = _series_network(segments)
    delta_eff = s2 / s3
    sigma = 1.0 - delta_eff / h

    # 壓差驅動流 (前後緣脈衝) 的衰減: 只看阻力比
    cavity_s3 = baffle.x_extent / (h ** 3)
    pulse_factor = cavity_s3 / s3

    # 斜角把來流往上導開, 進一步削弱脈衝
    pulse_factor *= 1.0 - K_ANGLE_DEFLECT * math.sin(baffle.angle)

    # 洩壓孔: 釋放脈衝, 但造成旁通洩漏而降低密封
    if baffle.vent_ratio > 0:
        pulse_factor *= 1.0 - K_VENT_RELIEF * baffle.vent_ratio
        sigma -= K_VENT_LEAK * baffle.vent_ratio

    sigma = np.clip(sigma, 0.0, 1.0)
    pulse_factor = np.clip(pulse_factor, 0.0, 1.0)

    # --- Y 方向: 端板對橫向逸流的抑制, 同一套 δ³ 機制 ---
    if baffle.y_end_plates:
        y_segments = [(baffle.half_width, h)] + [(l_b, delta)] * n
        y2, y3 = _series_network(y_segments)
        sigma_y = np.clip(1.0 - (y2 / y3) / h, 0.0, 1.0)
    else:
        sigma_y = np.zeros_like(h)

    # 擋板比噴印帶寬時, 橫向逸流出口外移 -> 陣列上的梯度變小
    lateral_half_length = max(baffle.half_width, plate_half_y)

    # --- 前緣圓角: 抑制分離渦流 ---
    # 圓角需與下緣間隙同量級才能有效抑制分離
    r_crit = 0.5 * delta
    sharpness = np.clip(1.0 - baffle.edge_radius / np.maximum(r_crit, 1e-12), 0.0, 1.0)
    turbulence = K_SEPARATION * sharpness if baffle.n_sides > 0 else np.zeros_like(h)

    return SealingResult(
        sigma=sigma,
        pulse_factor=pulse_factor,
        delta_eff=delta_eff,
        sigma_y=sigma_y,
        lateral_half_length=lateral_half_length,
        turbulence_intensity=turbulence,
    )


# ----------------------------------------------------------------------
# 撞擊風險
# ----------------------------------------------------------------------
@dataclass
class CollisionCheck:
    """擋板下緣與箱面的機械餘裕評估。"""

    clearance_mm: float
    required_mm: float
    margin_mm: float
    level: str  # "safe" | "warn" | "collision"
    message: str

    @property
    def ok(self) -> bool:
        return self.level == "safe"


def check_collision(
    baffle: BaffleParams,
    bounce_amplitude_mm: float,
) -> CollisionCheck:
    """擋板下緣間隙是否留得住箱體跳動與箱高公差。

    箱體跳動振幅 ∝ 1/箱重 (輕箱跳得明顯), 由 BeltKinematics 提供。
    """
    required = bounce_amplitude_mm + baffle.box_height_tolerance_mm
    margin = baffle.clearance_mm - required

    if margin >= 0.3:
        level = "safe"
        msg = f"餘裕 {margin:.2f} mm，安全"
    elif margin >= 0.0:
        level = "warn"
        msg = (
            f"餘裕僅 {margin:.2f} mm —— 箱頂面稍有翹曲或貼標凸起就會刮到，"
            f"建議下緣間隙 ≥ {required + 0.3:.2f} mm"
        )
    else:
        level = "collision"
        msg = (
            f"會撞擊：需要 {required:.2f} mm (跳動 {bounce_amplitude_mm:.2f}"
            f" + 箱高公差 {baffle.box_height_tolerance_mm:.2f})，"
            f"目前只有 {baffle.clearance_mm:.2f} mm"
        )

    return CollisionCheck(
        clearance_mm=baffle.clearance_mm,
        required_mm=required,
        margin_mm=margin,
        level=level,
        message=msg,
    )


# ----------------------------------------------------------------------
# 設計摘要
# ----------------------------------------------------------------------
def design_summary(baffle: BaffleParams, gap: float, plate_half_y: float,
                   belt_speed: float) -> str:
    """人可讀的擋板設計與氣動效果摘要 (供「預覽」使用)。"""
    if not baffle.active:
        return "擋板未啟用 —— 間隙為完全開放的 Couette 剪切流。"

    r = solve_sealing(baffle, gap, plate_half_y)
    placement = []
    if baffle.upstream:
        placement.append("上游")
    if baffle.downstream:
        placement.append("下游")
    if baffle.y_end_plates:
        placement.append("Y 端板")

    lines = [
        "=== 擋板設計 ===",
        f"  配置          : {' + '.join(placement) or '無'}"
        f"  ({baffle.n_sides} 個 X 側)",
        f"  下緣間隙      : {baffle.clearance_mm:.2f} mm  (間隙 {gap*1000:.2f} mm 的"
        f" {baffle.clearance_mm/(gap*1000)*100:.0f}%)",
        f"  厚度 / 斜角   : {baffle.thickness_mm:.2f} mm / {baffle.angle_deg:.0f}°",
        f"  寬度          : {baffle.width_in:.2f} 吋 ({baffle.width*1000:.1f} mm)",
        f"  X 內腔長度    : {baffle.x_extent_mm:.1f} mm"
        f"  (總佔用 {baffle.total_x_footprint*1000:.1f} mm)",
        f"  串列片數/間距 : {baffle.n_stages} 片 / {baffle.stage_pitch_mm:.2f} mm",
        f"  前緣圓角      : {baffle.edge_radius_mm:.2f} mm",
        f"  洩壓孔面積比  : {baffle.vent_ratio*100:.0f}%",
        "--- 氣動效果 ---",
        f"  密封有效度 σ  : {float(r.sigma):.3f}"
        f"  ({'零淨流量循環流' if float(r.sigma) > 0.9 else '部分循環' if float(r.sigma) > 0.3 else '仍以貫流為主'})",
        f"  等效通流高度  : {r.delta_eff*1000:.3f} mm  (原 {gap*1000:.2f} mm)",
        f"  淨流量        : 無擋板的 {r.net_flux_fraction*100:.1f}%",
        f"  邊緣脈衝衰減  : ×{float(r.pulse_factor):.3f}",
        f"  橫向密封 σ_y  : {float(r.sigma_y):.3f}",
        f"  橫向逸流半長  : {r.lateral_half_length*1000:.1f} mm"
        f"  (無擋板 {plate_half_y*1000:.1f} mm)",
        f"  前緣亂流強度  : {r.turbulence_intensity*100:.1f}% of U"
        f"  = {r.turbulence_intensity*belt_speed:.3f} m/s"
        f"{'  ← 圓角不足，分離渦流會落在噴印區' if float(r.turbulence_intensity) > 0.04 else ''}",
    ]
    return "\n".join(lines)
