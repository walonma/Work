"""皮帶運動學：啟停控制、質量耦合、連續箱列的推進。

箱重對空氣動力學沒有直接影響 —— 風場由速度與幾何決定。質量透過
兩個真實的機械路徑進入模型：

1. 驅動力有限 -> 箱越重, 整線加速越慢 (a_belt = F_drive / (m_belt + m_box))
2. 摩擦力有限 -> 箱子最大加速度 = mu*g, 超過即打滑, 箱速落後皮帶
   (注意純庫倫摩擦下 mu*m*g/m = mu*g 與質量無關, 質量是透過路徑 1
    才產生差異 —— 輕箱的皮帶衝得快, 箱子跟不上而打滑;
    重箱的皮帶本身就慢, 反而不打滑但整體更慢達速)

此外箱子在皮帶上的跳動會調變間隙 h(t), 振幅 ∝ 1/m (重箱阻尼大)。
由於 Couette 分量 ∝ 1/h、壓差分量 ∝ 1/h^2, 間隙波動會直接調變風場。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import GRAVITY, SimParams


@dataclass
class SpeedCommand:
    """一段速度指令：在 time 時刻把目標速度設為 speed_m_min。"""

    time: float
    speed_m_min: float


class BeltKinematics:
    """皮帶與箱列的運動學。

    兩種模式:
      - 穩態 (預設): 皮帶與箱子都以 params.belt_speed 等速前進。
      - 啟停程序: 給定 SpeedCommand 序列, 預先積分出時間軸後內插查詢。
    """

    def __init__(
        self,
        params: SimParams,
        commands: list[SpeedCommand] | None = None,
        t_end: float = 5.0,
        dt: float = 1e-4,
    ) -> None:
        self.params = params
        self.commands = commands
        self._steady = commands is None
        if not self._steady:
            self._integrate(t_end, dt)

    # ------------------------------------------------------------------
    # 質量耦合的加速度上限
    # ------------------------------------------------------------------
    @property
    def belt_accel_limit(self) -> float:
        """皮帶的最大加速度 (m/s^2) —— 受驅動力與總慣量限制。

        箱越重, 整線加速越慢。這是質量影響風場的主要路徑。
        """
        p = self.params
        return p.drive_force_max_n / (p.belt_inertia_kg + p.box_mass_kg)

    @property
    def box_accel_limit(self) -> float:
        """箱子的最大加速度 (m/s^2) —— 受靜摩擦限制, 超過即打滑。"""
        return self.params.friction_mu * GRAVITY

    # ------------------------------------------------------------------
    # 啟停程序的積分
    # ------------------------------------------------------------------
    def _target_speed(self, t: float) -> float:
        """在時刻 t 的指令目標速度 (m/s)。"""
        target = self.params.belt_speed_m_min
        for cmd in self.commands or []:
            if t >= cmd.time:
                target = cmd.speed_m_min
        return target / 60.0

    def _integrate(self, t_end: float, dt: float) -> None:
        n = int(math.ceil(t_end / dt)) + 1
        t = np.arange(n) * dt
        v_belt = np.zeros(n)
        v_box = np.zeros(n)
        d_box = np.zeros(n)

        a_belt_max = self.belt_accel_limit
        a_box_max = self.box_accel_limit

        # 起始狀態: 若第一個指令在 t=0 之後, 視為從該速度穩態啟動
        v_belt[0] = self._target_speed(0.0)
        v_box[0] = v_belt[0]

        for i in range(1, n):
            target = self._target_speed(t[i - 1])

            # 皮帶追向目標速度, 受驅動力/慣量限制
            dv = target - v_belt[i - 1]
            a_belt = np.clip(dv / dt, -a_belt_max, a_belt_max)
            v_belt[i] = v_belt[i - 1] + a_belt * dt

            # 箱子靠摩擦追向皮帶速度, 受 mu*g 限制 (超過即打滑)
            dv_box = v_belt[i] - v_box[i - 1]
            a_box = np.clip(dv_box / dt, -a_box_max, a_box_max)
            v_box[i] = v_box[i - 1] + a_box * dt

            d_box[i] = d_box[i - 1] + 0.5 * (v_box[i] + v_box[i - 1]) * dt

        self._t = t
        self._v_belt = v_belt
        self._v_box = v_box
        self._d_box = d_box

    # ------------------------------------------------------------------
    # 查詢
    # ------------------------------------------------------------------
    def belt_speed_at(self, t):
        """皮帶速度 (m/s)。"""
        if self._steady:
            return np.full_like(np.asarray(t, dtype=float), self.params.belt_speed)
        return np.interp(t, self._t, self._v_belt)

    def box_speed_at(self, t):
        """箱子實際速度 (m/s)。啟停過渡期會落後皮帶。"""
        if self._steady:
            return np.full_like(np.asarray(t, dtype=float), self.params.belt_speed)
        return np.interp(t, self._t, self._v_box)

    def distance_at(self, t):
        """箱列累積行進距離 (m)。"""
        if self._steady:
            return np.asarray(t, dtype=float) * self.params.belt_speed
        return np.interp(t, self._t, self._d_box)

    def slip_at(self, t):
        """皮帶與箱子的速度差 (m/s)。正值代表箱子打滑落後。"""
        return self.belt_speed_at(t) - self.box_speed_at(t)

    # ------------------------------------------------------------------
    # 箱列的相位幾何
    # ------------------------------------------------------------------
    def phase_at(self, t):
        """箱列相位 u (m), 定義域 [0, pitch)。

        u < box_length : 有箱子在噴頭下方, 箱前緣已通過噴頭 u 公尺
        u >= box_length: 箱間開放區
        """
        p = self.params
        return np.mod(self.distance_at(t), p.train_pitch)

    def leading_edge_offsets(self, t):
        """回傳最近兩個「箱前緣」相對噴頭的有號距離 (m)。

        正值 = 該前緣已通過噴頭; 負值 = 尚未抵達 (接近中)。
        同時回傳兩個, 讓前後箱的前緣脈衝能自然疊加 —— 這正是
        30cm 間距時前一箱尾流與下一箱前緣互相干涉的來源。
        """
        u = self.phase_at(t)
        current = u  # 目前箱子的前緣 (已通過 u 公尺)
        upcoming = u - self.params.train_pitch  # 下一個箱子的前緣 (負值)
        return current, upcoming

    def trailing_edge_offset(self, t):
        """箱後緣相對噴頭的有號距離 (m)。正值 = 已通過。"""
        return self.phase_at(t) - self.params.box_length_m

    def under_box(self, t):
        """布林值: 噴頭正下方是否有箱子 (硬判定, 供診斷用)。"""
        u = self.phase_at(t)
        return u < self.params.box_length_m

    def confinement(self, t):
        """拘束因子 c(t) ∈ [0, 1] —— 噴嘴板被箱頂面覆蓋的比例。

        箱前緣掃過噴嘴板時是漸進的 (板寬 plate_x), 因此用平滑過渡
        而非硬切換。c=1 表示完全拘束 (Couette 成立);
        c=0 表示箱間開放區, 噴頭下方是 50cm 空曠區, Couette 流消失。
        """
        p = self.params
        w = max(p.plate_x / 2.0, 1e-6)
        u = self.phase_at(t)
        entering = _smoothstep(u / w)  # 前緣進入
        leaving = _smoothstep((p.box_length_m - u) / w)  # 後緣離開
        return entering * leaving

    def gap_at(self, t):
        """瞬時間隙 h(t) (m), 含箱體跳動調變。

        跳動振幅 ∝ 1/m (重箱阻尼大), 頻率 f = sqrt(k/m)/2pi。
        只在有箱子且皮帶在動時才有跳動。
        """
        p = self.params
        t_arr = np.asarray(t, dtype=float)

        amp = (
            p.bounce_amp_ref_mm
            / 1000.0
            * (p.bounce_mass_ref_kg / p.box_mass_kg)
        )
        # 跳動由行進激發, 速度為零時不跳
        speed_ratio = np.clip(self.box_speed_at(t_arr) / max(p.belt_speed, 1e-9), 0.0, 1.0)
        if p.belt_speed <= 0:
            speed_ratio = np.zeros_like(t_arr)

        f_b = math.sqrt(p.bounce_stiffness_n_m / p.box_mass_kg) / (2.0 * math.pi)
        modulation = amp * speed_ratio * self.confinement(t_arr) * np.sin(
            2.0 * math.pi * f_b * t_arr
        )
        # 間隙不得塌到零或超出量測範圍
        return np.clip(p.gap + modulation, 0.3 * p.gap, 2.0 * p.gap)

    @property
    def bounce_frequency_hz(self) -> float:
        """箱體跳動頻率 (Hz)。5kg -> 55.9Hz; 50kg -> 17.7Hz。"""
        p = self.params
        return math.sqrt(p.bounce_stiffness_n_m / p.box_mass_kg) / (2.0 * math.pi)

    @property
    def bounce_amplitude_mm(self) -> float:
        """箱體跳動振幅 (mm)。"""
        p = self.params
        return p.bounce_amp_ref_mm * (p.bounce_mass_ref_kg / p.box_mass_kg)


def _smoothstep(x):
    """平滑階梯函數: x<=0 -> 0, x>=1 -> 1, 中間為 3x^2-2x^3。"""
    xc = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return xc * xc * (3.0 - 2.0 * xc)
