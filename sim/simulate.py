"""模擬編排 —— 噴發、飄移誤差基準、循環掃描與參數掃描。

飄移誤差的基準很關鍵。飛行 0.55ms 期間箱子已移動約 3.2mm
(350 m/min), 遠大於風場造成的 ~0.15mm 飄移。這 3.2mm 實務上由噴發
時序補償掉, 因此不能拿它當誤差。正確定義是::

    飄移誤差 Δ = 有風落點(箱座標) − 無風標稱落點(箱座標)

兩者都換算到箱子座標系 (扣掉飛行期間箱子的位移), 3.2mm 的共同項
自然相消, 剩下的才是真正的噴印誤差。而 Δ 隨產線速度變化, 固定的
時序補償無法在全速域內修正 —— 這正是實務上的痛點。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .belt import BeltKinematics
from .config import SimParams
from .droplet import DropletSwarm, FlightResult
from .windfield import WindField


@dataclass
class BurstResult:
    """單次噴發 (300 顆墨滴同時噴出) 的完整結果。"""

    params: SimParams
    t_fire: float
    gap_at_fire: float
    nozzle_y: np.ndarray  # (N,) 噴嘴 Y 座標 (m)
    drift_x: np.ndarray  # (N,) 縱向飄移誤差 (m)
    drift_y: np.ndarray  # (N,) 橫向飄移誤差 (m)
    flight_time: np.ndarray  # (N,) 有風時的飛行時間 (s)
    nominal_flight_time: np.ndarray  # (N,) 無風標稱飛行時間 (s)
    wind_flight: FlightResult
    nominal_flight: FlightResult
    confinement: float  # 噴發當下的拘束因子 (循環階段指標)
    box_speed: float  # 噴發當下的箱速 (m/s)

    # ------------------------------------------------------------------
    @property
    def drift_magnitude(self) -> np.ndarray:
        """合成飄移量 (m)。"""
        return np.hypot(self.drift_x, self.drift_y)

    @property
    def drift_x_um(self) -> np.ndarray:
        return self.drift_x * 1e6

    @property
    def drift_y_um(self) -> np.ndarray:
        return self.drift_y * 1e6

    @property
    def drift_magnitude_um(self) -> np.ndarray:
        return self.drift_magnitude * 1e6

    @property
    def drift_in_dot_pitches(self) -> np.ndarray:
        """以噴嘴點距為單位的飄移量 —— 直接反映噴印品質劣化程度。"""
        return self.drift_magnitude / self.params.nozzle_pitch

    @property
    def max_drift_um(self) -> float:
        return float(np.max(self.drift_magnitude_um))

    @property
    def mean_drift_um(self) -> float:
        return float(np.mean(self.drift_magnitude_um))

    def summary(self) -> str:
        p = self.params
        return "\n".join(
            [
                f"噴發時刻 t = {self.t_fire*1e3:.2f} ms"
                f" (拘束因子 c = {self.confinement:.2f}, 箱速 {self.box_speed:.2f} m/s)",
                f"  噴發間隙      : {self.gap_at_fire*1000:.3f} mm",
                f"  飛行時間      : {np.mean(self.flight_time)*1e3:.3f} ms",
                f"  縱向飄移 Δx   : {np.mean(self.drift_x_um):+.1f} um"
                f" (範圍 {np.min(self.drift_x_um):+.1f} ~ {np.max(self.drift_x_um):+.1f})",
                f"  橫向飄移 Δy   : {np.mean(self.drift_y_um):+.1f} um"
                f" (範圍 {np.min(self.drift_y_um):+.1f} ~ {np.max(self.drift_y_um):+.1f})",
                f"  合成飄移      : 平均 {self.mean_drift_um:.1f} um,"
                f" 最大 {self.max_drift_um:.1f} um",
                f"  換算點距      : 最大 {np.max(self.drift_in_dot_pitches):.2f} 個點距"
                f" (點距 {p.nozzle_pitch*1e6:.1f} um @ {p.dpi:.0f} dpi)",
            ]
        )


# ----------------------------------------------------------------------
# 噴發
# ----------------------------------------------------------------------
def fire_burst(
    params: SimParams,
    t_fire: float = 0.0,
    belt: BeltKinematics | None = None,
    wind: WindField | None = None,
    record_frames: int = 0,
    _nominal_cache: dict | None = None,
) -> BurstResult:
    """在 t_fire 時刻讓 300 個噴嘴同時噴發, 計算各自的飄移誤差。"""
    belt = belt if belt is not None else BeltKinematics(params)
    wind = wind if wind is not None else WindField(params, belt)

    gap = float(np.asarray(belt.gap_at(t_fire)).reshape(-1)[0])
    conf = float(np.asarray(belt.confinement(t_fire)).reshape(-1)[0])
    u_box = float(np.asarray(belt.box_speed_at(t_fire)).reshape(-1)[0])

    swarm = DropletSwarm(params)

    # 有風
    wind_flight = swarm.fly(wind, t0=t_fire, launch_gap=gap, record_frames=record_frames)

    # 無風標稱基準 (只受重力與阻力), 相同的噴發間隙
    key = round(gap, 12)
    nominal = None
    if _nominal_cache is not None:
        nominal = _nominal_cache.get(key)
    if nominal is None:
        nominal = swarm.fly(None, t0=t_fire, launch_gap=gap, record_frames=record_frames)
        if _nominal_cache is not None:
            _nominal_cache[key] = nominal

    # 換算到箱子座標系: 扣掉飛行期間箱子的位移
    d0 = float(np.asarray(belt.distance_at(t_fire)).reshape(-1)[0])
    box_disp_wind = np.asarray(belt.distance_at(t_fire + wind_flight.flight_time)) - d0
    box_disp_nom = np.asarray(belt.distance_at(t_fire + nominal.flight_time)) - d0

    x_box_wind = wind_flight.landing_x - box_disp_wind
    x_box_nom = nominal.landing_x - box_disp_nom

    drift_x = x_box_wind - x_box_nom
    drift_y = wind_flight.landing_y - nominal.landing_y

    return BurstResult(
        params=params,
        t_fire=t_fire,
        gap_at_fire=gap,
        nozzle_y=params.nozzle_y(),
        drift_x=drift_x,
        drift_y=drift_y,
        flight_time=wind_flight.flight_time,
        nominal_flight_time=nominal.flight_time,
        wind_flight=wind_flight,
        nominal_flight=nominal,
        confinement=conf,
        box_speed=u_box,
    )


# ----------------------------------------------------------------------
# 循環相位工具
# ----------------------------------------------------------------------
def time_at_phase(params: SimParams, phase_m: float) -> float:
    """把「箱列相位 (公尺)」換算成時間 (s), 適用等速情形。

    phase_m = 0        : 箱前緣正好抵達噴頭
    phase_m = box_len/2: 箱子中段在噴頭下 (穩態 Couette)
    phase_m = box_len  : 箱後緣正好通過噴頭
    """
    if params.belt_speed <= 0:
        return 0.0
    return phase_m / params.belt_speed


def steady_fire_time(params: SimParams) -> float:
    """取箱子中段的時刻 —— 穩態 Couette 階段, 適合當基準案例。"""
    return time_at_phase(params, params.box_length_m * 0.5)


@dataclass
class CycleScan:
    """一個完整箱列循環的掃描結果 (5-2d)。"""

    params: SimParams
    times: np.ndarray
    phase: np.ndarray  # (M,) 相位 (m)
    confinement: np.ndarray
    mean_drift_um: np.ndarray
    max_drift_um: np.ndarray
    mean_drift_x_um: np.ndarray
    mean_drift_y_um: np.ndarray
    edge_drift_y_um: np.ndarray  # 陣列最外側噴嘴的橫向飄移
    gap_mean_speed: np.ndarray  # 間隙平均風速 (m/s)


def sweep_over_cycle(
    params: SimParams,
    n_samples: int = 60,
    belt: BeltKinematics | None = None,
) -> CycleScan:
    """掃過一個完整的箱列循環 (箱下 + 箱間開放區), 觀察飄移如何隨階段變化。"""
    belt = belt if belt is not None else BeltKinematics(params)
    wind = WindField(params, belt)

    period = params.cycle_period
    if not np.isfinite(period):
        period = 1.0
    times = np.linspace(0.0, period, n_samples, endpoint=False)

    cache: dict = {}
    rows = []
    for t in times:
        rows.append(fire_burst(params, t_fire=float(t), belt=belt, wind=wind, _nominal_cache=cache))

    return CycleScan(
        params=params,
        times=times,
        phase=np.asarray(belt.phase_at(times), dtype=float),
        confinement=np.array([r.confinement for r in rows]),
        mean_drift_um=np.array([r.mean_drift_um for r in rows]),
        max_drift_um=np.array([r.max_drift_um for r in rows]),
        mean_drift_x_um=np.array([float(np.mean(r.drift_x_um)) for r in rows]),
        mean_drift_y_um=np.array([float(np.mean(r.drift_y_um)) for r in rows]),
        edge_drift_y_um=np.array([float(r.drift_y_um[-1]) for r in rows]),
        gap_mean_speed=np.array([wind.gap_mean_speed(float(t)) for t in times]),
    )


# ----------------------------------------------------------------------
# 參數掃描
# ----------------------------------------------------------------------
@dataclass
class ParameterSweep:
    field: str
    label: str
    values: np.ndarray
    mean_drift_um: np.ndarray
    max_drift_um: np.ndarray
    mean_drift_x_um: np.ndarray
    edge_drift_y_um: np.ndarray
    gap_mean_speed: np.ndarray


_SWEEP_LABELS = {
    "belt_speed_m_min": "產線速度 (m/min)",
    "droplet_pl": "墨滴體積 (pL)",
    "gap_mm": "噴嘴間隙 (mm)",
    "box_spacing_cm": "箱間距 (cm)",
    "box_mass_kg": "箱重 (kg)",
}


def sweep_parameter(
    base: SimParams,
    field: str,
    values,
    phase_fraction: float = 0.5,
) -> ParameterSweep:
    """掃描單一參數, 在箱子的固定相位上噴發並記錄飄移。

    phase_fraction: 噴發時箱子已通過噴頭的比例 (0=前緣, 0.5=中段, 1=後緣)
    """
    values = np.asarray(values, dtype=float)
    mean_d, max_d, mean_dx, edge_dy, speeds = [], [], [], [], []

    for v in values:
        p = base.replace(**{field: float(v)})
        belt = BeltKinematics(p)
        wind = WindField(p, belt)
        t = time_at_phase(p, p.box_length_m * phase_fraction)
        r = fire_burst(p, t_fire=t, belt=belt, wind=wind)
        mean_d.append(r.mean_drift_um)
        max_d.append(r.max_drift_um)
        mean_dx.append(float(np.mean(r.drift_x_um)))
        edge_dy.append(float(r.drift_y_um[-1]))
        speeds.append(wind.gap_mean_speed(t))

    return ParameterSweep(
        field=field,
        label=_SWEEP_LABELS.get(field, field),
        values=values,
        mean_drift_um=np.array(mean_d),
        max_drift_um=np.array(max_d),
        mean_drift_x_um=np.array(mean_dx),
        edge_drift_y_um=np.array(edge_dy),
        gap_mean_speed=np.array(speeds),
    )


# ----------------------------------------------------------------------
# 次要診斷: X 方向解析度
# ----------------------------------------------------------------------
def x_resolution_report(params: SimParams) -> str:
    """X 向解析度診斷。

    使用者回報的症狀是「字歪斜 / 邊緣模糊」(風場飄移), 而非
    「字被拉長 / 點變稀疏」(噴發頻率不足), 所以這只是次要數值顯示。
    """
    p = params
    return "\n".join(
        [
            "--- X 方向解析度 (次要診斷) ---",
            f"  Y 向 (噴嘴陣列): {p.dpi:.0f} dpi, 點距 {p.nozzle_pitch*1e6:.1f} um",
            f"  X 向 (噴發頻率): {p.x_dpi:.0f} dpi, 點距 {p.x_dot_pitch*1e6:.0f} um"
            f" @ {p.fire_frequency_hz/1000:.0f} kHz",
            f"  若要 X 向也達到 {p.dpi:.0f} dpi, 需要 {p.required_fire_frequency_hz/1000:.1f} kHz"
            f" (TIJ 上限約 12~15 kHz)",
            f"  X:Y 解析度比 = 1 : {p.dpi / max(p.x_dpi, 1e-9):.1f}",
        ]
    )
