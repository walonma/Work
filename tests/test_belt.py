"""皮帶運動學、質量耦合與箱列相位。"""

import numpy as np
import pytest

from sim.belt import BeltKinematics, SpeedCommand
from sim.config import GRAVITY, SimParams


def test_steady_mode_tracks_commanded_speed():
    p = SimParams(belt_speed_m_min=350.0)
    b = BeltKinematics(p)
    t = np.linspace(0, 1, 11)
    assert np.allclose(b.box_speed_at(t), p.belt_speed)
    assert np.allclose(b.belt_speed_at(t), p.belt_speed)
    assert np.allclose(b.slip_at(t), 0.0)


def test_steady_distance_is_linear():
    p = SimParams(belt_speed_m_min=350.0)
    b = BeltKinematics(p)
    assert float(np.asarray(b.distance_at(2.0))) == pytest.approx(2.0 * p.belt_speed)


# ----------------------------------------------------------------------
# 質量耦合
# ----------------------------------------------------------------------
def test_heavier_box_slows_line_acceleration():
    """路徑 1: 驅動力有限 -> 箱越重, 整線加速越慢。"""
    light = BeltKinematics(SimParams(box_mass_kg=5.0))
    heavy = BeltKinematics(SimParams(box_mass_kg=50.0))
    assert light.belt_accel_limit > heavy.belt_accel_limit


def test_box_accel_limit_is_friction_bound():
    """路徑 2: 箱子最大加速度 = mu*g (與質量無關 —— 庫倫摩擦下質量相消)。"""
    p = SimParams(friction_mu=0.6)
    b = BeltKinematics(p)
    assert b.box_accel_limit == pytest.approx(0.6 * GRAVITY)
    # 質量改變不影響此上限, 質量是透過驅動力路徑才產生差異
    assert BeltKinematics(p.replace(box_mass_kg=50.0)).box_accel_limit == pytest.approx(
        b.box_accel_limit
    )


def test_light_box_slips_heavy_box_does_not():
    """輕箱: 皮帶衝得快 -> 箱子跟不上而打滑。重箱: 皮帶本身就慢 -> 不打滑。"""
    cmds = [SpeedCommand(0.0, 0.0), SpeedCommand(0.1, 350.0)]
    t = np.linspace(0.0, 2.0, 400)

    light = BeltKinematics(SimParams(box_mass_kg=5.0), commands=cmds, t_end=3.0)
    heavy = BeltKinematics(SimParams(box_mass_kg=50.0), commands=cmds, t_end=3.0)

    assert np.max(light.slip_at(t)) > 0.1  # 明顯打滑
    assert np.max(heavy.slip_at(t)) < np.max(light.slip_at(t))


def test_heavier_box_takes_longer_to_reach_speed():
    """整體而言重箱達速較慢 —— 啟停過渡期正是噴印最不穩的時候。"""
    cmds = [SpeedCommand(0.0, 0.0), SpeedCommand(0.1, 350.0)]
    target = SimParams(belt_speed_m_min=350.0).belt_speed

    times = []
    for m in (5.0, 50.0):
        b = BeltKinematics(SimParams(box_mass_kg=m), commands=cmds, t_end=4.0)
        t = np.linspace(0.0, 4.0, 4000)
        v = np.asarray(b.box_speed_at(t))
        reached = np.nonzero(v >= 0.99 * target)[0]
        times.append(t[reached[0]] if reached.size else np.inf)

    assert times[1] > times[0]


def test_bounce_amplitude_inversely_proportional_to_mass():
    """路徑 3: 箱體跳動振幅 ∝ 1/m (重箱阻尼大)。"""
    light = BeltKinematics(SimParams(box_mass_kg=5.0))
    heavy = BeltKinematics(SimParams(box_mass_kg=50.0))
    assert light.bounce_amplitude_mm == pytest.approx(10.0 * heavy.bounce_amplitude_mm)


def test_bounce_frequency_decreases_with_mass():
    """f = sqrt(k/m)/2pi -> 5kg 約 55.9Hz, 50kg 約 17.7Hz。"""
    light = BeltKinematics(SimParams(box_mass_kg=5.0))
    heavy = BeltKinematics(SimParams(box_mass_kg=50.0))
    assert light.bounce_frequency_hz > heavy.bounce_frequency_hz
    assert light.bounce_frequency_hz == pytest.approx(55.9, rel=0.02)
    assert heavy.bounce_frequency_hz == pytest.approx(17.7, rel=0.02)


def test_gap_modulation_stays_in_physical_bounds():
    p = SimParams(box_mass_kg=5.0)
    b = BeltKinematics(p)
    t = np.linspace(0.0, 0.2, 2000)
    gap = np.asarray(b.gap_at(t))
    assert np.all(gap > 0.0)
    assert np.all(gap >= 0.3 * p.gap - 1e-15)
    assert np.all(gap <= 2.0 * p.gap + 1e-15)


def test_no_bounce_when_line_stopped():
    """產線停止 -> 沒有行進激發 -> 間隙不跳動。"""
    p = SimParams(belt_speed_m_min=0.0)
    b = BeltKinematics(p)
    t = np.linspace(0.0, 0.5, 200)
    assert np.allclose(np.asarray(b.gap_at(t)), p.gap)


# ----------------------------------------------------------------------
# 箱列相位
# ----------------------------------------------------------------------
def test_phase_wraps_at_train_pitch():
    p = SimParams(belt_speed_m_min=350.0, box_spacing_cm=40.0)
    b = BeltKinematics(p)
    t = np.linspace(0.0, 3 * p.cycle_period, 500)
    phase = np.asarray(b.phase_at(t))
    assert np.all(phase >= 0.0)
    assert np.all(phase < p.train_pitch + 1e-12)


def test_under_box_duty_cycle_matches_geometry():
    """箱下時間佔比 = 箱長 / (箱長+間距)。"""
    p = SimParams(belt_speed_m_min=350.0, box_spacing_cm=50.0)
    b = BeltKinematics(p)
    t = np.linspace(0.0, 10 * p.cycle_period, 20001)
    duty = float(np.mean(np.asarray(b.under_box(t))))
    assert duty == pytest.approx(p.box_length_m / p.train_pitch, rel=0.02)


def test_confinement_transitions_smoothly():
    """前緣掃過噴嘴板是漸進的, 拘束因子應平滑而非硬切換。"""
    p = SimParams(belt_speed_m_min=350.0)
    b = BeltKinematics(p)
    t = np.linspace(0.0, p.cycle_period, 2000)
    c = np.asarray(b.confinement(t))
    assert c.min() < 0.02 and c.max() > 0.98
    # 沒有單步的巨大跳變
    assert np.max(np.abs(np.diff(c))) < 0.2


def test_confinement_one_mid_box_zero_mid_gap():
    p = SimParams(belt_speed_m_min=350.0, box_spacing_cm=50.0)
    b = BeltKinematics(p)
    t_mid_box = (p.box_length_m / 2) / p.belt_speed
    t_mid_gap = (p.box_length_m + p.box_spacing / 2) / p.belt_speed
    assert float(np.asarray(b.confinement(t_mid_box))) == pytest.approx(1.0, abs=1e-6)
    assert float(np.asarray(b.confinement(t_mid_gap))) == pytest.approx(0.0, abs=1e-6)


def test_leading_edge_offsets_are_one_pitch_apart():
    """同時追蹤目前箱與下一箱的前緣 —— 兩者相隔一個 pitch。"""
    p = SimParams(belt_speed_m_min=350.0)
    b = BeltKinematics(p)
    t = np.linspace(0.0, p.cycle_period, 50)
    cur, nxt = b.leading_edge_offsets(t)
    assert np.allclose(np.asarray(cur) - np.asarray(nxt), p.train_pitch)


def test_trailing_edge_offset_zero_at_box_end():
    p = SimParams(belt_speed_m_min=350.0)
    b = BeltKinematics(p)
    t = p.box_length_m / p.belt_speed
    assert float(np.asarray(b.trailing_edge_offset(t))) == pytest.approx(0.0, abs=1e-9)
