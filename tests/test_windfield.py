"""風場邊界條件與物理極限。"""

import numpy as np
import pytest

from sim.belt import BeltKinematics
from sim.config import SimParams
from sim.simulate import steady_fire_time
from sim.windfield import WindField


def _steady_setup(**kwargs):
    """取一個箱體完全在噴頭下 (拘束因子 = 1) 的穩態狀態。"""
    p = SimParams(**kwargs)
    belt = BeltKinematics(p)
    wind = WindField(p, belt)
    t = steady_fire_time(p)
    gap = float(np.asarray(belt.gap_at(t)).reshape(-1)[0])
    return p, belt, wind, t, gap


def test_confinement_is_one_mid_box():
    _, belt, _, t, _ = _steady_setup()
    assert float(np.asarray(belt.confinement(t))) == pytest.approx(1.0, abs=1e-6)


def test_no_slip_at_box_surface():
    """z=0 貼著箱頂面 -> ux = U_box (箱面帶著空氣走)。"""
    p, _, wind, t, _ = _steady_setup()
    ux, uy, uz = wind.velocity(0.0, 0.0, 0.0, t)
    assert float(ux) == pytest.approx(p.belt_speed, rel=1e-9)
    assert float(uy) == pytest.approx(0.0, abs=1e-12)
    assert float(uz) == pytest.approx(0.0, abs=1e-12)


def test_no_slip_at_nozzle_plate():
    """z=h 貼著靜止的噴嘴板 -> 全部為零。"""
    _, _, wind, t, gap = _steady_setup()
    ux, uy, uz = wind.velocity(0.0, 0.0, gap, t)
    assert float(ux) == pytest.approx(0.0, abs=1e-12)
    assert float(uy) == pytest.approx(0.0, abs=1e-12)
    assert float(uz) == pytest.approx(0.0, abs=1e-12)


def test_no_slip_at_nozzle_plate_holds_in_every_cycle_phase():
    """噴嘴板是固定實體壁面, 不論下方有無箱子都存在。

    因此 z=h 的 no-slip 與不可穿透必須在循環的每個相位都成立 ——
    包含箱間開放區 (此時 z=0 下方是 50cm 空曠, 但板面仍是壁面)。
    """
    p = SimParams(belt_speed_m_min=350.0, box_spacing_cm=50.0)
    belt = BeltKinematics(p)
    wind = WindField(p, belt)

    t = np.linspace(0.0, p.cycle_period, 200)
    gap = np.asarray(belt.gap_at(t))
    conf = np.asarray(belt.confinement(t))
    assert conf.min() < 0.05 and conf.max() > 0.95  # 確實涵蓋兩種階段

    for y in (0.0, p.plate_half_y * 0.5, p.plate_half_y):
        ux, uy, uz = wind.velocity(0.0, y, gap, t)
        assert np.allclose(ux, 0.0, atol=1e-12), "板面縱向 no-slip 失效"
        assert np.allclose(uy, 0.0, atol=1e-12), "板面橫向 no-slip 失效"
        assert np.allclose(uz, 0.0, atol=1e-12), "板面不可穿透失效"


def test_no_penetration_at_box_surface_when_confined():
    """箱體在下方時 z=0 也是實體壁面 -> 垂直分量為零。"""
    p, belt, wind, t, _ = _steady_setup()
    _, _, uz = wind.velocity(np.linspace(-0.03, 0.03, 25), 0.0, 0.0, t)
    assert np.allclose(uz, 0.0, atol=1e-12)


def test_couette_profile_is_linear():
    """拘束段的 ux 應為線性剪切 U(1 - z/h)。"""
    p, _, wind, t, gap = _steady_setup()
    z = np.linspace(0.0, gap, 21)
    ux, _, _ = wind.velocity(0.0, 0.0, z, t)
    expected = p.belt_speed * (1.0 - z / gap)
    assert np.allclose(ux, expected, atol=1e-9)


def test_gap_mean_speed_is_half_belt_speed():
    """線性剪切的厚度平均 = U/2。350 m/min -> 2.92 m/s。"""
    p, _, wind, t, _ = _steady_setup()
    assert wind.gap_mean_speed(t) == pytest.approx(p.belt_speed / 2.0, rel=1e-3)


def test_zero_speed_gives_zero_field():
    """產線停止 -> 全場為零 (所有分量都正比於 U)。"""
    p = SimParams(belt_speed_m_min=0.0)
    wind = WindField(p)
    z = np.linspace(0.0, p.gap, 11)
    for t in (0.0, 0.01, 0.1):
        ux, uy, uz = wind.velocity(0.0, 0.0, z, t)
        assert np.allclose(ux, 0.0, atol=1e-15)
        assert np.allclose(uy, 0.0, atol=1e-15)
        assert np.allclose(uz, 0.0, atol=1e-15)


def test_lateral_zero_at_array_centre():
    """陣列中央 (y=0) 的橫向逸流為零 —— 逸出路徑最長。"""
    _, _, wind, _, gap = _steady_setup()
    for t in (0.0, 0.005, 0.02):
        _, uy, _ = wind.velocity(0.0, 0.0, gap / 2, t)
        assert float(uy) == pytest.approx(0.0, abs=1e-15)


def test_lateral_antisymmetric_about_centre():
    """橫向逸流對中線反對稱 -> 300 顆墨滴呈對稱「外八」。"""
    p, _, wind, _, gap = _steady_setup()
    y = np.linspace(0.0, p.plate_half_y, 15)
    t = 0.0  # 箱前緣抵達, 擠壓膜逸流最強
    _, uy_pos, _ = wind.velocity(0.0, y, gap / 2, t)
    _, uy_neg, _ = wind.velocity(0.0, -y, gap / 2, t)
    assert np.allclose(uy_pos, -uy_neg, atol=1e-15)


def test_lateral_grows_outward():
    """越靠陣列兩端, 橫向逸流越強 (緊鄰逸出口)。"""
    p, _, wind, _, gap = _steady_setup()
    y = np.linspace(0.0, p.plate_half_y, 12)
    _, uy, _ = wind.velocity(0.0, y, gap / 2, 0.0)
    assert np.all(np.diff(np.abs(uy)) > -1e-15)
    assert abs(float(uy[-1])) > abs(float(uy[1]))


def test_leading_edge_pulse_peaks_near_edge():
    """前緣壓差脈衝應在箱前緣掃過噴頭時達到峰值。"""
    p = SimParams(belt_speed_m_min=350.0)
    belt = BeltKinematics(p)
    wind = WindField(p, belt)
    period = p.cycle_period
    t = np.linspace(0.0, period, 400)
    gap = np.asarray(belt.gap_at(t))
    ux, _, _ = wind.velocity(0.0, 0.0, gap / 2, t)
    i = int(np.argmax(np.abs(ux)))
    phase = float(np.asarray(belt.phase_at(t[i])))
    # 峰值應落在箱前緣附近 (相位接近 0 或接近一個週期)
    assert min(phase, p.train_pitch - phase) < 4 * p.sigma_edge


def test_open_region_loses_couette():
    """箱間開放區沒有拘束 -> Couette 消失, 風速顯著低於箱下穩態。"""
    p = SimParams(belt_speed_m_min=350.0, box_spacing_cm=50.0)
    belt = BeltKinematics(p)
    wind = WindField(p, belt)
    t_open = (p.box_length_m + p.box_spacing / 2) / p.belt_speed
    t_under = steady_fire_time(p)
    assert float(np.asarray(belt.confinement(t_open))) < 0.05
    assert wind.gap_mean_speed(t_open) < 0.5 * wind.gap_mean_speed(t_under)


def test_tighter_spacing_leaves_more_residual_wake():
    """30cm 間距時, 下一箱前緣抵達時前一箱的尾流尚未消散。

    比較點取在「下一箱前緣抵達前 10mm」—— 兩種間距在此點與來箱的距離
    相同, 前緣脈衝貢獻一致, 差別純粹來自尾流殘留。尾流是逆向流 (負值),
    因此殘留越多, 淨 u_x 越低。
    """
    ux_at = {}
    residual = {}
    for spacing in (30.0, 50.0):
        p = SimParams(belt_speed_m_min=350.0, box_spacing_cm=spacing)
        belt = BeltKinematics(p)
        wind = WindField(p, belt)
        t = (p.train_pitch - 0.01) / p.belt_speed
        gap = float(np.asarray(belt.gap_at(t)).reshape(-1)[0])
        ux, _, _ = wind.velocity(0.0, 0.0, gap / 2, t)
        ux_at[spacing] = float(ux)
        s_trail = float(np.asarray(belt.trailing_edge_offset(t)))
        residual[spacing] = np.exp(-s_trail / p.wake_length_m)
        assert float(np.asarray(belt.confinement(t))) < 0.05  # 確實在開放區

    # 30cm 的尾流殘留因子明顯較大
    assert residual[30.0] > 3.0 * residual[50.0]
    # 逆向殘留較強 -> 淨縱向風速較低
    assert ux_at[30.0] < ux_at[50.0]


def test_velocity_at_matches_velocity():
    _, _, wind, t, gap = _steady_setup()
    pos = np.array([[0.0, 0.0, gap / 2], [0.001, 0.005, gap / 3]])
    got = wind.velocity_at(pos, t)
    for k in range(pos.shape[0]):
        ux, uy, uz = wind.velocity(pos[k, 0], pos[k, 1], pos[k, 2], t)
        assert got[k] == pytest.approx([float(ux), float(uy), float(uz)])


def test_slices_have_expected_shapes():
    p, _, wind, t, _ = _steady_setup()
    X, Z, Ux, Uz, sp = wind.slice_xz(t, nx=21, nz=11)
    assert X.shape == Z.shape == Ux.shape == Uz.shape == sp.shape == (11, 21)
    Y, Z2, Uy, Uz2, Ux2 = wind.slice_yz(t, ny=21, nz=11)
    assert Y.shape == (11, 21)
