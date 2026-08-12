"""導風擋板 —— 密封網路、回流剖面、零飄移點與機械餘裕。"""

import numpy as np
import pytest

from sim.baffle import (
    BaffleParams,
    check_collision,
    design_summary,
    solve_sealing,
)
from sim.belt import BeltKinematics
from sim.config import SimParams
from sim.simulate import (
    crosswind_response,
    find_zero_drift_clearance,
    fire_burst,
    steady_fire_time,
    turbulence_scatter,
)
from sim.windfield import WindField


def _setup(baffle: BaffleParams, **kw):
    p = SimParams(baffle=baffle, **kw)
    belt = BeltKinematics(p)
    wind = WindField(p, belt)
    t = steady_fire_time(p)
    return p, belt, wind, t


# ======================================================================
# 回歸: 未啟用時必須精確退回無擋板的解
# ======================================================================
def test_disabled_baffle_is_exact_no_op():
    """擋板未啟用時 sigma=0、pulse=1 —— 風場必須逐點完全相同。"""
    off = SimParams(baffle=BaffleParams(enabled=False))
    ref = SimParams()  # 預設也是未啟用
    for p in (off, ref):
        belt = BeltKinematics(p)
        wind = WindField(p, belt)
        t = steady_fire_time(p)
        seal = wind.sealing_at(t)
        assert float(seal.sigma) == 0.0
        assert float(seal.pulse_factor) == 1.0
        assert seal.lateral_half_length == pytest.approx(p.plate_half_y)
        assert float(seal.turbulence_intensity) == 0.0


def test_disabled_baffle_preserves_couette_profile():
    _, _, wind, t = _setup(BaffleParams(enabled=False))
    p = wind.params
    gap = float(np.asarray(wind.belt.gap_at(t)).reshape(-1)[0])
    z = np.linspace(0.0, gap, 21)
    ux, _, _ = wind.velocity(0.0, 0.0, z, t)
    assert np.allclose(ux, p.belt_speed * (1.0 - z / gap), atol=1e-9)


def test_enabled_but_all_surfaces_off_is_no_op():
    """enabled=True 但三個面都關掉 -> 仍然沒有擋板效果。"""
    b = BaffleParams(enabled=True, upstream=False, downstream=False, y_end_plates=False)
    assert not b.active
    _, _, wind, t = _setup(b)
    assert float(wind.sealing_at(t).sigma) == 0.0


# ======================================================================
# 密封網路
# ======================================================================
def test_sigma_zero_when_clearance_equals_gap():
    """下緣間隙等於間隙本身 = 沒有收縮 -> 無密封效果。

    取間隙 3mm (規格下限) 搭配下緣間隙 3mm (亦為規格上限) 來驗證這個極限。
    """
    gap = 0.003
    b = BaffleParams(enabled=True, clearance_mm=3.0, n_stages=1)
    r = solve_sealing(b, gap, 0.0127)

    # 淨流量不變: Couette 泵送量與潤滑阻力都同比例隨高度縮放, 故 σ=0
    assert float(r.sigma) == pytest.approx(0.0, abs=1e-12)

    # 但脈衝仍被衰減 —— 擋板即使不縮小間隙, 也把流道加長了,
    # 而壓差驅動流的阻力正比於路徑長度。加上斜角的導流效果。
    path = b.x_extent + b.n_sides * b.n_stages * b.constriction_length
    expected = (b.x_extent / path) * (1.0 - 0.5 * np.sin(b.angle))
    assert float(r.pulse_factor) == pytest.approx(expected, rel=1e-9)
    assert float(r.pulse_factor) < 1.0


def test_sigma_increases_as_clearance_tightens():
    gap = 0.004
    sigmas = [
        float(solve_sealing(
            BaffleParams(enabled=True, clearance_mm=c), gap, 0.0127).sigma)
        for c in (3.0, 2.0, 1.5, 1.0, 0.5, 0.3)
    ]
    assert all(np.diff(sigmas) > 0)
    assert sigmas[-1] > 0.9  # 0.3mm 已接近零淨流量


def test_delta_eff_approaches_clearance_as_it_tightens():
    """δ→0 時 δ_eff→δ, σ→1−δ/h。"""
    gap = 0.004
    b = BaffleParams(enabled=True, clearance_mm=0.3, n_stages=1, thickness_mm=2.0,
                     edge_radius_mm=0.5)
    r = solve_sealing(b, gap, 0.0127)
    assert float(r.delta_eff) == pytest.approx(b.clearance, rel=0.15)


def test_more_sides_seals_better():
    gap = 0.004
    one = solve_sealing(
        BaffleParams(enabled=True, upstream=True, downstream=False), gap, 0.0127)
    two = solve_sealing(
        BaffleParams(enabled=True, upstream=True, downstream=True), gap, 0.0127)
    assert float(two.sigma) > float(one.sigma)


def test_labyrinth_stages_help_less_than_clearance():
    """誠實的發現: 在層流潤滑極限下, 串列片數的效益遠小於縮小下緣間隙。

    純潤滑阻力串聯時 Couette 泵送量也同比例增加, 因此淨流量比主要由
    δ 決定 (∝δ³ 的阻力 vs ∝δ 的泵送)。真實迷宮密封靠腔室的紊流動能
    耗散, 但此處收縮處 Re = U·δ/ν ≈ 580 偏層流, 該機制很弱。
    """
    gap = 0.004
    base = BaffleParams(enabled=True, clearance_mm=1.5, n_stages=1)
    more_stages = float(solve_sealing(base.replace(n_stages=4), gap, 0.0127).sigma)
    tighter = float(solve_sealing(base.replace(clearance_mm=1.0), gap, 0.0127).sigma)
    one = float(solve_sealing(base, gap, 0.0127).sigma)

    assert more_stages > one            # 多片確實有幫助
    assert tighter - one > more_stages - one   # 但縮小間隙的幫助更大


def test_pulse_factor_attenuated_by_baffle():
    gap = 0.004
    r = solve_sealing(BaffleParams(enabled=True, clearance_mm=1.0), gap, 0.0127)
    assert 0.0 < float(r.pulse_factor) < 0.3


def test_vent_reduces_sealing_but_relieves_pulse():
    """洩壓孔的權衡: 釋放前緣脈衝, 但造成旁通洩漏而降低密封。"""
    gap = 0.004
    closed = solve_sealing(BaffleParams(enabled=True, vent_ratio=0.0), gap, 0.0127)
    vented = solve_sealing(BaffleParams(enabled=True, vent_ratio=0.2), gap, 0.0127)
    assert float(vented.sigma) < float(closed.sigma)
    assert float(vented.pulse_factor) < float(closed.pulse_factor)


def test_angle_increases_constriction_and_deflects():
    gap = 0.004
    flat = BaffleParams(enabled=True, angle_deg=0.0)
    tilted = BaffleParams(enabled=True, angle_deg=45.0)
    assert tilted.constriction_length > flat.constriction_length
    assert float(solve_sealing(tilted, gap, 0.0127).pulse_factor) < float(
        solve_sealing(flat, gap, 0.0127).pulse_factor
    )


# ======================================================================
# 回流剖面 —— 物理核心
# ======================================================================
def test_net_flux_matches_analytic_relation():
    """淨流量必須等於 U·h·(1−σ)/2 —— 這是回流剖面的定義性質。"""
    for clearance in (2.0, 1.5, 1.0, 0.5):
        p, belt, wind, t = _setup(
            BaffleParams(enabled=True, clearance_mm=clearance))
        gap = float(np.asarray(belt.gap_at(t)).reshape(-1)[0])
        sigma = float(wind.sealing_at(t).sigma)
        expected = p.belt_speed * gap * (1.0 - sigma) / 2.0
        assert wind.net_flux(t) == pytest.approx(expected, rel=2e-3)


def test_net_flux_shrinks_toward_zero_with_sealing():
    fluxes = []
    for clearance in (3.0, 2.0, 1.0, 0.5, 0.3):
        _, _, wind, t = _setup(BaffleParams(enabled=True, clearance_mm=clearance))
        fluxes.append(wind.net_flux(t))
    assert all(np.diff(fluxes) < 0)
    assert fluxes[-1] < 0.1 * fluxes[0]


def test_reverse_flow_appears_in_upper_gap():
    """密封後上方應出現逆向流 —— 這正是抵消飄移的機制。"""
    _, belt, wind, t = _setup(BaffleParams(enabled=True, clearance_mm=0.5))
    gap = float(np.asarray(belt.gap_at(t)).reshape(-1)[0])
    z = np.linspace(0.0, gap, 41)
    ux, _, _ = wind.velocity(0.0, 0.0, z, t)
    assert ux.max() > 0  # 底部仍是正向 (no-slip 必然)
    assert ux.min() < 0  # 上方出現逆向


def test_boundary_conditions_survive_sealing():
    """回流項用 Poiseuille 型剖面, 兩壁為零 -> 邊界條件不受影響。"""
    p, belt, wind, t = _setup(BaffleParams(enabled=True, clearance_mm=0.4))
    gap = float(np.asarray(belt.gap_at(t)).reshape(-1)[0])
    ux0, uy0, uz0 = wind.velocity(0.0, 0.0, 0.0, t)
    uxh, uyh, uzh = wind.velocity(0.0, 0.0, gap, t)
    assert float(ux0) == pytest.approx(p.belt_speed, rel=1e-9)
    assert float(uxh) == pytest.approx(0.0, abs=1e-12)
    assert float(uyh) == pytest.approx(0.0, abs=1e-12)
    assert float(uzh) == pytest.approx(0.0, abs=1e-12)


def test_line_stopped_still_zero_field_with_baffle():
    p, _, wind, _ = _setup(BaffleParams(enabled=True, clearance_mm=0.5),
                           belt_speed_m_min=0.0)
    z = np.linspace(0.0, p.gap, 11)
    for t in (0.0, 0.01):
        ux, uy, uz = wind.velocity(0.0, 0.0, z, t)
        assert np.allclose(ux, 0.0, atol=1e-15)
        assert np.allclose(uy, 0.0, atol=1e-15)
        assert np.allclose(uz, 0.0, atol=1e-15)


# ======================================================================
# 橫向 (外八) 的抑制
# ======================================================================
def test_y_end_plates_reduce_lateral_drift():
    _, _, w_off, t = _setup(BaffleParams(enabled=True, y_end_plates=False),
                            n_nozzles=60)
    p_on, belt_on, w_on, t_on = _setup(
        BaffleParams(enabled=True, y_end_plates=True), n_nozzles=60)

    r_off = fire_burst(w_off.params, t_fire=t, wind=w_off, belt=w_off.belt)
    r_on = fire_burst(p_on, t_fire=t_on, wind=w_on, belt=belt_on)
    assert np.max(np.abs(r_on.drift_y_um)) < np.max(np.abs(r_off.drift_y_um))


def test_wider_baffle_reduces_lateral_gradient():
    """擋板比噴印帶寬 -> 逸流出口外移 -> 陣列上的橫向梯度變小。"""
    gap = 0.004
    narrow = solve_sealing(BaffleParams(enabled=True, width_in=1.0), gap, 0.0127)
    wide = solve_sealing(BaffleParams(enabled=True, width_in=1.5), gap, 0.0127)
    assert wide.lateral_half_length > narrow.lateral_half_length
    assert narrow.lateral_half_length == pytest.approx(0.0127, rel=1e-6)


# ======================================================================
# 飄移降幅與零飄移點
# ======================================================================
def test_baffle_reduces_drift_substantially():
    _, _, w_off, t_off = _setup(BaffleParams(enabled=False), n_nozzles=40)
    p_on, belt_on, w_on, t_on = _setup(
        BaffleParams(enabled=True, clearance_mm=1.5), n_nozzles=40)

    off = fire_burst(w_off.params, t_fire=t_off, wind=w_off, belt=w_off.belt)
    on = fire_burst(p_on, t_fire=t_on, wind=w_on, belt=belt_on)
    assert on.mean_drift_um < 0.6 * off.mean_drift_um


def test_drift_flips_sign_when_oversealed():
    """密封過度 -> 回流佔上風 -> 飄移變成負向 (往上游)。"""
    _, belt_l, w_l, t_l = _setup(
        BaffleParams(enabled=True, clearance_mm=2.5), n_nozzles=20)
    p_t, belt_t, w_t, t_t = _setup(
        BaffleParams(enabled=True, clearance_mm=0.4, thickness_mm=1.0,
                     edge_radius_mm=0.5), n_nozzles=20)

    loose = fire_burst(w_l.params, t_fire=t_l, wind=w_l, belt=belt_l)
    tight = fire_burst(p_t, t_fire=t_t, wind=w_t, belt=belt_t)
    assert float(np.mean(loose.drift_x_um)) > 0
    assert float(np.mean(tight.drift_x_um)) < 0


def test_zero_drift_clearance_found_and_mechanically_safe():
    p = SimParams(baffle=BaffleParams(enabled=True, thickness_mm=1.0,
                                      edge_radius_mm=0.5), n_nozzles=20)
    sol = find_zero_drift_clearance(p)
    assert sol.found
    assert 0.3 < sol.clearance_mm < 3.0
    assert abs(sol.drift_x_um) < 1.0
    assert 0.0 < sol.sigma < 1.0
    assert sol.collision is not None and sol.collision.ok


def test_zero_drift_requires_both_sides():
    """單側擋板無法形成回流 -> 不存在零飄移點。"""
    p = SimParams(baffle=BaffleParams(enabled=True, upstream=True,
                                      downstream=False), n_nozzles=8)
    assert not find_zero_drift_clearance(p).found


def test_zero_drift_reports_no_root_when_disabled():
    sol = find_zero_drift_clearance(SimParams(n_nozzles=8))
    assert not sol.found
    assert "未啟用" in sol.note


# ======================================================================
# 亂流散布 (前緣圓角)
# ======================================================================
def test_crosswind_response_is_positive_and_physical():
    """單位橫風的落點位移應為正, 且遠小於「完全跟隨」的 u·t_f。"""
    p = SimParams(n_nozzles=8)
    resp = crosswind_response(p)
    assert resp > 0
    assert resp < 1.0 * p.estimated_flight_time()  # 部分響應而非完全跟隨


def test_sharper_edge_gives_more_scatter():
    base = BaffleParams(enabled=True, thickness_mm=2.0, clearance_mm=1.0)
    scatters = []
    for r in (0.05, 0.2, 0.4):
        p = SimParams(baffle=base.replace(edge_radius_mm=r), n_nozzles=4)
        scatters.append(turbulence_scatter(p, BeltKinematics(p),
                                          steady_fire_time(p)))
    assert all(np.diff(scatters) < 0)
    assert scatters[0] > 0


def test_fully_rounded_nose_kills_separation():
    """圓角達下緣間隙的一半即完全抑制分離。"""
    p = SimParams(
        baffle=BaffleParams(enabled=True, thickness_mm=2.0, clearance_mm=1.0,
                            edge_radius_mm=0.6), n_nozzles=4)
    assert turbulence_scatter(p, BeltKinematics(p), steady_fire_time(p)) == 0.0


def test_no_scatter_without_x_baffles():
    p = SimParams(
        baffle=BaffleParams(enabled=True, upstream=False, downstream=False,
                            y_end_plates=True, edge_radius_mm=0.0), n_nozzles=4)
    assert turbulence_scatter(p, BeltKinematics(p), steady_fire_time(p)) == 0.0


# ======================================================================
# 機械餘裕與參數驗證
# ======================================================================
def test_collision_levels():
    b = BaffleParams(enabled=True, box_height_tolerance_mm=0.5)
    assert check_collision(b.replace(clearance_mm=2.0), 0.15).level == "safe"
    assert check_collision(b.replace(clearance_mm=0.75), 0.15).level == "warn"
    assert check_collision(b.replace(clearance_mm=0.4), 0.6).level == "collision"


def test_light_box_needs_more_clearance():
    """輕箱跳得明顯 (振幅 ∝ 1/m) -> 需要更大的下緣間隙。"""
    b = BaffleParams(enabled=True, clearance_mm=0.8)
    light = BeltKinematics(SimParams(box_mass_kg=5.0)).bounce_amplitude_mm
    heavy = BeltKinematics(SimParams(box_mass_kg=50.0)).bounce_amplitude_mm
    assert check_collision(b, light).margin_mm < check_collision(b, heavy).margin_mm


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width_in": 0.9},
        {"width_in": 1.6},
        {"clearance_mm": 0.2},
        {"clearance_mm": 3.5},
        {"thickness_mm": 0.0},
        {"angle_deg": 70.0},
        {"n_stages": 0},
        {"stage_pitch_mm": 0.0},
        {"edge_radius_mm": -0.1},
        {"vent_ratio": 0.5},
        {"x_extent_mm": 0.0},
        {"thickness_mm": 1.0, "edge_radius_mm": 0.6},  # 圓角超過半厚, 幾何不可能
    ],
)
def test_invalid_designs_rejected(kwargs):
    with pytest.raises(ValueError):
        BaffleParams(**kwargs)


def test_edge_radius_at_half_thickness_is_allowed():
    """完全圓化的鼻端 (半徑 = 半厚) 是合法的極限。"""
    BaffleParams(thickness_mm=1.0, edge_radius_mm=0.5)


def test_sim_params_validates_nested_baffle():
    with pytest.raises(ValueError):
        SimParams(baffle=BaffleParams(width_in=2.0))


def test_effective_plate_x_grows_with_baffle():
    off = SimParams()
    on = SimParams(baffle=BaffleParams(enabled=True))
    assert on.effective_plate_x > off.effective_plate_x
    assert on.effective_plate_x == pytest.approx(on.baffle.total_x_footprint)


def test_design_summary_renders():
    b = BaffleParams(enabled=True)
    text = design_summary(b, 0.004, 0.0127, 5.8333)
    assert "密封有效度" in text
    assert "擋板設計" in text
    assert "未啟用" in design_summary(BaffleParams(enabled=False), 0.004, 0.0127, 5.8)
