"""整合層: 飄移誤差基準、外八圖樣、參數趨勢。"""

import numpy as np
import pytest

from sim.belt import BeltKinematics
from sim.config import SimParams
from sim.simulate import (
    fire_burst,
    steady_fire_time,
    sweep_over_cycle,
    sweep_parameter,
)
from sim.windfield import WindField


def _burst(**kwargs):
    p = SimParams(**kwargs)
    belt = BeltKinematics(p)
    wind = WindField(p, belt)
    return p, fire_burst(p, t_fire=steady_fire_time(p), belt=belt, wind=wind)


# ----------------------------------------------------------------------
# 基準與量級
# ----------------------------------------------------------------------
def test_line_stopped_gives_zero_drift():
    """產線停止 -> 風場為零 -> 300 顆墨滴零飄移。"""
    _, r = _burst(belt_speed_m_min=0.0, n_nozzles=20)
    assert np.allclose(r.drift_x, 0.0, atol=1e-15)
    assert np.allclose(r.drift_y, 0.0, atol=1e-15)


def test_baseline_drift_magnitude_is_physical():
    """350 m/min 下飄移應落在 100~200 µm 量級 (規劃階段手算估計 ~145 µm)。"""
    _, r = _burst(belt_speed_m_min=350.0, droplet_pl=30.0, gap_mm=4.0, n_nozzles=60)
    assert 100.0 < r.mean_drift_um < 200.0


def test_baseline_drift_in_dot_pitches():
    """約 1.5~2 個點距 —— 對 300 dpi 噴印是顯著的品質問題。

    必須用真實的 300 噴嘴: 點距 = 噴印帶高度 / 噴嘴數, 減少噴嘴數會
    連帶放大點距, 使「換算點距」失真。
    """
    _, r = _burst(belt_speed_m_min=350.0, droplet_pl=30.0, n_nozzles=300)
    assert r.params.nozzle_pitch * 1e6 == pytest.approx(84.667, rel=1e-3)
    assert 1.0 < float(np.max(r.drift_in_dot_pitches)) < 3.0


def test_drift_baseline_removes_box_travel():
    """飛行期間箱移 ~3mm 必須被基準扣掉, 否則會淹沒 ~0.15mm 的真實誤差。"""
    p, r = _burst(belt_speed_m_min=350.0, n_nozzles=20)
    assert p.box_travel_during_flight * 1e6 > 2000.0  # 箱移 > 2000 µm
    assert r.mean_drift_um < 300.0  # 但飄移誤差仍在數百 µm 以下


def test_drift_is_downstream():
    """氣流由箱面帶動 -> 墨滴被往箱子前進方向帶偏。"""
    _, r = _burst(belt_speed_m_min=350.0, n_nozzles=20)
    assert np.all(r.drift_x > 0.0)


# ----------------------------------------------------------------------
# 300 顆墨滴的「外八」圖樣
# ----------------------------------------------------------------------
def test_lateral_drift_is_symmetric_outward_fan():
    """陣列中央 Δy≈0, 兩端對稱外偏 —— 這就是「外八」。"""
    p, r = _burst(belt_speed_m_min=350.0, n_nozzles=300)
    dy = r.drift_y_um
    mid = p.n_nozzles // 2
    assert abs(float(dy[mid])) < abs(float(dy[0])) / 5
    assert float(dy[0]) * float(dy[-1]) < 0  # 兩端反向
    assert float(dy[0]) == pytest.approx(-float(dy[-1]), rel=1e-6)


def test_lateral_drift_grows_monotonically_outward():
    _, r = _burst(belt_speed_m_min=350.0, n_nozzles=300)
    half = r.drift_y_um[150:]
    assert np.all(np.diff(half) > -1e-9)


def test_longitudinal_drift_nearly_uniform_across_array():
    """Δx 幾乎與噴嘴位置無關 (25mm 陣列內縱向氣流均勻)。"""
    _, r = _burst(belt_speed_m_min=350.0, n_nozzles=300)
    dx = r.drift_x_um
    assert (dx.max() - dx.min()) < 0.05 * dx.mean()


def test_head_offset_moves_array_without_breaking_symmetry():
    """噴頭在 Y 向平移時, 飄移圖樣相對陣列仍然對稱。

    這是正確的物理: 主導橫向飄移的是噴嘴板底下的擠壓膜逸流, 而噴嘴
    陣列永遠固定在噴嘴板上, 整個頭平移時逸流圖樣跟著一起走。

    模型邊界: 箱緣 (±250mm) 的邊界逸流未納入模型 —— 已確認的配置是
    噴頭對準箱子中線, 25.4mm 陣列只佔箱寬 5%, 距箱緣超過 230mm,
    箱緣效應傳不到。若日後改成噴頭大幅偏向箱緣安裝, 此模型需擴充。
    """
    p = SimParams(belt_speed_m_min=350.0, n_nozzles=100, head_y_offset_mm=8.0)
    belt = BeltKinematics(p)
    r = fire_burst(p, t_fire=steady_fire_time(p), belt=belt, wind=WindField(p, belt))

    # 陣列整體平移了
    assert float(np.mean(r.nozzle_y)) == pytest.approx(0.008, rel=1e-6)
    # 但相對陣列的飄移圖樣維持對稱
    assert float(r.drift_y_um[0]) == pytest.approx(-float(r.drift_y_um[-1]), rel=1e-6)


# ----------------------------------------------------------------------
# 參數趨勢
# ----------------------------------------------------------------------
def test_drift_increases_with_belt_speed():
    sw = sweep_parameter(
        SimParams(n_nozzles=20), "belt_speed_m_min", [0.0, 100.0, 200.0, 350.0]
    )
    assert np.all(np.diff(sw.mean_drift_um) > 0)


def test_smaller_droplet_drifts_more():
    """10 pL 飄移 > 50 pL (小墨滴 tau 小, 響應快)。"""
    sw = sweep_parameter(SimParams(n_nozzles=20), "droplet_pl", [10.0, 30.0, 50.0])
    assert sw.mean_drift_um[0] > sw.mean_drift_um[-1]
    assert np.all(np.diff(sw.mean_drift_um) < 0)


def test_larger_gap_drifts_more():
    """5mm 間隙飄移 > 3mm (飛行時間較長, 暴露在氣流中更久)。"""
    sw = sweep_parameter(SimParams(n_nozzles=20), "gap_mm", [3.0, 4.0, 5.0])
    assert np.all(np.diff(sw.mean_drift_um) > 0)


def test_sweeps_are_smooth():
    """掃描曲線應平滑, 無數值爆掉。"""
    sw = sweep_parameter(
        SimParams(n_nozzles=12), "belt_speed_m_min", np.linspace(0, 350, 12)
    )
    assert np.all(np.isfinite(sw.mean_drift_um))
    second_diff = np.diff(sw.mean_drift_um, 2)
    assert np.max(np.abs(second_diff)) < 0.5 * np.max(sw.mean_drift_um)


# ----------------------------------------------------------------------
# 循環掃描
# ----------------------------------------------------------------------
def test_cycle_scan_covers_both_phases():
    p = SimParams(belt_speed_m_min=350.0, n_nozzles=12, box_spacing_cm=50.0)
    scan = sweep_over_cycle(p, n_samples=16)
    assert scan.confinement.max() > 0.9
    assert scan.confinement.min() < 0.1
    assert np.all(np.isfinite(scan.mean_drift_um))


def test_drift_peaks_are_larger_than_steady_state():
    """前緣撞擊的暫態應造成高於箱下穩態的飄移尖峰。"""
    p = SimParams(belt_speed_m_min=350.0, n_nozzles=12)
    scan = sweep_over_cycle(p, n_samples=40)
    steady = scan.mean_drift_um[scan.confinement > 0.95]
    assert steady.size > 0
    assert scan.max_drift_um.max() > 1.05 * float(np.median(steady))


def test_cycle_scan_period_matches_geometry():
    p = SimParams(belt_speed_m_min=350.0, n_nozzles=8, box_spacing_cm=30.0)
    scan = sweep_over_cycle(p, n_samples=8)
    assert scan.times[-1] < p.cycle_period
    assert p.cycle_frequency_hz == pytest.approx(7.29, rel=0.02)


# ----------------------------------------------------------------------
# 報表
# ----------------------------------------------------------------------
def test_summaries_render():
    p, r = _burst(n_nozzles=8)
    assert "飄移" in r.summary()
    assert "產線速度" in p.summary()
