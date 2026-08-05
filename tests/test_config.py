"""參數換算與範圍驗證。"""

import math

import pytest

from sim.config import INCH, SimParams


def test_speed_unit_conversion():
    p = SimParams(belt_speed_m_min=350.0)
    assert p.belt_speed == pytest.approx(350.0 / 60.0)
    assert p.belt_speed == pytest.approx(5.8333, rel=1e-4)


def test_nozzle_pitch_is_300_dpi():
    """1 吋 / 300 噴嘴 = 84.67um = 恰好 300 dpi。"""
    p = SimParams(print_height_in=1.0, n_nozzles=300)
    assert p.nozzle_pitch == pytest.approx(INCH / 300.0)
    assert p.nozzle_pitch * 1e6 == pytest.approx(84.667, rel=1e-3)
    assert p.dpi == pytest.approx(300.0)


def test_half_inch_head_doubles_severity():
    """1/2 吋 300 噴嘴 = 600 dpi, 點距減半 -> 同樣飄移的點距數加倍。"""
    p = SimParams(print_height_in=0.5, n_nozzles=300)
    assert p.dpi == pytest.approx(600.0)
    assert p.nozzle_pitch * 1e6 == pytest.approx(42.333, rel=1e-3)


def test_lateral_length_scale_is_plate_not_box():
    """橫向逸流特徵長度是噴嘴板半長 12.7mm, 不是箱半寬 250mm。"""
    p = SimParams(print_height_in=1.0, box_width_m=0.5)
    assert p.plate_half_y == pytest.approx(0.0127, rel=1e-3)
    assert p.plate_half_y < p.box_width_m / 2 / 10


@pytest.mark.parametrize(
    "pl, expect_um",
    [(10.0, 26.73), (30.0, 38.55), (50.0, 45.70)],
)
def test_droplet_diameter(pl, expect_um):
    p = SimParams(droplet_pl=pl)
    assert p.droplet_diameter * 1e6 == pytest.approx(expect_um, rel=1e-3)


def test_volume_diameter_roundtrip():
    p = SimParams(droplet_pl=37.0)
    d = p.droplet_diameter
    v = math.pi / 6.0 * d**3
    assert v == pytest.approx(p.droplet_volume, rel=1e-12)


def test_stokes_number_is_ballistic_dominated():
    """St >> 1 -> 墨滴來不及跟隨氣流, 必須用完整彈道模型。"""
    p = SimParams(droplet_pl=30.0, gap_mm=4.0, eject_velocity=8.0)
    assert p.stokes_number > 1.0
    assert p.relaxation_time > p.estimated_flight_time()


def test_smaller_droplet_responds_faster():
    """小墨滴 tau 小 -> 響應快 -> 飄得更多。"""
    small = SimParams(droplet_pl=10.0)
    big = SimParams(droplet_pl=50.0)
    assert small.relaxation_time < big.relaxation_time


def test_cycle_frequency_matches_train_geometry():
    """擾動頻率 = U / (箱長 + 箱間距)。350 m/min + 30~50cm -> 5.8~7.3 Hz。"""
    for spacing_cm, lo, hi in [(30.0, 7.2, 7.4), (50.0, 5.7, 5.9)]:
        p = SimParams(belt_speed_m_min=350.0, box_spacing_cm=spacing_cm)
        assert p.cycle_frequency_hz == pytest.approx(
            p.belt_speed / (p.box_length_m + p.box_spacing)
        )
        assert lo < p.cycle_frequency_hz < hi


def test_gap_reynolds_transitional():
    p = SimParams(belt_speed_m_min=350.0, gap_mm=4.0)
    assert 1000 < p.gap_reynolds < 2500


def test_box_travel_during_flight_dominates_drift():
    """飛行期間箱移 ~3mm, 遠大於風場飄移 ~0.15mm —— 故需時序補償基準。"""
    p = SimParams(belt_speed_m_min=350.0)
    assert p.box_travel_during_flight * 1000 > 2.0


def test_required_fire_frequency():
    """350 m/min 要 X 向 300 dpi 需要 68.9 kHz, 遠超 TIJ 上限。"""
    p = SimParams(belt_speed_m_min=350.0, print_height_in=1.0, n_nozzles=300)
    assert p.required_fire_frequency_hz / 1000 == pytest.approx(68.9, rel=1e-2)
    assert p.x_dpi < p.dpi / 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"belt_speed_m_min": 400.0},
        {"belt_speed_m_min": -1.0},
        {"box_mass_kg": 4.0},
        {"box_mass_kg": 51.0},
        {"gap_mm": 2.0},
        {"gap_mm": 6.0},
        {"droplet_pl": 5.0},
        {"droplet_pl": 60.0},
        {"box_spacing_cm": 20.0},
        {"box_spacing_cm": 60.0},
    ],
)
def test_out_of_range_rejected(kwargs):
    with pytest.raises(ValueError):
        SimParams(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"belt_speed_m_min": 0.0},
        {"belt_speed_m_min": 350.0},
        {"box_mass_kg": 5.0},
        {"box_mass_kg": 50.0},
        {"gap_mm": 3.0},
        {"gap_mm": 5.0},
        {"droplet_pl": 10.0},
        {"droplet_pl": 50.0},
        {"box_spacing_cm": 30.0},
        {"box_spacing_cm": 50.0},
    ],
)
def test_range_endpoints_accepted(kwargs):
    SimParams(**kwargs)


def test_replace_revalidates():
    p = SimParams()
    assert p.replace(gap_mm=5.0).gap_mm == 5.0
    with pytest.raises(ValueError):
        p.replace(gap_mm=9.0)


def test_nozzle_y_centred_and_spanning_print_height():
    p = SimParams(n_nozzles=300, print_height_in=1.0)
    y = p.nozzle_y()
    assert y.size == 300
    assert float(y.mean()) == pytest.approx(0.0, abs=1e-15)
    span = float(y[-1] - y[0])
    assert span == pytest.approx(299 * p.nozzle_pitch, rel=1e-12)
    assert span < p.print_height


def test_head_offset_shifts_array():
    p = SimParams(head_y_offset_mm=5.0)
    assert float(p.nozzle_y().mean()) == pytest.approx(0.005, rel=1e-9)
