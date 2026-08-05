"""墨滴阻力律與積分器正確性。"""

import math

import numpy as np
import pytest

from sim.config import GRAVITY, SimParams
from sim.droplet import DropletSwarm


class ConstantWind:
    """回傳固定氣流速度的假風場, 用來做解析比對。"""

    def __init__(self, u):
        self.u = np.asarray(u, dtype=float)

    def velocity_at(self, pos, t):
        return np.broadcast_to(self.u, np.shape(pos)).copy()


def test_drag_law_matches_full_cd_formula():
    """驗證 3·pi·mu·d·(1+0.15Re^0.687)·rel 等價於 ½ρ·C_d·A·|rel|·rel。

    這是程式中把 Schiller-Naumann 化簡成無奇異點形式的關鍵代數步驟。
    """
    p = SimParams(droplet_pl=30.0)
    swarm = DropletSwarm(p)

    pos = np.zeros((4, 3))
    vel = np.array(
        [[0.0, 0.0, -8.0], [1.0, 0.5, -5.0], [3.0, -2.0, -1.0], [0.2, 0.0, -0.05]]
    )
    wind = ConstantWind([2.9, 0.3, 0.0])

    a = swarm._acceleration(pos, vel, 0.0, wind)

    # 獨立以完整 Cd 公式重算
    rel = wind.velocity_at(pos, 0.0) - vel
    speed = np.linalg.norm(rel, axis=1, keepdims=True)
    d = p.droplet_diameter
    area = math.pi * d**2 / 4.0
    re = p.air.density * d * speed / p.air.viscosity
    cd = (24.0 / re) * (1.0 + 0.15 * re**0.687)
    force = 0.5 * p.air.density * cd * area * speed * rel
    expected = force / p.droplet_mass
    expected[:, 2] -= GRAVITY

    assert np.allclose(a, expected, rtol=1e-12, atol=1e-12)


def test_drag_reduces_to_stokes_at_low_re():
    """Re -> 0 時修正因子 -> 1, 加速度應退化為 rel/tau - g。"""
    p = SimParams(droplet_pl=10.0)
    swarm = DropletSwarm(p)
    pos = np.zeros((1, 3))
    vel = np.zeros((1, 3))
    tiny = 1e-9
    wind = ConstantWind([tiny, 0.0, 0.0])

    a = swarm._acceleration(pos, vel, 0.0, wind)
    assert a[0, 0] == pytest.approx(tiny / p.relaxation_time, rel=1e-6)
    assert a[0, 2] == pytest.approx(-GRAVITY, rel=1e-12)


def test_droplet_reynolds_is_beyond_pure_stokes():
    """Re_p ~ 10 -> 純 Stokes (Re<1) 不適用, 必須用 Schiller-Naumann。"""
    p = SimParams(droplet_pl=30.0, eject_velocity=8.0)
    re = p.air.density * p.droplet_diameter * p.eject_velocity / p.air.viscosity
    assert re > 5.0


def test_no_wind_gives_straight_vertical_fall():
    """無風時墨滴應垂直落下, 不產生任何橫向位移。"""
    p = SimParams(n_nozzles=8)
    swarm = DropletSwarm(p)
    res = swarm.fly(None, t0=0.0, launch_gap=p.gap)

    assert res.landed.all()
    assert np.allclose(res.landing_x, 0.0, atol=1e-15)
    assert np.allclose(res.landing_y, p.nozzle_y(), atol=1e-15)
    # 所有噴嘴的飛行時間相同
    assert np.allclose(res.flight_time, res.flight_time[0], rtol=1e-12)


def test_flight_time_is_sub_millisecond():
    p = SimParams(gap_mm=4.0, eject_velocity=8.0, n_nozzles=4)
    res = DropletSwarm(p).fly(None, launch_gap=p.gap)
    t_f = float(res.flight_time[0])
    assert 0.2e-3 < t_f < 1.0e-3


def test_uniform_crosswind_drifts_all_droplets_equally():
    p = SimParams(n_nozzles=8)
    swarm = DropletSwarm(p)
    wind = ConstantWind([2.9, 0.0, 0.0])
    res = swarm.fly(wind, launch_gap=p.gap)
    assert np.allclose(res.landing_x, res.landing_x[0], rtol=1e-12)
    assert float(res.landing_x[0]) > 0.0


def test_partial_response_not_full_entrainment():
    """St >> 1 -> 墨滴只部分響應。位移應遠小於「瞬間跟隨氣流」的 u·t_f。

    若簡化成瞬間跟隨會高估飄移數倍 —— 這正是必須用完整彈道模型的理由。
    """
    p = SimParams(droplet_pl=30.0, n_nozzles=4)
    swarm = DropletSwarm(p)
    u_air = 2.9
    res = swarm.fly(ConstantWind([u_air, 0.0, 0.0]), launch_gap=p.gap)
    t_f = float(res.flight_time[0])

    full_entrainment = u_air * t_f
    actual = float(res.landing_x[0])
    assert 0.0 < actual < 0.5 * full_entrainment


def test_smaller_droplet_drifts_more():
    """小墨滴 tau 小 -> 響應快 -> 飄移大。"""
    drifts = {}
    for pl in (10.0, 50.0):
        p = SimParams(droplet_pl=pl, n_nozzles=4)
        res = DropletSwarm(p).fly(ConstantWind([2.9, 0.0, 0.0]), launch_gap=p.gap)
        drifts[pl] = float(res.landing_x[0])
    assert drifts[10.0] > drifts[50.0]


def test_rk4_converges_under_step_refinement():
    """把 dt 減半, 落點變化應可忽略 (證明 dt=1us 已足夠)。"""
    base = SimParams(n_nozzles=4, dt_droplet=1e-6)
    fine = base.replace(dt_droplet=2.5e-7, max_flight_steps=16000)
    wind = ConstantWind([2.9, 0.4, 0.1])

    a = DropletSwarm(base).fly(wind, launch_gap=base.gap)
    b = DropletSwarm(fine).fly(wind, launch_gap=fine.gap)

    assert float(a.landing_x[0]) == pytest.approx(float(b.landing_x[0]), rel=2e-3)
    assert float(a.flight_time[0]) == pytest.approx(float(b.flight_time[0]), rel=2e-3)


def test_upward_gust_lengthens_flight_time():
    """向上的陣風會延長飛行時間 (進而放大橫向飄移)。"""
    p = SimParams(n_nozzles=4)
    swarm = DropletSwarm(p)
    calm = swarm.fly(ConstantWind([0.0, 0.0, 0.0]), launch_gap=p.gap)
    lifted = swarm.fly(ConstantWind([0.0, 0.0, 1.0]), launch_gap=p.gap)
    assert float(lifted.flight_time[0]) > float(calm.flight_time[0])


def test_trajectory_recording():
    p = SimParams(n_nozzles=4)
    res = DropletSwarm(p).fly(None, launch_gap=p.gap, record_frames=20)
    assert res.trajectory is not None
    assert res.trajectory.shape[1:] == (4, 3)
    assert res.trajectory.shape[0] >= 2
    # 高度應單調下降
    z = res.trajectory[:, 0, 2]
    assert np.all(np.diff(z) < 0)
