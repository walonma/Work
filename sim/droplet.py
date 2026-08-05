"""墨滴動力學 —— 完整彈道 + Schiller-Naumann 阻力, 對 300 顆向量化。

為何必須用完整彈道模型而非「墨滴瞬間跟隨氣流」:

  鬆弛時間 tau = rho_ink·d²/(18·mu) ≈ 2.9 ms  (30um 墨滴)
  飛行時間 t_f ≈ 0.55 ms
  Stokes 數 St = tau / t_f ≈ 5    >> 1

墨滴根本來不及跟隨氣流, 只部分響應。若簡化成「瞬間跟隨」會高估
飄移量 5 倍以上。這同時解釋了為何 10pL 小墨滴比 50pL 大墨滴飄得更多
(小墨滴 tau 小, 響應快)。

阻力採 Schiller-Naumann 修正, 因為墨滴 Re_p ≈ 10 已超出純 Stokes
(Re<1) 的適用範圍::

    F_drag = 3·pi·mu·d·(1 + 0.15·Re^0.687)·(u_air − v)

注意這個形式等價於 ½ρC_d A|u_rel|u_rel 但沒有 Re→0 的奇異點:
代入 C_d = (24/Re)(1+0.15Re^0.687) 與 A = pi d²/4 後 ½ρC_dA|u|u
恰好化簡成上式。除以質量後::

    a_drag = (1/tau)·(1 + 0.15·Re^0.687)·(u_air − v)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GRAVITY, SimParams


@dataclass
class FlightResult:
    """一次噴發 (300 顆墨滴) 的飛行結果, 全部在實驗室座標系。"""

    landing_x: np.ndarray  # (N,) 落點 X (m)
    landing_y: np.ndarray  # (N,) 落點 Y (m)
    flight_time: np.ndarray  # (N,) 飛行時間 (s)
    launch_z: float  # 噴發時的實際間隙 (m)
    trajectory: np.ndarray | None = None  # (n_frames, N, 3)
    trajectory_time: np.ndarray | None = None  # (n_frames,)
    landed: np.ndarray | None = None  # (N,) 是否成功落地

    @property
    def n(self) -> int:
        return self.landing_x.size


class DropletSwarm:
    """一整排 300 顆墨滴的向量化積分器。"""

    def __init__(self, params: SimParams) -> None:
        self.params = params

    # ------------------------------------------------------------------
    def _acceleration(self, pos, vel, t, wind):
        """回傳 (N,3) 加速度。wind 為 None 時代表無風的標稱彈道。"""
        p = self.params
        if wind is None:
            u_air = np.zeros_like(vel)
        else:
            u_air = wind.velocity_at(pos, t)

        rel = u_air - vel
        speed = np.linalg.norm(rel, axis=-1, keepdims=True)

        d = p.droplet_diameter
        re_p = p.air.density * d * speed / p.air.viscosity
        correction = 1.0 + 0.15 * np.power(np.maximum(re_p, 0.0), 0.687)

        tau = p.relaxation_time
        a_drag = correction * rel / tau

        a = a_drag.copy()
        a[..., 2] -= GRAVITY
        return a

    # ------------------------------------------------------------------
    def fly(self, wind, t0=0.0, launch_gap=None, record_frames=0) -> FlightResult:
        """從噴嘴板積分到箱頂面 (z=0)。

        wind         : WindField 或 None (None = 無風的標稱彈道基準)
        t0           : 噴發時刻 (s)
        launch_gap   : 噴發時的實際間隙 (m)。None 時取標稱值
        record_frames: 要保留幾個軌跡取樣點 (0 = 不保留)
        """
        p = self.params
        n = p.n_nozzles
        dt = p.dt_droplet

        z0 = float(launch_gap) if launch_gap is not None else p.gap

        pos = np.zeros((n, 3))
        pos[:, 1] = p.nozzle_y()
        pos[:, 2] = z0

        vel = np.zeros((n, 3))
        vel[:, 2] = -p.eject_velocity

        landed = np.zeros(n, dtype=bool)
        land_x = np.zeros(n)
        land_y = np.zeros(n)
        land_t = np.zeros(n)

        frames: list[np.ndarray] = []
        frame_times: list[float] = []
        record_every = 0
        if record_frames > 0:
            record_every = max(1, p.max_flight_steps // record_frames)

        for step in range(p.max_flight_steps):
            if landed.all():
                break

            t = t0 + step * dt
            if record_every and step % record_every == 0:
                frames.append(pos.copy())
                frame_times.append(t)

            new_pos, new_vel = self._rk4_step(pos, vel, t, dt, wind)

            # 偵測穿越 z=0, 用線性內插求精確落點
            crossing = (~landed) & (new_pos[:, 2] <= 0.0)
            if crossing.any():
                z_a = pos[crossing, 2]
                z_b = new_pos[crossing, 2]
                denom = z_a - z_b
                frac = np.where(np.abs(denom) > 1e-18, z_a / np.where(denom == 0, 1.0, denom), 0.0)
                frac = np.clip(frac, 0.0, 1.0)
                land_x[crossing] = pos[crossing, 0] + frac * (new_pos[crossing, 0] - pos[crossing, 0])
                land_y[crossing] = pos[crossing, 1] + frac * (new_pos[crossing, 1] - pos[crossing, 1])
                land_t[crossing] = (t - t0) + frac * dt
                landed |= crossing

            # 已落地的凍結, 其餘前進
            active = ~landed
            pos[active] = new_pos[active]
            vel[active] = new_vel[active]

        if not landed.all():
            # 未落地者以最後位置記錄, 並標記
            rest = ~landed
            land_x[rest] = pos[rest, 0]
            land_y[rest] = pos[rest, 1]
            land_t[rest] = p.max_flight_steps * dt

        traj = np.stack(frames, axis=0) if frames else None
        traj_t = np.asarray(frame_times) if frame_times else None

        return FlightResult(
            landing_x=land_x,
            landing_y=land_y,
            flight_time=land_t,
            launch_z=z0,
            trajectory=traj,
            trajectory_time=traj_t,
            landed=landed.copy(),
        )

    # ------------------------------------------------------------------
    def _rk4_step(self, pos, vel, t, dt, wind):
        """標準 RK4, 狀態為 (位置, 速度)。"""
        a1 = self._acceleration(pos, vel, t, wind)
        k1p, k1v = vel, a1

        p2 = pos + 0.5 * dt * k1p
        v2 = vel + 0.5 * dt * k1v
        a2 = self._acceleration(p2, v2, t + 0.5 * dt, wind)
        k2p, k2v = v2, a2

        p3 = pos + 0.5 * dt * k2p
        v3 = vel + 0.5 * dt * k2v
        a3 = self._acceleration(p3, v3, t + 0.5 * dt, wind)
        k3p, k3v = v3, a3

        p4 = pos + dt * k3p
        v4 = vel + dt * k3v
        a4 = self._acceleration(p4, v4, t + dt, wind)
        k4p, k4v = v4, a4

        new_pos = pos + (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        new_vel = vel + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        return new_pos, new_vel
