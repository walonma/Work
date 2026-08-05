"""間隙風場模型 —— 四分量疊加 + 週期循環。

箱子連續不間斷通過 (間隔 30~50cm), 因此噴頭下方的風場是週期循環的。
350 m/min 下循環週期 137~171ms -> 擾動頻率 5.8~7.3 Hz。循環四階段：

  1. 開放區   : 噴頭下方無箱, 50cm 空曠 -> Couette 消失, 只剩前一箱尾流
  2. 前緣撞擊 : 箱前緣抵達, 空氣被活塞式推擠 -> 最劇烈的暫態脈衝
  3. 拘束間隙 : 箱體在噴頭下 -> 穩態 Couette 剪切 + 擠壓膜逸流
  4. 後緣脫離 : 箱尾通過 -> 尾流回吸、負壓

四個速度分量：

  ux = c·U·(1−ζ)                          Couette 剪切 (階段 3 主導)
     + A_edge(t,x)·6ζ(1−ζ)                前/後緣壓差脈衝 (Poiseuille 型)
     + (1−c)·wake                          箱間開放區的尾流回流
  uy = B(t)·sgn(η)|η|^p·6ζ(1−ζ)           橫向擠壓膜逸流
  uz = C_v·U·(env)·[c·6ζ(1−ζ) + (1−c)·ζ]  前緣抬升 / 後緣下沉

其中 ζ = z/h (無因次高度), η = (y − y_板心)/L_y。

關鍵尺度 L_y = 噴嘴板半長 = 12.7mm, 而非箱半寬 250mm。噴嘴板是塊
小平板 (約 25mm × 15mm), 空氣從底下往四面八方逸散, 側向只需走
12.7mm。陣列中央的空氣要走最遠才能側向逸出 -> 橫向速度 ≈ 0;
兩端緊鄰逸出口 -> 明顯外流。這就是 300 顆墨滴呈現對稱「外八」
飄移圖樣的來源。
"""

from __future__ import annotations

import numpy as np

from .belt import BeltKinematics
from .config import SimParams


class WindField:
    """3~5mm 間隙內的解析風場。

    所有分量都正比於箱速 U, 因此產線停止時全場為零。
    完全拘束 (c=1) 時邊界條件精確成立::

        ux(z=0) = U   (箱頂面 no-slip, 帶著空氣走)
        ux(z=h) = 0   (噴嘴板 no-slip, 靜止)
        uy(z=0) = uy(z=h) = 0
        uz(z=0) = uz(z=h) = 0
    """

    def __init__(self, params: SimParams, belt: BeltKinematics | None = None) -> None:
        self.params = params
        self.belt = belt if belt is not None else BeltKinematics(params)

    # ------------------------------------------------------------------
    # 主查詢
    # ------------------------------------------------------------------
    def velocity(self, x, y, z, t):
        """回傳 (ux, uy, uz), 形狀為 x/y/z/t 廣播後的形狀。

        x : 相對噴頭的縱向位置 (m), 噴頭在 x=0
        y : 橫向位置 (m), 箱子中線為 0
        z : 高度 (m), 箱頂面為 0
        t : 時間 (s)
        """
        p = self.params
        b = self.belt

        x, y, z, t = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (x, y, z, t))
        )

        u_box = np.asarray(b.box_speed_at(t), dtype=float)
        gap = np.asarray(b.gap_at(t), dtype=float)
        conf = np.asarray(b.confinement(t), dtype=float)

        zeta = np.clip(z / gap, 0.0, 1.0)
        prof_couette = 1.0 - zeta
        prof_pois = 6.0 * zeta * (1.0 - zeta)

        # --- 1. Couette 剪切流 (階段 3 主導) ---
        ux = conf * u_box * prof_couette

        # --- 2. 前/後緣壓差脈衝 ---
        a_edge, env_lead, env_trail = self._edge_amplitude(x, t, u_box)
        ux = ux + a_edge * prof_pois

        # --- 3. 箱間開放區的尾流回流 ---
        # 噴嘴板是固定實體壁面, 不論下方有無箱子都存在, 因此 z=h 的
        # no-slip 必須無條件成立 -> 尾流也要帶 (1-zeta) 剖面。
        s_trail = np.asarray(b.trailing_edge_offset(t), dtype=float)
        wake = -p.c_wake * u_box * np.exp(
            -np.maximum(s_trail - x, 0.0) / p.wake_length_m
        )
        ux = ux + (1.0 - conf) * wake * prof_couette

        # --- 4. 橫向擠壓膜逸流 (產生 300 顆墨滴的差異) ---
        eta = (y - p.head_y_offset_mm / 1000.0) / p.plate_half_y
        b_lat = p.c_lateral * a_edge + p.c_lateral_steady * conf * u_box
        uy = b_lat * np.sign(eta) * np.abs(eta) ** p.p_lateral * prof_pois

        # --- 5. 前緣抬升 / 後緣下沉 ---
        # 拘束時上下兩壁皆不可穿透 -> Poiseuille 型 (兩端為零);
        # 開放時下方是 50cm 空曠區可自由流動, 但上方的噴嘴板仍是實體壁面
        # -> 用 (1-zeta): 底部最大、板面為零。兩種情形 z=h 都嚴格為零。
        prof_vert = conf * prof_pois + (1.0 - conf) * prof_couette
        uz = p.c_vertical * u_box * (env_lead - 0.5 * env_trail) * prof_vert

        return ux, uy, uz

    def velocity_at(self, pos, t):
        """給定 (N,3) 的位置陣列, 回傳 (N,3) 的氣流速度。"""
        pos = np.asarray(pos, dtype=float)
        ux, uy, uz = self.velocity(pos[..., 0], pos[..., 1], pos[..., 2], t)
        return np.stack([ux, uy, uz], axis=-1)

    # ------------------------------------------------------------------
    # 內部
    # ------------------------------------------------------------------
    def _edge_amplitude(self, x, t, u_box):
        """前/後緣壓差脈衝的振幅與包絡。

        同時疊加「目前箱子」與「下一個箱子」的前緣脈衝 —— 這正是
        30cm 間距時前一箱尾流尚未消散、下一箱前緣已抵達的干涉來源。
        """
        p = self.params
        s_cur, s_next = self.belt.leading_edge_offsets(t)
        s_cur = np.asarray(s_cur, dtype=float)
        s_next = np.asarray(s_next, dtype=float)
        s_trail = np.asarray(self.belt.trailing_edge_offset(t), dtype=float)

        sigma = p.sigma_edge
        env_lead = np.exp(-(((s_cur - x) / sigma) ** 2)) + np.exp(
            -(((s_next - x) / sigma) ** 2)
        )
        # 後緣包絡較寬較弱
        env_trail = np.exp(-(((s_trail - x) / (1.5 * sigma)) ** 2))

        a_lead = p.c_leading * u_box * env_lead
        a_trail = -p.c_trailing * u_box * env_trail
        return a_lead + a_trail, env_lead, env_trail

    # ------------------------------------------------------------------
    # 供繪圖用的切面
    # ------------------------------------------------------------------
    def slice_xz(self, t, x_half_span=0.04, nx=81, nz=41):
        """X-Z 側視切面 (5-1a)。回傳 (X, Z, Ux, Uz, speed), 單位 m 與 m/s。"""
        gap = float(np.asarray(self.belt.gap_at(t)).reshape(-1)[0])
        xs = np.linspace(-x_half_span, x_half_span, nx)
        zs = np.linspace(0.0, gap, nz)
        X, Z = np.meshgrid(xs, zs, indexing="xy")
        ux, uy, uz = self.velocity(X, 0.0, Z, t)
        speed = np.sqrt(ux**2 + uy**2 + uz**2)
        return X, Z, ux, uz, speed

    def slice_yz(self, t, ny=81, nz=41):
        """Y-Z 切面 (5-1b), 跨噴嘴陣列範圍。回傳 (Y, Z, Uy, Uz, Ux)。"""
        p = self.params
        gap = float(np.asarray(self.belt.gap_at(t)).reshape(-1)[0])
        span = p.plate_half_y * 1.2
        ys = np.linspace(-span, span, ny) + p.head_y_offset_mm / 1000.0
        zs = np.linspace(0.0, gap, nz)
        Y, Z = np.meshgrid(ys, zs, indexing="xy")
        ux, uy, uz = self.velocity(0.0, Y, Z, t)
        return Y, Z, uy, uz, ux

    def centreline_series(self, t_array, z_frac=0.5):
        """間隙中線 (z = z_frac·h) 的風速時間序列 (5-1c)。

        回傳 (ux, uy, uz, conf), conf 為拘束因子, 用來標示循環階段。
        """
        t_array = np.asarray(t_array, dtype=float)
        gap = np.asarray(self.belt.gap_at(t_array), dtype=float)
        z = z_frac * gap
        ux, uy, uz = self.velocity(0.0, 0.0, z, t_array)
        conf = np.asarray(self.belt.confinement(t_array), dtype=float)
        return ux, uy, uz, conf

    def gap_mean_speed(self, t, n=41):
        """間隙內 |u| 的厚度平均 (m/s) —— UI 數值面板用。"""
        gap = float(np.asarray(self.belt.gap_at(t)).reshape(-1)[0])
        zs = np.linspace(0.0, gap, n)
        ux, uy, uz = self.velocity(0.0, 0.0, zs, t)
        return float(np.mean(np.sqrt(ux**2 + uy**2 + uz**2)))

    def gap_max_speed(self, t, n=41):
        """間隙內 |u| 的最大值 (m/s)。"""
        gap = float(np.asarray(self.belt.gap_at(t)).reshape(-1)[0])
        zs = np.linspace(0.0, gap, n)
        ux, uy, uz = self.velocity(0.0, 0.0, zs, t)
        return float(np.max(np.sqrt(ux**2 + uy**2 + uz**2)))
