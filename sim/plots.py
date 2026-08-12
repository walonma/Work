"""matplotlib 圖表輸出 —— 對應需求 5-1 (風場) 與 5-2 (墨滴飄移)。

5-1a  X-Z 側視風速向量場 + 色階
5-1b  Y-Z 切面的橫向擠壓膜逸流
5-1c  間隙中線風速時間序列 (標出循環四階段)
5-2a  300 顆墨滴的飛行軌跡 (縱向 + 橫向)
5-2b  飄移誤差 vs 噴嘴編號 (預期對稱外八)
5-2c  300 個落點的飄移向量俯視圖
5-2d  飄移誤差 vs 箱子通過階段
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .baffle import BaffleParams, check_collision, solve_sealing
from .belt import BeltKinematics
from .config import SimParams
from .simulate import (
    BaffleClearanceSweep,
    BurstResult,
    CycleScan,
    ParameterSweep,
    find_zero_drift_clearance,
    fire_burst,
    steady_fire_time,
)
from .windfield import WindField

# 中文字型 (容器內為 WenQuanYi Zen Hei)
plt.rcParams["font.sans-serif"] = [
    "WenQuanYi Zen Hei",
    "Noto Sans CJK TC",
    "Microsoft JhengHei",
    "DejaVu Sans",
]
plt.rcParams["font.monospace"] = ["WenQuanYi Zen Hei Mono", "DejaVu Sans Mono"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25


def _save(fig, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path)
    plt.close(fig)
    return path


# ======================================================================
# 5-1 風場
# ======================================================================
def plot_wind_xz(params: SimParams, t, outdir: Path, wind: WindField | None = None,
                 title_suffix: str = "") -> Path:
    """5-1a: 3~5mm 間隙的 X-Z 側視風速向量場 + 風速色階。"""
    wind = wind or WindField(params)
    X, Z, Ux, Uz, speed = wind.slice_xz(t, x_half_span=0.04, nx=101, nz=41)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    mesh = ax.pcolormesh(X * 1000, Z * 1000, speed, shading="auto", cmap="viridis")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("風速大小 |u| (m/s)")

    s = 5
    ax.quiver(
        X[::4, ::s] * 1000, Z[::4, ::s] * 1000,
        Ux[::4, ::s], Uz[::4, ::s],
        color="white", scale=60, width=0.0025, alpha=0.85,
    )

    gap_mm = float(np.asarray(wind.belt.gap_at(t)).reshape(-1)[0]) * 1000
    ax.axhline(gap_mm, color="#d62728", lw=3)
    ax.axhline(0.0, color="#8c564b", lw=3)
    ax.text(-39, gap_mm * 0.93, "噴嘴板 (靜止, no-slip)", color="#d62728", fontsize=9, va="top")
    ax.text(-39, gap_mm * 0.07, f"箱頂面 (隨箱移動 {params.belt_speed:.2f} m/s)",
            color="#8c564b", fontsize=9, va="bottom")

    # 標出噴嘴板的 X 向覆蓋範圍
    half = params.plate_x_mm / 2
    ax.plot([-half, half], [gap_mm, gap_mm], color="yellow", lw=5, alpha=0.7,
            solid_capstyle="butt", label=f"噴嘴板 X 寬 {params.plate_x_mm:.0f}mm")

    ax.set_xlabel("X — 沿產線方向, 相對噴頭 (mm)   →箱子前進方向")
    ax.set_ylabel("Z — 高度 (mm)")
    ax.set_title(f"5-1a  間隙風場 X-Z 側視{title_suffix}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, gap_mm)
    return _save(fig, outdir, "5-1a_windfield_xz.png")


def plot_wind_yz(params: SimParams, t, outdir: Path, wind: WindField | None = None,
                 title_suffix: str = "") -> Path:
    """5-1b: Y-Z 切面的橫向擠壓膜逸流 (跨噴嘴陣列範圍)。"""
    wind = wind or WindField(params)
    Y, Z, Uy, Uz, Ux = wind.slice_yz(t, ny=101, nz=41)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    lim = max(np.abs(Uy).max(), 1e-9)
    mesh = ax.pcolormesh(Y * 1000, Z * 1000, Uy, shading="auto",
                         cmap="coolwarm", vmin=-lim, vmax=lim)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("橫向風速 u_y (m/s)   正 = 向外(+Y)")

    s = 5
    ax.quiver(
        Y[::4, ::s] * 1000, Z[::4, ::s] * 1000,
        Uy[::4, ::s], Uz[::4, ::s],
        color="black", scale=6, width=0.0025, alpha=0.7,
    )

    gap_mm = float(np.asarray(wind.belt.gap_at(t)).reshape(-1)[0]) * 1000
    half_y = params.plate_half_y * 1000
    for sign in (-1, 1):
        ax.axvline(sign * half_y, color="#2ca02c", ls="--", lw=1.6)
    ax.axvline(0.0, color="gray", ls=":", lw=1.2)
    ax.axhline(gap_mm, color="#d62728", lw=3)
    ax.axhline(0.0, color="#8c564b", lw=3)

    ax.set_xlabel(f"Y — 噴嘴陣列方向 (mm)   綠線 = 噴嘴板邊緣 ±{half_y:.1f}mm (逸流出口)")
    ax.set_ylabel("Z — 高度 (mm)")
    ax.set_title(
        f"5-1b  橫向擠壓膜逸流 Y-Z 切面{title_suffix}\n"
        f"中央逸流路徑最長→u_y≈0；兩端緊鄰出口→明顯外流 (這是「外八」飄移的來源)"
    )
    ax.set_ylim(0, gap_mm)
    return _save(fig, outdir, "5-1b_windfield_yz.png")


def plot_centreline_series(params: SimParams, outdir: Path,
                           belt: BeltKinematics | None = None, n_cycles: float = 2.0,
                           n_pts: int = 1200) -> Path:
    """5-1c: 間隙中線風速時間序列, 標出循環四階段。"""
    belt = belt or BeltKinematics(params)
    wind = WindField(params, belt)

    period = params.cycle_period
    if not np.isfinite(period):
        period = 1.0
    t = np.linspace(0.0, period * n_cycles, n_pts)
    ux, uy, uz, conf = wind.centreline_series(t, z_frac=0.5)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(t * 1000, ux, color="#1f77b4", lw=1.8, label="u_x 縱向")
    ax1.plot(t * 1000, uz, color="#ff7f0e", lw=1.4, label="u_z 垂直")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.axhline(params.belt_speed / 2, color="#1f77b4", ls=":", lw=1.2,
                label=f"穩態 Couette U/2 = {params.belt_speed/2:.2f} m/s")
    ax1.set_ylabel("間隙中線風速 (m/s)")
    ax1.set_title(
        f"5-1c  間隙中線風速的週期循環  —  "
        f"{params.belt_speed_m_min:.0f} m/min, 箱間距 {params.box_spacing_cm:.0f} cm, "
        f"擾動頻率 {params.cycle_frequency_hz:.2f} Hz"
    )
    ax1.legend(loc="upper right", fontsize=9, ncol=2)

    # 標示循環階段
    for k in range(int(np.ceil(n_cycles)) + 1):
        t0 = k * period * 1000
        t_box_end = t0 + params.box_length_m / params.belt_speed * 1000
        t_end = t0 + period * 1000
        if t0 > t[-1] * 1000:
            break
        ax1.axvspan(t0, min(t_box_end, t[-1] * 1000), color="#2ca02c", alpha=0.07)
        ax1.axvspan(min(t_box_end, t[-1] * 1000), min(t_end, t[-1] * 1000),
                    color="#d62728", alpha=0.07)
        if k == 0:
            ax1.text((t0 + t_box_end) / 2, ax1.get_ylim()[1] * 0.92,
                     "箱體在噴頭下\n(拘束間隙 Couette)", ha="center", fontsize=8, color="#2ca02c")
            ax1.text((t_box_end + t_end) / 2, ax1.get_ylim()[1] * 0.92,
                     "箱間開放區\n(50cm 空曠, 尾流)", ha="center", fontsize=8, color="#d62728")

    ax2.plot(t * 1000, conf, color="#9467bd", lw=1.8)
    ax2.set_ylabel("拘束因子 c")
    ax2.set_xlabel("時間 (ms)")
    ax2.set_ylim(-0.05, 1.05)

    return _save(fig, outdir, "5-1c_centreline_timeseries.png")


# ======================================================================
# 5-2 墨滴飄移
# ======================================================================
def plot_trajectories(result: BurstResult, outdir: Path) -> Path:
    """5-2a: 300 顆墨滴在 3~5mm 間隙內的飛行軌跡 (縱向 + 橫向)。"""
    p = result.params
    traj = result.wind_flight.trajectory
    nom = result.nominal_flight.trajectory
    if traj is None:
        raise ValueError("需要 record_frames > 0 才能繪製軌跡")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    idx = np.linspace(0, p.n_nozzles - 1, 25).astype(int)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(idx)))

    wf, nf = result.wind_flight, result.nominal_flight

    def _path(frames, flight, i, comp):
        """軌跡取樣點 + 內插求得的精確落點, 讓曲線確實接到箱頂面。"""
        m = frames[:, i, 2] > 0
        vals = np.append(frames[m, i, comp], (flight.landing_x, flight.landing_y)[comp][i])
        zs = np.append(frames[m, i, 2], 0.0)
        return vals, zs

    # --- 左: X-Z 縱向飄移 ---
    for c, i in zip(colors, idx):
        xs, zs = _path(traj, wf, i, 0)
        ax1.plot(xs * 1e6, zs * 1000, color=c, lw=1.0, alpha=0.8)
    if nom is not None:
        xs, zs = _path(nom, nf, idx[0], 0)
        ax1.plot(xs * 1e6, zs * 1000, color="k", ls="--", lw=1.8, label="無風標稱彈道")
    ax1.axhline(result.gap_at_fire * 1000, color="#d62728", lw=2.5)
    ax1.axhline(0, color="#8c564b", lw=2.5)
    ax1.set_xlabel("X 偏移 (µm)  →氣流帶動方向")
    ax1.set_ylabel("Z 高度 (mm)")
    ax1.set_title("縱向飄移 (X-Z)")
    ax1.legend(fontsize=8, loc="upper left")

    # --- 右: Y 相對偏移 ---
    y0 = p.nozzle_y()
    for c, i in zip(colors, idx):
        ys, zs = _path(traj, wf, i, 1)
        ax2.plot((ys - y0[i]) * 1e6, zs * 1000, color=c, lw=1.0, alpha=0.85)
    ax2.axvline(0, color="k", lw=0.9, ls=":")
    ax2.axhline(result.gap_at_fire * 1000, color="#d62728", lw=2.5)
    ax2.axhline(0, color="#8c564b", lw=2.5)
    ax2.set_xlabel("Y 相對偏移 (µm)   ←藍=陣列一端  紅=另一端→")
    ax2.set_ylabel("Z 高度 (mm)")
    ax2.set_title("橫向飄移 (Y-Z) — 呈扇形外散")

    fig.suptitle(
        f"5-2a  300 顆墨滴的飛行軌跡  ({p.droplet_pl:.0f} pL, 直徑 {p.droplet_diameter*1e6:.1f} µm, "
        f"間隙 {result.gap_at_fire*1000:.2f} mm, 飛行 {np.mean(result.flight_time)*1e3:.3f} ms)",
        fontsize=11,
    )
    fig.tight_layout()
    return _save(fig, outdir, "5-2a_trajectories.png")


def plot_drift_vs_nozzle(result: BurstResult, outdir: Path) -> Path:
    """5-2b: 飄移誤差 vs 噴嘴編號 (1~300) — 預期對稱外八。"""
    p = result.params
    n = np.arange(1, p.n_nozzles + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)

    dx = result.drift_x_um
    ax1.plot(n, dx, color="#1f77b4", lw=2.0, label="Δx 縱向飄移")
    # 從 0 起算, 否則自動縮放會把 0.1µm 的起伏放大成誤導性的大波形
    ax1.set_ylim(0.0, float(dx.max()) * 1.25)
    ax1.set_ylabel("Δx (µm)")
    ax1.set_title(
        f"5-2b  飄移誤差 vs 噴嘴編號   "
        f"({p.belt_speed_m_min:.0f} m/min, {p.droplet_pl:.0f} pL, 間隙 {p.gap_mm:.1f} mm)"
    )
    ax1.legend(fontsize=9, loc="lower right")
    ax1.text(
        p.n_nozzles * 0.5, float(dx.max()) * 0.45,
        f"縱向飄移在整排噴嘴上幾乎均勻\n"
        f"({dx.min():.2f} ~ {dx.max():.2f} µm, 變化僅 {(dx.max()-dx.min())/dx.mean()*100:.2f}%)\n"
        f"—— 25.4mm 陣列內縱向氣流一致",
        fontsize=9, color="#1f77b4", ha="center", va="center",
    )

    ax2.plot(n, result.drift_y_um, color="#d62728", lw=2.0, label="Δy 橫向飄移")
    ax2.axhline(0, color="k", lw=0.9)
    ax2.axvline(p.n_nozzles / 2, color="gray", ls=":", lw=1.2)
    ax2.set_xlabel("噴嘴編號 (1 ~ 300)")
    ax2.set_ylabel("Δy (µm)")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.annotate(
        "陣列中央 Δy ≈ 0\n(逸出路徑最長)",
        xy=(p.n_nozzles / 2, 0),
        xytext=(p.n_nozzles * 0.60, float(np.min(result.drift_y_um)) * 0.55),
        fontsize=8, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.9),
    )
    ax2.text(
        p.n_nozzles * 0.02, float(np.max(result.drift_y_um)) * 0.75,
        "兩端緊鄰噴嘴板邊緣的逸流出口\n→ 對稱外偏 (「外八」圖樣)",
        fontsize=8, color="#d62728", va="center",
    )

    # 右軸: 換算成點距
    ax3 = ax1.twinx()
    ax3.set_ylim(np.array(ax1.get_ylim()) / (p.nozzle_pitch * 1e6))
    ax3.set_ylabel(f"換算點距 ({p.nozzle_pitch*1e6:.1f} µm @ {p.dpi:.0f} dpi)")
    ax3.grid(False)

    fig.tight_layout()
    return _save(fig, outdir, "5-2b_drift_vs_nozzle.png")


def plot_drift_vectors(result: BurstResult, outdir: Path) -> Path:
    """5-2c: 300 顆墨滴的飄移方向 (俯視箱頂面)。

    Δx (~136µm) 遠大於 Δy (~20µm), 若直接在實際尺度畫箭頭會全部看起來
    朝同一方向。因此左圖用等比例的 Δx-Δy 平面畫出真正的向量扇形,
    右圖直接量化每顆墨滴偏離純縱向的角度。
    """
    p = result.params
    dx, dy = result.drift_x_um, result.drift_y_um
    angle_deg = np.degrees(np.arctan2(dy, dx))
    n = np.arange(1, p.n_nozzles + 1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1.0, 2.2]}
    )

    # --- 左: 等比例的飄移向量扇形 ---
    step = max(1, p.n_nozzles // 40)
    idx = np.arange(0, p.n_nozzles, step)
    colors = plt.cm.coolwarm(np.linspace(0, 1, idx.size))
    for c, i in zip(colors, idx):
        ax1.annotate(
            "", xy=(dy[i], dx[i]), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=c, lw=1.2, alpha=0.9),
        )
    ax1.scatter(dy, dx, c=n, cmap="coolwarm", s=6, zorder=3)

    # 明確含括原點, 否則等比例縮放下箭頭會衝出座標軸
    y_lim = float(np.max(np.abs(dy))) * 1.35
    ax1.set_xlim(-y_lim, y_lim)
    ax1.set_ylim(0.0, float(dx.max()) * 1.10)
    ax1.set_aspect("equal")
    ax1.axvline(0, color="gray", lw=0.9, ls=":")
    ax1.locator_params(axis="y", nbins=5)
    ax1.locator_params(axis="x", nbins=5)
    ax1.set_xlabel("Δy 橫向飄移 (µm)")
    ax1.set_ylabel("Δx 縱向飄移 (µm)  →箱子前進方向")
    ax1.set_title("飄移向量扇形\n(等比例, 原點=噴嘴)", fontsize=10)

    # --- 右: 飄移角度 + 對噴印的實際影響 ---
    ax2.plot(n, angle_deg, color="#d62728", lw=2.0)
    ax2.axhline(0, color="k", lw=0.9)
    ax2.axvline(p.n_nozzles / 2, color="gray", ls=":", lw=1.2)
    ax2.set_xlabel("噴嘴編號 (1 ~ 300)")
    ax2.set_ylabel("飄移方向偏離純縱向的角度 (度)")
    ax2.set_title(
        f"飄移角度  —  兩端各偏 {abs(float(angle_deg[0])):.1f}°, "
        f"全幅 {float(angle_deg[-1] - angle_deg[0]):.1f}°"
    )

    # Δy 沿陣列線性變化 = 整條噴印帶被橫向拉伸
    span_um = float(result.nozzle_y[-1] - result.nozzle_y[0]) * 1e6
    widening_um = float(dy[-1] - dy[0])
    ax2.text(
        p.n_nozzles * 0.03, float(np.max(angle_deg)) * 0.62,
        "Δy 沿陣列近乎線性 → 整條噴印帶被橫向拉伸\n"
        f"總增寬 {widening_um:.1f} µm / {span_um/1000:.1f} mm 帶寬 "
        f"= {widening_um/span_um*100:.3f}%\n"
        f"最外側噴嘴偏移 {abs(float(dy[0])):.1f} µm "
        f"= {abs(float(dy[0]))/(p.nozzle_pitch*1e6):.2f} 個點距",
        fontsize=8.5, color="#d62728", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d62728", alpha=0.85),
    )

    fig.suptitle(
        f"5-2c  300 顆墨滴的飄移方向 (俯視箱頂面)  —  "
        f"合成飄移最大 {result.max_drift_um:.1f} µm = "
        f"{np.max(result.drift_in_dot_pitches):.2f} 個點距 "
        f"(點距 {p.nozzle_pitch*1e6:.1f} µm @ {p.dpi:.0f} dpi)",
        fontsize=11,
    )
    fig.tight_layout()
    return _save(fig, outdir, "5-2c_drift_vectors.png")


def plot_cycle_scan(scan: CycleScan, outdir: Path) -> Path:
    """5-2d: 飄移誤差 vs 箱子通過階段 —— 顯示前緣撞擊的誤差尖峰。"""
    p = scan.params
    t_ms = scan.times * 1000

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True,
                                        gridspec_kw={"height_ratios": [2, 2, 1]})

    ax1.plot(t_ms, scan.gap_mean_speed, color="#1f77b4", lw=2.0)
    ax1.set_ylabel("間隙平均風速 (m/s)")
    ax1.set_title(
        f"5-2d  飄移誤差如何隨箱子通過階段變化  —  "
        f"{p.belt_speed_m_min:.0f} m/min, 箱間距 {p.box_spacing_cm:.0f} cm, "
        f"循環 {p.cycle_period*1000:.0f} ms ({p.cycle_frequency_hz:.2f} Hz)"
    )

    ax2.plot(t_ms, scan.mean_drift_um, color="#d62728", lw=2.0, label="平均合成飄移")
    ax2.plot(t_ms, scan.max_drift_um, color="#ff7f0e", lw=1.4, ls="--", label="最大合成飄移")
    ax2.plot(t_ms, np.abs(scan.edge_drift_y_um), color="#2ca02c", lw=1.4,
             label="最外側噴嘴 |Δy|")
    ax2.set_ylabel("飄移誤差 (µm)")
    ax2.legend(fontsize=9, loc="upper right")

    ax4 = ax2.twinx()
    ax4.set_ylim(np.array(ax2.get_ylim()) / (p.nozzle_pitch * 1e6))
    ax4.set_ylabel("換算點距")
    ax4.grid(False)

    ax3.plot(t_ms, scan.confinement, color="#9467bd", lw=1.8)
    ax3.set_ylabel("拘束因子 c")
    ax3.set_xlabel("循環內的時間 (ms)")
    ax3.set_ylim(-0.05, 1.05)

    t_box_end = p.box_length_m / p.belt_speed * 1000
    for ax in (ax1, ax2, ax3):
        ax.axvspan(0, t_box_end, color="#2ca02c", alpha=0.07)
        ax.axvspan(t_box_end, t_ms[-1], color="#d62728", alpha=0.07)
    ax1.text(t_box_end / 2, ax1.get_ylim()[1] * 0.9, "箱體在噴頭下",
             ha="center", fontsize=9, color="#2ca02c")
    ax1.text((t_box_end + t_ms[-1]) / 2, ax1.get_ylim()[1] * 0.9, "箱間開放區",
             ha="center", fontsize=9, color="#d62728")

    fig.tight_layout()
    return _save(fig, outdir, "5-2d_drift_over_cycle.png")


# ======================================================================
# 參數掃描
# ======================================================================
def plot_sweep(sweep: ParameterSweep, params: SimParams, outdir: Path) -> Path:
    """參數掃描圖: 飄移量與間隙風速隨參數的變化。"""
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(sweep.values, sweep.mean_drift_um, "o-", color="#d62728",
             lw=2.0, ms=4, label="平均合成飄移")
    ax1.plot(sweep.values, sweep.max_drift_um, "s--", color="#ff7f0e",
             lw=1.4, ms=3.5, label="最大合成飄移")
    ax1.plot(sweep.values, np.abs(sweep.edge_drift_y_um), "^-", color="#2ca02c",
             lw=1.4, ms=3.5, label="最外側噴嘴 |Δy|")
    ax1.set_xlabel(sweep.label)
    ax1.set_ylabel("飄移誤差 (µm)")
    ax1.legend(fontsize=9, loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(sweep.values, sweep.gap_mean_speed, ":", color="#1f77b4", lw=1.6)
    ax2.set_ylabel("間隙平均風速 (m/s)", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.grid(False)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 52))
    ax3.set_ylim(np.array(ax1.get_ylim()) / (params.nozzle_pitch * 1e6))
    ax3.set_ylabel(f"換算點距 ({params.dpi:.0f} dpi)")
    ax3.grid(False)

    ax1.set_title(f"參數掃描: 飄移誤差 vs {sweep.label}")
    fig.tight_layout()
    return _save(fig, outdir, f"sweep_{sweep.field}.png")


def plot_startstop(params: SimParams, belt: BeltKinematics, outdir: Path,
                   masses=(5.0, 50.0)) -> Path:
    """啟停過渡: 箱重如何透過機械耦合影響箱速, 進而影響風場。

    箱重對空氣動力學沒有直接影響 —— 這張圖顯示質量的兩條真實路徑:
    驅動力有限造成的整線加速差異, 以及摩擦極限造成的打滑落後。
    """
    from .belt import SpeedCommand

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    cmds = [SpeedCommand(0.0, 0.0), SpeedCommand(0.2, params.belt_speed_m_min),
            SpeedCommand(2.5, 0.0)]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    for c, m in zip(colors, masses):
        p = params.replace(box_mass_kg=float(m))
        b = BeltKinematics(p, commands=cmds, t_end=4.0, dt=2e-4)
        t = np.linspace(0, 4.0, 2000)
        ax1.plot(t, b.belt_speed_at(t) * 60, ls="--", lw=1.2, color=c, alpha=0.65,
                 label=f"皮帶 {m:.0f}kg (a_max={b.belt_accel_limit:.1f} m/s²)")
        ax1.plot(t, b.box_speed_at(t) * 60, lw=2.2, color=c,
                 label=f"箱子 {m:.0f}kg")
        ax2.plot(t, b.slip_at(t) * 60, lw=2.0, color=c,
                 label=f"{m:.0f}kg (跳動 {b.bounce_amplitude_mm:.3f}mm @ {b.bounce_frequency_hz:.0f}Hz)")

    ax1.set_ylabel("速度 (m/min)")
    ax1.set_title(
        "啟停過渡中的質量耦合 —— 箱重「不」直接影響空氣動力學\n"
        "路徑1: 驅動力有限 → 箱越重整線加速越慢   路徑2: 摩擦極限 μg → 箱子打滑落後"
    )
    ax1.legend(fontsize=8, ncol=2, loc="lower right")

    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("打滑量 (皮帶 - 箱) (m/min)")
    ax2.set_xlabel("時間 (s)")
    ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    return _save(fig, outdir, "startstop_mass_coupling.png")


# ======================================================================
# 整組輸出
# ======================================================================
def render_baseline(params: SimParams, outdir: Path, t_fire=None) -> list[Path]:
    """產出基準案例的全套 5-1 / 5-2 圖表。"""
    outdir = Path(outdir)
    belt = BeltKinematics(params)
    wind = WindField(params, belt)
    t_steady = steady_fire_time(params) if t_fire is None else t_fire
    t_lead = 0.0  # 箱前緣正抵達噴頭

    paths = []
    paths.append(plot_wind_xz(params, t_steady, outdir, wind,
                              title_suffix="  (箱體中段 — 穩態 Couette)"))
    paths.append(plot_wind_yz(params, t_lead, outdir, wind,
                              title_suffix="  (箱前緣抵達 — 擠壓膜逸流最強)"))
    paths.append(plot_centreline_series(params, outdir, belt))

    burst = fire_burst(params, t_fire=t_steady, belt=belt, wind=wind, record_frames=80)
    paths.append(plot_trajectories(burst, outdir))
    paths.append(plot_drift_vs_nozzle(burst, outdir))
    paths.append(plot_drift_vectors(burst, outdir))
    return paths


# ======================================================================
# 導風擋板
# ======================================================================
def _draw_baffle_stack(ax, b: BaffleParams, gap_mm: float, side: int,
                       cavity_half_mm: float):
    """在 X-Z 剖面上畫出單側的串列擋板。side=-1 上游, +1 下游。"""
    h_mm = gap_mm - b.clearance_mm            # 擋板的垂直高度
    lean = h_mm * np.tan(b.angle)             # 下緣往外偏移 (斜角導流)
    l_b = b.constriction_length * 1000.0
    x = cavity_half_mm
    for k in range(b.n_stages):
        # 傾斜的平行四邊形: 上緣在 z=gap, 下緣在 z=clearance 且往外偏 lean
        top_in, top_out = side * x, side * (x + l_b)
        bot_in = side * (x + lean)
        bot_out = side * (x + l_b + lean)
        ax.fill(
            [top_in, top_out, bot_out, bot_in],
            [gap_mm, gap_mm, b.clearance_mm, b.clearance_mm],
            facecolor="#9467bd", edgecolor="#5b3a86", lw=1.2, zorder=4,
        )
        # 前緣圓角 (面向來流的那一角)
        if b.edge_radius_mm > 0:
            cx = bot_in if side < 0 else bot_out
            ax.add_patch(plt.Circle(
                (cx, b.clearance_mm + b.edge_radius_mm), b.edge_radius_mm,
                facecolor="#c9a0ff", edgecolor="none", zorder=5))
        x += l_b + (b.stage_pitch_mm if k < b.n_stages - 1 else 0.0)
    return x


def plot_baffle_preview(params: SimParams, outdir: Path, gap: float | None = None) -> Path:
    """擋板設計預覽 —— X-Z 與 Y-Z 剖面 + 氣動效果數值。

    這是「預覽」鍵對應的輸出: 先看設計樣式與密封度, 再開始模擬。
    """
    b = params.baffle
    gap_m = params.gap if gap is None else float(gap)
    gap_mm = gap_m * 1000.0
    seal = solve_sealing(b, gap_m, params.plate_half_y)
    belt = BeltKinematics(params)
    coll = check_collision(b, belt.bounce_amplitude_mm)

    fig = plt.figure(figsize=(13, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1, 1],
                          hspace=0.42, wspace=0.22)
    ax_xz = fig.add_subplot(gs[0, 0])
    ax_yz = fig.add_subplot(gs[1, 0])
    ax_txt = fig.add_subplot(gs[:, 1])
    ax_txt.axis("off")

    cavity_half = b.x_extent_mm / 2.0

    # ---------------- X-Z 剖面 ----------------
    outer = cavity_half
    if b.active:
        if b.upstream:
            outer = max(outer, _draw_baffle_stack(ax_xz, b, gap_mm, -1, cavity_half))
        if b.downstream:
            outer = max(outer, _draw_baffle_stack(ax_xz, b, gap_mm, +1, cavity_half))
    span = outer * 1.35 + 4

    ax_xz.add_patch(plt.Rectangle((-outer, gap_mm), 2 * outer, 1.1,
                                  facecolor="#d62728", zorder=3))
    ax_xz.add_patch(plt.Rectangle((-span, -1.1), 2 * span, 1.1,
                                  facecolor="#a8834a", zorder=3))
    # 噴嘴帶
    ax_xz.plot([-params.plate_x_mm / 2, params.plate_x_mm / 2], [gap_mm, gap_mm],
               color="#e3b341", lw=5, solid_capstyle="butt", zorder=6,
               label=f"噴嘴板 {params.plate_x_mm:.0f}mm")
    # 洩壓孔
    if b.vent_ratio > 0:
        for xv in np.linspace(-cavity_half * 0.6, cavity_half * 0.6, 4):
            ax_xz.plot([xv, xv], [gap_mm, gap_mm + 1.1], color="#010409", lw=3,
                       zorder=7)
        ax_xz.text(0, gap_mm + 1.6, f"洩壓孔 {b.vent_ratio*100:.0f}%",
                   ha="center", fontsize=8, color="#8b949e")

    # 尺寸標註 —— 放在內腔的空白處, 避免與幾何或軸標籤重疊
    x_c = -b.x_extent_mm * 0.30
    ax_xz.annotate("", xy=(x_c, 0), xytext=(x_c, b.clearance_mm),
                   arrowprops=dict(arrowstyle="<->", color="#3fb950", lw=1.3))
    ax_xz.text(x_c + 0.8, b.clearance_mm * 0.55,
               f"下緣間隙 {b.clearance_mm:.2f}mm",
               fontsize=8.5, color="#3fb950", ha="left", va="center")

    x_g = b.x_extent_mm * 0.30
    ax_xz.annotate("", xy=(x_g, 0), xytext=(x_g, gap_mm),
                   arrowprops=dict(arrowstyle="<->", color="#8b949e", lw=1.1))
    ax_xz.text(x_g + 1.0, gap_mm * 0.30, f"間隙\n{gap_mm:.2f}mm",
               fontsize=8.5, color="#8b949e", va="center")

    ax_xz.annotate("", xy=(-cavity_half, gap_mm + 2.4), xytext=(cavity_half, gap_mm + 2.4),
                   arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.2))
    ax_xz.text(0, gap_mm + 2.8, f"內腔 {b.x_extent_mm:.0f}mm", fontsize=8.5,
               color="#1f77b4", ha="center")

    if b.active and b.n_sides >= 2:
        ax_xz.text(0, gap_mm * 0.68,
                   f"零淨流量循環區\nσ = {float(seal.sigma):.3f}",
                   fontsize=8.5, color="#1f77b4", ha="center", va="center",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4",
                             alpha=0.75, lw=0.8))

    ax_xz.set_xlim(-span, span)
    ax_xz.set_ylim(-1.4, gap_mm + 4.2)
    ax_xz.set_xlabel("X — 沿產線 (mm)   →氣流被箱面帶動方向")
    ax_xz.set_ylabel("Z (mm)")
    ax_xz.set_title(
        f"X-Z 剖面   厚度 {b.thickness_mm:.2f}mm · 斜角 {b.angle_deg:.0f}° · "
        f"{b.n_stages} 片串列 @ {b.stage_pitch_mm:.1f}mm · 圓角 R{b.edge_radius_mm:.2f}",
        fontsize=10)
    ax_xz.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax_xz.set_aspect("equal", adjustable="box")

    # ---------------- Y-Z 剖面 ----------------
    half_y = b.half_width * 1000.0 if b.active else params.plate_half_y * 1000
    span_y = half_y * 1.3 + 3
    ax_yz.add_patch(plt.Rectangle((-half_y, gap_mm), 2 * half_y, 1.1,
                                  facecolor="#d62728", zorder=3))
    ax_yz.add_patch(plt.Rectangle((-span_y, -1.1), 2 * span_y, 1.1,
                                  facecolor="#a8834a", zorder=3))
    if b.active and b.y_end_plates:
        l_b = b.constriction_length * 1000.0
        for sgn in (-1, 1):
            ax_yz.add_patch(plt.Rectangle(
                (sgn * half_y - (l_b if sgn > 0 else 0), b.clearance_mm),
                l_b, gap_mm - b.clearance_mm,
                facecolor="#9467bd", edgecolor="#5b3a86", lw=1.2, zorder=4))
    noz = params.print_height * 1000 / 2
    ax_yz.plot([-noz, noz], [gap_mm, gap_mm], color="#e3b341", lw=5,
               solid_capstyle="butt", zorder=6,
               label=f"300 噴嘴 {params.print_height*1000:.1f}mm")
    ax_yz.set_xlim(-span_y, span_y)
    ax_yz.set_ylim(-1.4, gap_mm + 3.4)
    ax_yz.set_xlabel("Y — 噴嘴陣列方向 (mm)")
    ax_yz.set_ylabel("Z (mm)")
    ax_yz.set_title(
        f"Y-Z 剖面   寬度 {b.width_in:.2f} 吋 ({b.width*1000:.1f}mm)"
        f"{' · 含 Y 端板' if b.active and b.y_end_plates else ' · 無 Y 端板'}",
        fontsize=10)
    ax_yz.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax_yz.set_aspect("equal", adjustable="box")

    # ---------------- 數值 ----------------
    colour = {"safe": "#3fb950", "warn": "#f0883e", "collision": "#f85149"}[coll.level]
    body = design_summary_text(params, gap_m)
    ax_txt.text(0.0, 1.0, body, fontsize=9.2, va="top", family="monospace",
                transform=ax_txt.transAxes)
    ax_txt.text(0.0, 0.055, f"機械餘裕：{coll.message}", fontsize=9,
                va="bottom", color=colour, transform=ax_txt.transAxes, wrap=True)

    fig.suptitle("擋板設計預覽", fontsize=13, y=0.98)
    return _save(fig, Path(outdir), "baffle_preview.png")


def design_summary_text(params: SimParams, gap: float) -> str:
    """預覽用的文字摘要 (與 baffle.design_summary 同內容, 方便繪圖取用)。"""
    from .baffle import design_summary

    return design_summary(params.baffle, gap, params.plate_half_y, params.belt_speed)


def plot_baffle_comparison(params: SimParams, outdir: Path) -> Path:
    """有/無擋板的對照 —— 速度剖面、墨滴軌跡、飄移量。"""
    from .windfield import WindField

    cases = [
        ("無擋板", params.replace(baffle=params.baffle.replace(enabled=False)), "#d62728"),
        ("有擋板", params, "#2ca02c"),
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.8))

    for label, p, colour in cases:
        belt = BeltKinematics(p)
        wind = WindField(p, belt)
        t = steady_fire_time(p)
        gap = float(np.asarray(belt.gap_at(t)).reshape(-1)[0])
        sigma = float(wind.sealing_at(t).sigma)

        # --- 速度剖面 ---
        z = np.linspace(0.0, gap, 121)
        ux, _, _ = wind.velocity(0.0, 0.0, z, t)
        ax1.plot(ux, z * 1000, color=colour, lw=2.2,
                 label=f"{label}  σ={sigma:.3f}")

        # --- 軌跡與飄移 ---
        r = fire_burst(p, t_fire=t, belt=belt, wind=wind, record_frames=90)
        traj = r.wind_flight.trajectory
        mid = p.n_nozzles // 2
        m = traj[:, mid, 2] > 0
        xs = np.append(traj[m, mid, 0], r.wind_flight.landing_x[mid]) * 1e6
        zs = np.append(traj[m, mid, 2], 0.0) * 1000
        ax2.plot(xs, zs, color=colour, lw=2.0, label=label)

        n = np.arange(1, p.n_nozzles + 1)
        ax3.plot(n, r.drift_y_um, color=colour, lw=2.0,
                 label=f"{label}  Δx={np.mean(r.drift_x_um):+.1f}µm")

    ax1.axvline(0, color="k", lw=0.9)
    ax1.set_xlabel("u_x (m/s)")
    ax1.set_ylabel("Z 高度 (mm)")
    ax1.set_title("間隙速度剖面\n擋板把貫流變成回流", fontsize=10)
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.text(0.05, 0.12, "← 逆向流抵消飄移", transform=ax1.transAxes,
             fontsize=8.5, color="#2ca02c", va="center")

    ax2.axvline(0, color="k", lw=0.9, ls=":")
    ax2.set_xlabel("X 偏移 (µm)")
    ax2.set_ylabel("Z 高度 (mm)")
    ax2.set_title("墨滴軌跡 (陣列中央)", fontsize=10)
    ax2.legend(fontsize=8.5)

    ax3.axhline(0, color="k", lw=0.9)
    ax3.set_xlabel("噴嘴編號")
    ax3.set_ylabel("Δy 橫向飄移 (µm)")
    ax3.set_title("300 噴嘴的橫向飄移", fontsize=10)
    ax3.legend(fontsize=8.5)

    fig.suptitle(
        f"擋板效果對照   {params.belt_speed_m_min:.0f} m/min · 間隙 {params.gap_mm:.1f}mm · "
        f"下緣間隙 {params.baffle.clearance_mm:.2f}mm",
        fontsize=12)
    fig.tight_layout()
    return _save(fig, Path(outdir), "baffle_comparison.png")


def plot_clearance_sweep(sweep: BaffleClearanceSweep, outdir: Path) -> Path:
    """飄移量 vs 擋板下緣間隙 —— 標出零飄移點與撞擊邊界。"""
    p = sweep.params
    c = sweep.clearance_mm

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(c, sweep.drift_x_um, "o-", color="#1f77b4", lw=2.2, ms=3.5,
             label="Δx 縱向飄移")
    ax1.plot(c, sweep.edge_drift_y_um, "s--", color="#d2a8ff", lw=1.5, ms=3,
             label="Δy 兩端橫向飄移")
    ax1.plot(c, sweep.scatter_um, "^:", color="#f0883e", lw=1.5, ms=3,
             label="亂流散布 RMS (前緣分離)")
    ax1.axhline(0, color="k", lw=1.0)

    # 零飄移點
    sign = np.sign(sweep.drift_x_um)
    cross = np.nonzero(np.diff(sign) != 0)[0]
    if cross.size:
        i = cross[0]
        x0, x1 = c[i], c[i + 1]
        y0, y1 = sweep.drift_x_um[i], sweep.drift_x_um[i + 1]
        c_zero = x0 - y0 * (x1 - x0) / (y1 - y0)
        ax1.axvline(c_zero, color="#3fb950", lw=2.0, ls="-")
        ax1.plot([c_zero], [0], "o", color="#3fb950", ms=9, zorder=5)
        ax1.annotate(
            f"零飄移點\n{c_zero:.3f} mm",
            xy=(c_zero, 0),
            xytext=(c_zero + (c[-1] - c[0]) * 0.10, np.max(sweep.drift_x_um) * 0.45),
            fontsize=10, color="#3fb950", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#3fb950", lw=1.4),
        )

    # 撞擊禁區
    ax1.axvspan(c[0], sweep.required_clearance_mm, color="#f85149", alpha=0.13)
    ax1.axvspan(sweep.required_clearance_mm, sweep.required_clearance_mm + 0.3,
                color="#f0883e", alpha=0.13)
    ax1.text(sweep.required_clearance_mm, np.min(sweep.drift_x_um) * 0.92,
             f" 撞擊禁區 < {sweep.required_clearance_mm:.2f}mm\n"
             f" (跳動 {sweep.required_clearance_mm - p.baffle.box_height_tolerance_mm:.2f}"
             f" + 箱高公差 {p.baffle.box_height_tolerance_mm:.2f})",
             fontsize=8.5, color="#f85149", va="bottom")

    ax1.set_ylabel("飄移 (µm)")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_title(
        f"擋板下緣間隙掃描   {p.belt_speed_m_min:.0f} m/min · 箱重 {p.box_mass_kg:.0f}kg · "
        f"間隙 {p.gap_mm:.1f}mm · 圓角 R{p.baffle.edge_radius_mm:.2f}mm\n"
        f"密封不足→殘留正向飄移；密封過度→回流佔上風而變成負向飄移",
        fontsize=11)

    ax4 = ax1.twinx()
    ax4.set_ylim(np.array(ax1.get_ylim()) / (p.nozzle_pitch * 1e6))
    ax4.set_ylabel(f"換算點距 ({p.dpi:.0f} dpi)")
    ax4.grid(False)

    ax2.plot(c, sweep.sigma, "o-", color="#9467bd", lw=2.0, ms=3.5,
             label="密封有效度 σ")
    ax2.plot(c, sweep.net_flux / max(sweep.net_flux.max(), 1e-12), "s--",
             color="#8b949e", lw=1.4, ms=3, label="淨流量 (正規化)")
    ax2.set_xlabel("擋板下緣與箱面的間隙 (mm)")
    ax2.set_ylabel("σ / 正規化淨流量")
    ax2.legend(fontsize=9)
    ax2.axvspan(c[0], sweep.required_clearance_mm, color="#f85149", alpha=0.13)

    fig.tight_layout()
    return _save(fig, Path(outdir), "baffle_clearance_sweep.png")
