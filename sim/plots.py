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

from .belt import BeltKinematics
from .config import SimParams
from .simulate import (
    BurstResult,
    CycleScan,
    ParameterSweep,
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
