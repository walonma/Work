"""指令列入口。

範例::

    python -m sim.cli --scenario baseline
    python -m sim.cli --scenario cycle --speed 350 --spacing 30
    python -m sim.cli --scenario startstop
    python -m sim.cli --sweep speed
    python -m sim.cli --scenario all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .belt import BeltKinematics
from .config import SimParams
from .simulate import (
    fire_burst,
    steady_fire_time,
    sweep_over_cycle,
    sweep_parameter,
    x_resolution_report,
)
from .windfield import WindField

SWEEP_FIELDS = {
    "speed": ("belt_speed_m_min", np.linspace(0.0, 350.0, 15)),
    "volume": ("droplet_pl", np.linspace(10.0, 50.0, 12)),
    "gap": ("gap_mm", np.linspace(3.0, 5.0, 11)),
    "spacing": ("box_spacing_cm", np.linspace(30.0, 50.0, 11)),
    "mass": ("box_mass_kg", np.linspace(5.0, 50.0, 10)),
}


def build_params(args) -> SimParams:
    return SimParams(
        belt_speed_m_min=args.speed,
        box_mass_kg=args.mass,
        gap_mm=args.gap,
        droplet_pl=args.volume,
        box_spacing_cm=args.spacing,
        print_height_in=args.print_height,
        plate_x_mm=args.plate_x,
        eject_velocity=args.eject,
        fire_frequency_hz=args.fire_hz,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="產線風速風場模擬觀測器 — 熱發泡噴碼墨滴飄移模擬",
    )
    ap.add_argument("--scenario", choices=["baseline", "cycle", "startstop", "all"],
                    default=None, help="要執行的場景")
    ap.add_argument("--sweep", choices=sorted(SWEEP_FIELDS), default=None,
                    help="參數掃描")
    ap.add_argument("--outdir", type=Path, default=Path("output"), help="圖表輸出目錄")

    ap.add_argument("--speed", type=float, default=350.0, help="產線速度 m/min (0~350)")
    ap.add_argument("--mass", type=float, default=20.0, help="箱重 kg (5~50)")
    ap.add_argument("--gap", type=float, default=4.0, help="噴嘴間隙 mm (3~5)")
    ap.add_argument("--volume", type=float, default=30.0, help="墨滴體積 pL (10~50)")
    ap.add_argument("--spacing", type=float, default=40.0, help="箱間距 cm (30~50)")
    ap.add_argument("--print-height", type=float, default=1.0,
                    help="噴印帶高度 (吋). 1.0=300dpi, 0.5=600dpi")
    ap.add_argument("--plate-x", type=float, default=15.0, help="噴嘴板 X 向寬度 mm")
    ap.add_argument("--eject", type=float, default=8.0, help="噴發初速 m/s")
    ap.add_argument("--fire-hz", type=float, default=15000.0, help="噴發頻率 Hz")
    ap.add_argument("--cycle-samples", type=int, default=48, help="循環掃描的取樣數")

    args = ap.parse_args(argv)
    if args.scenario is None and args.sweep is None:
        args.scenario = "baseline"

    params = build_params(args)
    outdir: Path = args.outdir
    print(params.summary())
    print()

    written: list[Path] = []

    # 延後匯入, 讓純數值使用者不必載入 matplotlib
    from . import plots

    if args.scenario in ("baseline", "all"):
        print(">> 基準案例 — 產出 5-1 / 5-2 圖表")
        belt = BeltKinematics(params)
        wind = WindField(params, belt)
        t = steady_fire_time(params)
        burst = fire_burst(params, t_fire=t, belt=belt, wind=wind, record_frames=80)
        print(burst.summary())
        print()
        print(x_resolution_report(params))
        print()
        written += plots.render_baseline(params, outdir, t_fire=t)

    if args.scenario in ("cycle", "all"):
        print(f">> 循環掃描 — {args.cycle_samples} 個相位取樣")
        scan = sweep_over_cycle(params, n_samples=args.cycle_samples)
        i_peak = int(np.argmax(scan.max_drift_um))
        print(f"   飄移峰值出現在 t = {scan.times[i_peak]*1000:.1f} ms"
              f" (相位 {scan.phase[i_peak]*100:.1f} cm, 拘束因子 {scan.confinement[i_peak]:.2f})")
        print(f"   峰值 {scan.max_drift_um[i_peak]:.1f} µm"
              f" = {scan.max_drift_um[i_peak]/(params.nozzle_pitch*1e6):.2f} 個點距")
        print(f"   箱下穩態約 {np.median(scan.mean_drift_um[scan.confinement > 0.9]):.1f} µm"
              if np.any(scan.confinement > 0.9) else "")
        print()
        written.append(plots.plot_cycle_scan(scan, outdir))

    if args.scenario in ("startstop", "all"):
        print(">> 啟停過渡 — 箱重的機械耦合")
        belt = BeltKinematics(params)
        written.append(plots.plot_startstop(params, belt, outdir))

    if args.sweep:
        field, values = SWEEP_FIELDS[args.sweep]
        print(f">> 參數掃描: {field}  ({len(values)} 點)")
        sw = sweep_parameter(params, field, values)
        for v, d in zip(sw.values, sw.mean_drift_um):
            print(f"   {v:8.2f}  ->  平均飄移 {d:7.2f} µm"
                  f"  ({d/(params.nozzle_pitch*1e6):.2f} 個點距)")
        print()
        written.append(plots.plot_sweep(sw, params, outdir))

    if written:
        print("已輸出:")
        for p in written:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
