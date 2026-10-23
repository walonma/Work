"""Tessellate every part into viewer/parts.json for the HTML review page.

Fixed parts are written in world coordinates; moving parts in their link
frame (see lib.parts), so the page can pose the arm with the same chain as
lib.dims.chain().
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cadgen import build123d as bd  # noqa: E402

from lib import parts  # noqa: E402
from lib import dims as D  # noqa: E402

STEEL, PETG, FILL = 7.85e-3, 1.27e-3, 0.45
TOL = 0.08

# key, label, group, frame, color, material, density, factory
PARTS = [
    ("base_plate", "A1 底座（含風扇夾板）", "sheet", "world", "#3a3f46", "SPCC 1.6t 黑色粉體烤漆", STEEL, parts.base_plate),
    ("lower_arm", "A2 下臂", "sheet", "lower", "#a7aeb6", "SPCC 1.6t 銀色粉體烤漆", STEEL, parts.lower_arm),
    ("upper_arm", "A3 上臂", "sheet", "upper", "#b9c0c7", "SPCC 1.6t 銀色粉體烤漆", STEEL, parts.upper_arm),
    ("head", "B1 手機頭座", "printed", "head", "#e8862a", "PETG（A1 列印）", PETG * FILL, parts.head),
    ("cradle_body", "B2 手機托架本體", "printed", "cradle", "#f09a3e", "PETG（A1 列印）", PETG * FILL, parts.cradle_body),
    ("jaw_right", "B3 右夾爪", "printed", "jaw_right", "#d9772a", "PETG（A1 列印）", PETG * FILL, lambda: parts.jaw(+1, 0.0)),
    ("jaw_left", "B4 左夾爪", "printed", "jaw_left", "#d9772a", "PETG（A1 列印）", PETG * FILL, lambda: parts.jaw(-1, 0.0)),
    ("cable_reel", "B5 捲線器＋USB-C 停靠口＋充電頭槽", "printed", "world", "#f0a050", "PETG（A1 列印）", PETG * FILL, parts.cable_reel),
    ("shoulder_pivot", "C1 肩部摩擦轉軸", "hardware", "lower", "#222428", "M5 低頭對鎖螺絲＋碟形彈簧", None,
     lambda: parts.pivot_bolt(D.LOW_W + 2 * D.T + 0.4)),
    ("elbow_pivot", "C2 肘部摩擦轉軸＋墊片", "hardware", "upper", "#222428", "M5 低頭對鎖螺絲＋POM 墊片", None,
     lambda: parts.pivot_bolt(D.LOW_W, spacer=(D.UP_W / 2, D.LOW_W / 2 - D.T))),
    ("head_pivot", "C3 頭部摩擦轉軸", "hardware", "head", "#222428", "M5 低頭對鎖螺絲＋碟形彈簧", None,
     lambda: parts.pivot_bolt(D.UP_W)),
    ("silicone_pad", "C4 矽膠防滑條", "hardware", "world", "#55575b", "2 mm 矽膠，模切背膠", None, parts.silicone_pad),
    ("phone_ref", "手機（參考 76 × 160 × 9）", "ref", "cradle", "#7fa6c9", "參考外形", None, parts.phone_ref),
    ("fan_ref", "DIKE DUF201 風扇（參考）", "ref", "world", "#9fb8cc", "參考外形", None, parts.fan_ref),
]


def mesh(shape):
    verts, tris = [], []
    for s in shape.solids() or [shape]:
        v, t = s.tessellate(TOL, 0.3)
        off = len(verts)
        verts += [(p.X, p.Y, p.Z) for p in v]
        tris += [(a + off, b + off, c + off) for a, b, c in t]
    pos = np.asarray(verts, dtype=np.float32)
    idx = np.asarray(tris, dtype=np.uint32)
    return pos, idx


def b64(a):
    return base64.b64encode(a.tobytes()).decode()


def main():
    out = []
    for key, label, group, frame, color, mat, rho, make in PARTS:
        shape = make()
        pos, idx = mesh(shape)
        bb = shape.bounding_box()
        out.append({
            "key": key, "label": label, "group": group, "frame": frame, "color": color, "material": mat,
            "mass": round(shape.volume * rho) if rho else None,
            "size": [round(bb.size.X, 1), round(bb.size.Y, 1), round(bb.size.Z, 1)],
            "pos": b64(pos), "idx": b64(idx), "tris": int(len(idx)),
        })
        print(f"{key}: {len(idx)} triangles")
    data = {
        "dims": {"H1": D.H1, "L1": D.L1, "L2": D.L2, "HEAD_AXIS_OFF": D.HEAD_AXIS_OFF,
                 "HEAD_PLATE_TOP": D.HEAD_PLATE_Z[1], "PHONE_C": [0.0, D.FACE_Y - 4.5, D.CRADLE_ZB + 4.0 + 80.0],
                 "BASE_D": D.BASE_D, "JAW_MIN": D.JAW_MIN, "JAW_MAX": D.JAW_MAX,
                 "SHOULDER": D.SHOULDER, "ELBOW": D.ELBOW, "HEAD": D.HEAD,
                 "SHOULDER_RANGE": D.SHOULDER_RANGE, "ELBOW_RANGE": D.ELBOW_RANGE, "HEAD_RANGE": D.HEAD_RANGE},
        "parts": out,
    }
    (ROOT / "viewer" / "parts.json").write_text(json.dumps(data, ensure_ascii=False))
    print("wrote viewer/parts.json", round((ROOT / "viewer" / "parts.json").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
