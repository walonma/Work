"""Tessellate every duck part (world, zero pose) into viewer/parts.json with its joint and mass."""
import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "checks"))

from lib import parts  # noqa: E402
from lib.assembly_tree import stage_parts  # noqa: E402
from lib.dims import JOINTS  # noqa: E402
from mass import mass_of  # noqa: E402
from robot_duck import KINEMATICS  # noqa: E402

ZH = {
    "body_shell_upper": "身體上殼", "body_shell_lower": "身體下殼", "wing_l": "左翅膀", "wing_r": "右翅膀",
    "tail": "尾巴", "pelvis_frame": "骨盆骨架（電池座＋脖子伺服座）", "battery": "2S 800 mAh 鋰電池",
    "rk3566_board": "RK3566 主板（含 IMU）", "speaker": "喇叭 Ø20", "tof_sensor": "ToF 深度感測器",
    "neck_link": "脖子連桿", "neck_cover": "頸套", "pitch_bracket": "頭部俯仰支架", "roll_bracket": "頭部側歪支架",
    "head_frame": "頭部骨架", "head_shell_front": "頭殼（前）", "head_shell_back": "頭殼（後）",
    "eye_lcd_l": "左眼 0.71 吋圓形 LCD", "eye_lcd_r": "右眼 0.71 吋圓形 LCD", "eye_lens_l": "左眼透明罩",
    "eye_lens_r": "右眼透明罩", "camera": "相機模組", "mics": "麥克風 × 2", "beak_upper": "上嘴喙",
    "beak_lower": "下嘴喙（可開合）",
}
JZH = {"hip_yaw": "髖部轉向", "hip_roll": "髖部側擺", "hip_pitch": "髖部前後", "knee": "膝", "ankle": "踝",
       "neck_pitch": "脖子俯仰", "head_yaw": "頭部轉向", "head_pitch": "頭部俯仰", "head_roll": "頭部側歪", "beak": "嘴巴開合"}
SIDE = {"l": "左", "r": "右"}


def jname(j):
    if j[:2] in ("l_", "r_"):
        return SIDE[j[0]] + JZH[j[2:]]
    return JZH[j]


def label_zh(label):
    for pre, zh in (("servo_", "伺服 · "), ("horn_", "舵盤 · ")):
        if label.startswith(pre):
            return zh + jname(label[len(pre):])
    for pre, zh in (("yaw_bracket_", "髖部轉向支架"), ("hip_bracket_", "髖部支架"), ("thigh_", "大腿"), ("shin_", "小腿"),
                    ("foot_pad_", "腳底防滑墊"), ("foot_", "腳掌")):
        if label.startswith(pre):
            return SIDE[label[-1]] + zh
    return ZH.get(label, label)


def group_of(label):
    if label.startswith(("servo_", "horn_")):
        return "actuator"
    if label in ("battery", "rk3566_board", "speaker", "tof_sensor", "camera", "mics", "eye_lcd_l", "eye_lcd_r"):
        return "electronics"
    if any(k in label for k in ("shell", "wing", "tail", "beak", "cover", "lens", "foot")):
        return "cosmetic"
    return "frame"


def mesh(shape):
    verts, tris = [], []
    for s in shape.solids() or [shape]:
        v, t = s.tessellate(0.12, 0.3)
        off = len(verts)
        verts += [(p.X, p.Y, p.Z) for p in v]
        tris += [(a + off, b + off, c + off) for a, b, c in t]
    return np.asarray(verts, np.float32), np.asarray(tris, np.uint32)


def main():
    out = []
    for stage, items in stage_parts().items():
        for s in items:
            pos, idx = mesh(s)
            bb = s.bounding_box()
            col = list(tuple(s.color)) if s.color else [0.5, 0.5, 0.5, 1.0]
            col = [((c * 12.92) if c <= 0.0031308 else (1.055 * c ** (1 / 2.4) - 0.055)) for c in col[:3]] + [col[3]]
            out.append({"key": s.label, "label": label_zh(s.label), "group": group_of(s.label), "joint": stage,
                        "rgba": [round(float(c), 4) for c in col], "mass": round(mass_of(s.label, s), 1),
                        "size": [round(bb.size.X, 1), round(bb.size.Y, 1), round(bb.size.Z, 1)],
                        "pos": base64.b64encode(pos.tobytes()).decode(), "idx": base64.b64encode(idx.tobytes()).decode()})
    joints = []
    for name, (parent, kind, *_r) in JOINTS.items():
        o, a, lim = parts.joint_axis(name)
        joints.append({"name": name, "label": jname(name), "parent": parent, "origin": o, "axis": a, "limits": lim,
                       "servo": "XL330 / HD-1910 級" if kind == "leg" else "SCS0009 級微型"})
    data = {"parts": out, "joints": joints, "poses": KINEMATICS["poses"]}
    (ROOT / "viewer" / "parts.json").write_text(json.dumps(data, ensure_ascii=False))
    print(len(out), "parts,", round((ROOT / "viewer" / "parts.json").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
