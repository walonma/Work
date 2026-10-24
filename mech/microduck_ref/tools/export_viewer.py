"""Microduck MJCF + STL meshes -> viewer/data.json (decimated, instanced) for the review page.

Source: pollen-robotics/microduck_rl, src/mjlab_microduck/robot/microduck/
(hardware design files CC BY-NC-SA). Usage: export_viewer.py <dir with robot_allcollisions.xml, scene.xml, assets/>
"""
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mjcf_scan import scan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRI_BUDGET = 9000

ZH = {
    "xl330": "伺服 XL330-M288", "seeed_bearing__configuration__22x16x4": "軸承 22×16×4",
    "seeed_bearing__configuration_default": "軸承（踝）", "yaw2roll": "髖部轉向→側擺支架", "bearing_roll": "髖部側擺軸承座",
    "hip_l": "髖部支架", "upper_leg_rigidity_plate": "大腿補強板", "leg": "小腿", "neck": "脖子連桿",
    "right_shell": "身體右殼", "left_shell": "身體左殼", "banana_pcb_locker": "主板固定座", "trunk_base": "身體骨架",
    "power_support": "電池座", "np_f970": "電池（NP-F970 外形）", "upper_leg_left": "左大腿", "upper_leg_right": "右大腿",
    "sole_left": "左腳底", "sole_right": "右腳底", "foot_left": "左腳掌", "foot_right": "右腳掌", "ankle_left": "左踝支架",
    "ankle_right": "右踝支架", "neck_pitch": "頭部俯仰支架", "yaw_roll_motion": "頭部轉向／側歪機構", "lens": "相機鏡頭",
    "face_part": "臉部面板", "jaw_soft": "下嘴喙軟墊", "motor_support": "頭部馬達座", "top_head_shell": "頭殼（上）",
    "elec_rpi_robot_hat_pcb": "擴充板 PCB", "soft_mouth_top": "上嘴喙軟墊", "noenoeil": "眼框", "jaw": "下嘴喙",
    "bottom_head_shell": "頭殼（下）", "m12_lens_holder": "M12 鏡頭座", "pcb__raspberry_pi_zero_2_w": "主板（Pi Zero 2W 外形）",
    "speaker": "喇叭",
}
GROUP = {"xl330": "actuator", "seeed_bearing__configuration__22x16x4": "hardware", "seeed_bearing__configuration_default": "hardware",
         "np_f970": "electronics", "elec_rpi_robot_hat_pcb": "electronics", "pcb__raspberry_pi_zero_2_w": "electronics",
         "speaker": "electronics", "lens": "electronics", "m12_lens_holder": "electronics"}
SHELL = {"right_shell", "left_shell", "top_head_shell", "bottom_head_shell", "face_part", "noenoeil", "jaw", "jaw_soft",
         "soft_mouth_top", "sole_left", "sole_right", "foot_left", "foot_right"}
JZH = {"hip_yaw": "髖部轉向", "hip_roll": "髖部側擺", "hip_pitch": "髖部前後", "knee": "膝", "ankle": "踝",
       "neck_pitch": "脖子俯仰", "head_pitch": "頭部俯仰", "head_yaw": "頭部轉向", "head_roll": "頭部側歪"}


def read_stl(p):
    b = p.read_bytes()
    n = int.from_bytes(b[80:84], "little")
    rec = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3).astype(np.float64) * 1000.0      # metres -> mm


def cluster(verts, cell):
    q = np.round(verts / cell).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    tri = inv.reshape(-1, 3)
    keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    tri = tri[keep]
    used, remap = np.unique(tri, return_inverse=True)
    pos = np.zeros((inv.max() + 1, 3))
    np.add.at(pos, inv, verts)
    cnt = np.bincount(inv)[:, None]
    pos = (pos / cnt)[used]
    return pos.astype(np.float32), remap.reshape(-1, 3).astype(np.uint32)


def decimate(verts):
    cell = 0.05
    while True:
        pos, idx = cluster(verts, cell)
        if len(idx) <= TRI_BUDGET or cell > 2.0:
            return pos, idx, cell
        cell *= 1.4


def b64(a):
    return base64.b64encode(a.tobytes()).decode()


def joint_label(name):
    side = "左" if name.startswith("left_") else "右" if name.startswith("right_") else ""
    return side + JZH[re.sub(r"^(left|right)_", "", name)]


def main(src):
    src = Path(src)
    bodies, geoms = scan(src / "robot_allcollisions.xml")
    meshes, stats = {}, []
    for m in sorted({g["mesh"] for g in geoms}):
        v = read_stl(src / "assets" / f"{m}.stl")
        pos, idx, cell = decimate(v)
        meshes[m] = {"pos": b64(pos), "idx": b64(idx)}
        stats.append((m, len(v) // 3, len(idx), cell))
    counts = {}
    out_geoms = []
    for g in geoms:
        counts[g["mesh"]] = counts.get(g["mesh"], 0) + 1
        L = g["local"].copy()
        L[:3, 3] *= 1000.0
        grp = GROUP.get(g["mesh"], "cosmetic" if g["mesh"] in SHELL else "structure")
        out_geoms.append({"key": f'{g["mesh"]}#{counts[g["mesh"]]}', "mesh": g["mesh"], "label": ZH.get(g["mesh"], g["mesh"]),
                          "group": grp, "body": g["body"], "local": L.T.reshape(-1).round(5).tolist(),
                          "rgba": [round(c, 3) for c in g["rgba"]]})
    out_bodies = []
    for b in bodies:
        L = b["local"].copy()
        L[:3, 3] *= 1000.0
        j = b["joint"]
        out_bodies.append({"name": b["name"], "parent": b["parent"], "local": L.T.reshape(-1).round(6).tolist(),
                           "joint": None if not j else {"name": j["name"], "label": joint_label(j["name"]), "axis": j["axis"],
                                                         "range": [round(np.degrees(r), 1) for r in j["range"]]}})
    order = [b["joint"]["name"] for b in bodies if b["joint"]]
    poses = {}
    scene = (src / "scene.xml").read_text()
    for name, qpos in re.findall(r'<key name="(\w+)" qpos="([^"]+)"', scene):
        q = [float(v) for v in qpos.split()][7:]
        poses[name] = {j: round(float(np.degrees(v)), 2) for j, v in zip(order, q)}
    data = {"meshes": meshes, "geoms": out_geoms, "bodies": out_bodies, "poses": poses,
            "source": "pollen-robotics/microduck_rl · src/mjlab_microduck/robot/microduck · CC BY-NC-SA"}
    out = ROOT / "viewer" / "data.json"
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    tris = sum(s[2] for s in stats)
    print(f"{len(meshes)} meshes, {len(out_geoms)} geoms, {tris} triangles after decimation, "
          f"{out.stat().st_size / 1e6:.2f} MB; poses {list(poses)}")


if __name__ == "__main__":
    main(sys.argv[1])
