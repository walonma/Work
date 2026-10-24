"""Parse the Microduck MJCF: body tree, joints, visual geoms (mesh, local pose, material)."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose(el):
    M = np.eye(4)
    M[:3, 3] = [float(v) for v in el.get("pos", "0 0 0").split()]
    q = [float(v) for v in el.get("quat", "1 0 0 0").split()]
    M[:3, :3] = quat_mat(np.array(q) / np.linalg.norm(q))
    return M


def scan(path):
    root = ET.parse(path).getroot()
    mats = {m.get("name"): [float(v) for v in m.get("rgba", "0.7 0.7 0.7 1").split()] for m in root.iter("material")}
    bodies, geoms = [], []

    def walk(el, parent, Mw):
        for b in el.findall("body"):
            M = Mw @ pose(b)
            j = b.find("joint")
            joint = None
            if j is not None and j.get("type", "hinge") == "hinge":
                ax = [float(v) for v in j.get("axis", "0 0 1").split()]
                rng = [float(v) for v in j.get("range", "-3.14 3.14").split()]
                joint = {"name": j.get("name"), "axis": ax, "range": rng}
            bodies.append({"name": b.get("name"), "parent": parent, "local": pose(b), "world": M, "joint": joint})
            for g in b.findall("geom"):
                if g.get("class") == "visual" and g.get("mesh"):
                    geoms.append({"body": b.get("name"), "mesh": g.get("mesh"), "local": pose(g),
                                  "rgba": mats.get(g.get("material"), [0.7, 0.7, 0.7, 1])})
            walk(b, b.get("name"), M)

    walk(root.find("worldbody"), None, np.eye(4))
    return bodies, geoms


if __name__ == "__main__":
    bodies, geoms = scan(sys.argv[1])
    for b in bodies:
        print(b["name"], "<-", b["parent"], (b["joint"] or {}).get("name"), np.round(b["world"][:3, 3] * 1000, 1))
    from collections import Counter
    print(len(geoms), "visual geoms")
    for m, n in Counter(g["mesh"] for g in geoms).most_common():
        print(f"  {n:2d} x {m}")
