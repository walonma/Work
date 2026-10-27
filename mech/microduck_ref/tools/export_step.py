"""Microduck meshes -> AP242 tessellated STEP: one file per part, an assembly at a pose, and a BOM.

Usage: export_step.py <src dir with robot_allcollisions.xml, scene.xml, assets/> [pose]   (default pose STAND)
Source: pollen-robotics/microduck_rl (hardware design files CC BY-NC-SA).
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Builder
from OCP.gp import gp_Pnt, gp_Trsf
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.Poly import Poly_Triangle, Poly_Triangulation
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Controller, STEPControl_Writer
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS_Compound, TopoDS_Face
from OCP.XCAFDoc import XCAFDoc_DocumentTool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_viewer import GROUP, SHELL, ZH, cluster, read_stl  # noqa: E402
from mjcf_scan import scan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GROUP_ZH = {"actuator": "伺服", "structure": "結構件", "cosmetic": "外殼與外觀件", "electronics": "電子模組", "hardware": "軸承"}


def setup():
    STEPControl_Controller.Init_s()
    Interface_Static.SetCVal_s("write.step.schema", "AP242DIS")
    Interface_Static.SetIVal_s("write.step.tessellated", 2)      # triangulated faces, no B-rep
    Interface_Static.SetCVal_s("write.step.unit", "MM")


def tess_face(pos, idx):
    tri = Poly_Triangulation(len(pos), len(idx), False)
    for i, p in enumerate(pos, 1):
        tri.SetNode(i, gp_Pnt(*map(float, p)))
    for i, (a, b, c) in enumerate(idx, 1):
        tri.SetTriangle(i, Poly_Triangle(int(a) + 1, int(b) + 1, int(c) + 1))
    f = TopoDS_Face()
    BRep_Builder().MakeFace(f, tri)
    return f


def loc(M):
    t = gp_Trsf()
    t.SetValues(*[float(v) for v in M[:3, :4].reshape(-1)])
    return TopLoc_Location(t)


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    M = np.eye(4)
    M[:3, :3] = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    return M


def mm(M):
    M = M.copy()
    M[:3, 3] *= 1000.0
    return M


def main(src, pose_name="STAND"):
    src = Path(src)
    setup()
    bodies, geoms = scan(src / "robot_allcollisions.xml")
    order = [b["joint"]["name"] for b in bodies if b["joint"]]
    q = dict.fromkeys(order, 0.0)
    for name, qpos in re.findall(r'<key name="(\w+)" qpos="([^"]+)"', (src / "scene.xml").read_text()):
        if name == pose_name:
            q = {j: float(np.degrees(v)) for j, v in zip(order, [float(x) for x in qpos.split()][7:])}

    parts_dir = ROOT / "STEP" / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    faces, bom = {}, []
    qty = {}
    for g in geoms:
        qty[g["mesh"]] = qty.get(g["mesh"], 0) + 1
    for n, m in enumerate(sorted(qty, key=lambda k: (GROUP.get(k, "cosmetic" if k in SHELL else "structure"), k)), 1):
        pos, idx = cluster(read_stl(src / "assets" / f"{m}.stl"), 1e-4)       # weld duplicate vertices only
        f = tess_face(pos, idx)
        faces[m] = f
        w = STEPControl_Writer()
        w.Transfer(f, STEPControl_AsIs)
        assert w.Write(str(parts_dir / f"{m}.stp")) == IFSelect_RetDone
        size = pos.max(0) - pos.min(0)
        grp = GROUP.get(m, "cosmetic" if m in SHELL else "structure")
        bom.append({"item": n, "part": m, "name_zh": ZH.get(m, m), "group": GROUP_ZH[grp], "qty": qty[m],
                    "size_x_mm": round(size[0], 1), "size_y_mm": round(size[1], 1), "size_z_mm": round(size[2], 1),
                    "triangles": len(idx), "step": f"STEP/parts/{m}.stp", "stl_source": f"assets/{m}.stl"})
        print(f"{n:2d} {m}: {len(idx)} tris, qty {qty[m]}")

    # assembly: root -> one sub-assembly per body (placed at its posed world frame) -> part instances
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    proto = {}
    for m, f in faces.items():
        lab = st.AddShape(f, False)
        TDataStd_Name.Set_s(lab, TCollection_ExtendedString(m))
        proto[m] = lab
    W = {}
    for b in bodies:
        base = W[b["parent"]] if b["parent"] else np.eye(4)
        M = base @ b["local"]
        if b["joint"]:
            M = M @ rot(b["joint"]["axis"], q[b["joint"]["name"]])
        W[b["name"]] = M
    root = st.NewShape()
    TDataStd_Name.Set_s(root, TCollection_ExtendedString(f"microduck_{pose_name.lower()}"))
    for b in bodies:
        sub = st.NewShape()
        jn = b["joint"]["name"] if b["joint"] else "base"
        TDataStd_Name.Set_s(sub, TCollection_ExtendedString(f"{b['name']} [{jn}]"))
        for g in (g for g in geoms if g["body"] == b["name"]):
            st.AddComponent(sub, proto[g["mesh"]], loc(mm(g["local"])))
        st.AddComponent(root, sub, loc(mm(W[b["name"]])))
    st.UpdateAssemblies()
    w = STEPCAFControl_Writer()
    w.SetNameMode(True)
    w.Transfer(doc, STEPControl_AsIs)
    out = ROOT / "STEP" / f"microduck_assembly_{pose_name.lower()}.stp"
    assert w.Write(str(out)) == IFSelect_RetDone

    with open(ROOT / "BOM_microduck.csv", "w", newline="", encoding="utf-8-sig") as fh:
        cw = csv.DictWriter(fh, fieldnames=list(bom[0]))
        cw.writeheader()
        cw.writerows(bom)
    total = sum(r["qty"] for r in bom)
    print(f"wrote {len(bom)} part files, assembly {out.name} ({out.stat().st_size / 1e6:.1f} MB), BOM {total} pcs")


if __name__ == "__main__":
    main(*sys.argv[1:])
