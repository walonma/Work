"""Read the assembly STEP back and check every part instance lands where the MJCF puts it."""
import re
import sys
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_step import mm, rot  # noqa: E402
from export_viewer import cluster, read_stl  # noqa: E402
from mjcf_scan import scan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def step_points(path):
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    r = STEPCAFControl_Reader()
    r.SetNameMode(True)
    r.ReadFile(str(path))
    r.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    roots = TDF_LabelSequence()
    st.GetFreeShapes(roots)
    shape = st.GetShape_s(roots.Value(1))
    pts = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    n = 0
    while ex.More():
        f = TopoDS.Face_s(ex.Current())
        lc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, lc)
        tr = lc.Transformation()
        nodes = [tri.Node(i).Transformed(tr) for i in range(1, tri.NbNodes() + 1)]
        pts.append(np.array([[p.X(), p.Y(), p.Z()] for p in nodes]))
        n += 1
        ex.Next()
    return pts, n


def main(src, pose="STAND"):
    src = Path(src)
    bodies, geoms = scan(src / "robot_allcollisions.xml")
    order = [b["joint"]["name"] for b in bodies if b["joint"]]
    q = dict.fromkeys(order, 0.0)
    for name, qpos in re.findall(r'<key name="(\w+)" qpos="([^"]+)"', (src / "scene.xml").read_text()):
        if name == pose:
            q = {j: float(np.degrees(v)) for j, v in zip(order, [float(x) for x in qpos.split()][7:])}
    W = {}
    for b in bodies:
        M = (W[b["parent"]] if b["parent"] else np.eye(4)) @ b["local"]
        if b["joint"]:
            M = M @ rot(b["joint"]["axis"], q[b["joint"]["name"]])
        W[b["name"]] = M
    expected = []
    for g in geoms:
        pos, _ = cluster(read_stl(src / "assets" / f"{g['mesh']}.stl"), 1e-4)
        M = mm(W[g["body"]]) @ mm(g["local"])
        P = pos @ M[:3, :3].T + M[:3, 3]
        expected.append((g["mesh"], P.mean(0), len(pos)))
    got, n = step_points(ROOT / "STEP" / f"microduck_assembly_{pose.lower()}.stp")
    allp = np.vstack(got)
    print(f"read {n} part instances (expected {len(geoms)}); envelope {np.ptp(allp, 0).round(1)} mm, "
          f"z {allp[:, 2].min():.1f}..{allp[:, 2].max():.1f}")
    cents = [p.mean(0) for p in got]
    worst = 0.0
    for mesh, c, nv in expected:
        d = min(np.linalg.norm(c - x) for x, p in zip(cents, got) if len(p) == nv)
        worst = max(worst, d)
    ok = n == len(geoms) and worst < 0.01
    print(f"worst instance centroid error {worst:.4f} mm -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
