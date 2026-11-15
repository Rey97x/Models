#!/usr/bin/env python3
"""
Overlay a candidate skeleton on the mesh to check joint placement.

  python3 tools/rig3d/preview_skeleton.py points.json out.png [--views front,left,back,top] [--zoom 1] [--center x,y,z]
                                           [--mode flat|pbr|wire] [--glb character3d/src/desert_assassin_tripo.glb]

points.json : {"points": {"Hips": [x,y,z], ...}}  or directly {"Hips": [x,y,z], ...}
Names not present are skipped (their children attach to the nearest present ancestor).
End sites / landmarks are drawn as extra leaf joints.  Blue lines = bones, red dots = joints (drawn through the mesh).
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from glb import GLB
from skeleton_def import HIERARCHY, END_SITES, LANDMARKS, PARENT

ap = argparse.ArgumentParser()
ap.add_argument("points"); ap.add_argument("out")
ap.add_argument("--glb", default=os.path.join(HERE, "..", "..", "character3d", "src", "desert_assassin_tripo.glb"))
ap.add_argument("--views", default="front,left,back,top"); ap.add_argument("--zoom", default="1")
ap.add_argument("--center"); ap.add_argument("--mode", default="flat"); ap.add_argument("--cell", default="520x680")
a = ap.parse_args()
d = json.load(open(a.points)); pts = d.get("points", d)
par = dict(PARENT); par.update(END_SITES)
par.update({"LeftHeel": "LeftFoot", "RightHeel": "RightFoot"})
order = [n for n, _ in HIERARCHY] + list(END_SITES) + LANDMARKS
joints = []
have = set()
for n in order:
    if n not in pts: continue
    p = par.get(n)
    while p is not None and p not in have: p = par.get(p)
    joints.append({"name": n, "parent": p, "head": pts[n]}); have.add(n)
if not joints or joints[0]["parent"] is not None or sum(j["parent"] is None for j in joints) > 1:
    joints = [{"name": "_origin", "parent": None, "head": [0, 0, 0]}] + [dict(j, parent=j["parent"] or "_origin") for j in joints]
g = GLB.load(a.glb)
n = len(g.mesh_data()["positions"])
ji = np.zeros((n, 4), int); w = np.zeros((n, 4)); w[:, 0] = 1
g.add_skin(joints, ji, w)
tmp = tempfile.mkdtemp(prefix="skelprev_"); f = os.path.join(tmp, "prev.glb"); g.save(f)
cmd = [sys.executable, os.path.join(HERE, "render.py"), f, a.out, "--views", a.views, "--skeleton", "--mode", a.mode, "--zoom", a.zoom, "--cell", a.cell]
if a.center: cmd += ["--center", a.center]
sys.exit(subprocess.run(cmd).returncode)
