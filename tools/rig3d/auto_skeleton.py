#!/usr/bin/env python3
"""
First-guess skeleton for a new humanoid mesh (A- or T-pose, facing +Z, feet on y = 0, about 1 unit tall like Tripo
exports).  The verified skeleton of the first character (template_skeleton.json) is scaled to the new height and then
refitted to the mesh: leg joints are centred in the leg cross-sections, feet follow the shoe extent, the arm chain
follows the measured arm centre line from the shoulder to the fingertip (wrist = thinnest section).

  python3 tools/rig3d/auto_skeleton.py <src.glb> <out skeleton.json>

The result is a starting point, not a verdict: ALWAYS overlay it (preview_skeleton.py) and correct what is off,
especially shoulders, hips (hidden by clothes) and the spine depth.  Cloth chains (ribbons, sash) are not guessed:
add `Left/RightRibbon1-2 + Ribbon_End`, `Sash1-2 + Sash_End` by hand if the character has such hanging cloth.
"""
import json, os, sys
import numpy as np
from scipy import ndimage as ndi
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from glb import GLB
from skin_weights import voxelize

H_VOX = 0.004


def main():
    src, out = sys.argv[1], sys.argv[2]
    md = GLB.load(src).mesh_data()
    P, T = md["positions"].astype(float), md["triangles"].astype(np.int64)
    lo_y, height = P[:, 1].min(), P[:, 1].max() - P[:, 1].min()
    notes = []
    if abs(lo_y) > 0.01 * height:
        notes.append(f"feet are not on y=0 (min y {lo_y:.3f}): move the mesh first")
    if not 0.8 < height < 1.25:
        notes.append(f"height {height:.3f}: the animation constants assume ~1 unit; rescale the mesh to about 1 unit")
    tpl = json.load(open(os.path.join(HERE, "template_skeleton.json")))
    s = height / tpl["height"]
    pts = {k: np.array(v, float) * s for k, v in tpl["points"].items()}

    solid, lo = voxelize(P, T, H_VOX * s)
    h = H_VOX * s
    C = (np.argwhere(solid) + 0.5) * h + lo                     # voxel centres
    core = ndi.binary_opening(solid, iterations=max(1, int(round(0.006 * s / h))))   # drops thin cloth strips
    Cc = (np.argwhere(core) + 0.5) * h + lo

    def slab(cloud, y, dy=0.006):
        return cloud[np.abs(cloud[:, 1] - y) < dy * s]

    for S, sg in (("Left", 1), ("Right", -1)):
        # ---- leg: centre of the leg section at knee / ankle height (component nearest to the template position)
        for jn in (S + "Leg", S + "Foot"):
            sl = slab(C, pts[jn][1])
            sl = sl[(sl[:, 0] * sg > 0.01 * s) & (np.abs(sl[:, 0]) < 0.2 * s)]
            if len(sl) > 20:
                d = np.linalg.norm(sl[:, [0, 2]] - pts[jn][[0, 2]], axis=1)
                near = sl[d < 0.07 * s]
                if len(near) > 20:
                    pts[jn][[0, 2]] = near[:, [0, 2]].mean(0)
        k_ratio = pts[S + "Leg"][0] / (tpl["points"][S + "Leg"][0] * s)
        pts[S + "UpLeg"][0] *= np.clip(k_ratio, 0.8, 1.25)
        # ---- foot: heel, toe tip and ball from the shoe extent
        ft = C[(C[:, 1] < 0.035 * s) & (C[:, 0] * sg > 0.01 * s) & (np.abs(C[:, 0] - pts[S + "Foot"][0]) < 0.09 * s)]
        if len(ft) > 50:
            z0, z1 = ft[:, 2].min(), ft[:, 2].max()
            xat = lambda z: float(ft[np.abs(ft[:, 2] - z) < 0.012 * s][:, 0].mean())
            pts[S + "Heel"] = np.array([xat(z0 + 0.01 * s), 0.0, z0 + 0.007 * s])
            pts[S + "Toe_End"] = np.array([xat(z1 - 0.01 * s), pts[S + "ToeBase"][1], z1 - 0.002 * s])
            zb = z0 + 0.68 * (z1 - z0)
            pts[S + "ToeBase"] = np.array([xat(zb), pts[S + "ToeBase"][1], zb])
        # ---- arm: fingertip = widest point of the body core on this side, then centre line shoulder -> tip
        side = Cc[(Cc[:, 0] * sg > abs(pts[S + "Arm"][0]) * 0.9) & (Cc[:, 1] > 0.3 * height)]
        if len(side) > 100:
            tip0 = side[np.argmax(side[:, 0] * sg)]
            sh = pts[S + "Arm"]
            near = C[np.linalg.norm(C - tip0, axis=1) < 0.03 * s]
            tip = near[np.argmax(np.linalg.norm(near - sh, axis=1))]
            axis = tip - sh; L = np.linalg.norm(axis); axis /= L
            line, area = [], []
            for f in np.linspace(0.08, 0.97, 60):
                c = sh + axis * L * f
                for _ in range(3):                                   # follow the bent arm
                    m = (np.abs((Cc - c) @ axis) < 0.006 * s) & (np.linalg.norm(Cc - c, axis=1) < 0.06 * s)
                    if m.sum() < 5:
                        break
                    c = Cc[m].mean(0)
                line.append(c); area.append(int(m.sum()))
            line, area = np.array(line), np.array(area)
            fr = np.linspace(0.08, 0.97, 60)
            win = (fr > 0.64) & (fr < 0.80)      # the wrist sits at ~0.72 of the shoulder-fingertip line
            iw = np.nonzero(win)[0][np.argmin(ndi.uniform_filter1d(area.astype(float), 5)[win])]   # wrist = thinnest
            wrist = line[iw]
            t_up = np.linalg.norm(np.subtract(tpl["points"][S + "ForeArm"], tpl["points"][S + "Arm"]))
            t_fo = np.linalg.norm(np.subtract(tpl["points"][S + "Hand"], tpl["points"][S + "ForeArm"]))
            cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(np.vstack([sh, line[:iw + 1]]), axis=0), axis=1))])
            ie = int(np.argmin(np.abs(cum[1:] - cum[-1] * t_up / (t_up + t_fo))))
            pts[S + "ForeArm"], pts[S + "Hand"], pts[S + "Hand_End"] = line[ie], wrist, tip
        else:
            notes.append(f"{S} arm not found: kept the scaled template")
    pts["Head_End"][1] = height
    pts["Root"] = np.zeros(3)
    doc = {"status": "AUTO ESTIMATE - verify with preview_skeleton.py and correct before skinning",
           "conventions": "+Y up, faces +Z, character's left = +X; identity rest rotations",
           "notes": notes, "points": {k: [round(float(c), 4) for c in v] for k, v in pts.items()}}
    txt = json.dumps(doc, indent=1)
    import re
    txt = re.sub(r"\[\s+([-\d.e]+),\s+([-\d.e]+),\s+([-\d.e]+)\s+\]", r"[\1, \2, \3]", txt)
    open(out, "w").write(txt + "\n")
    print(f"height {height:.3f}  scale {s:.3f}  -> {out}")
    for n in notes:
        print("NOTE:", n)


if __name__ == "__main__":
    main()
