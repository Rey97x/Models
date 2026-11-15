#!/usr/bin/env python3
"""
First-guess skeleton for a new humanoid mesh (A- or T-pose, facing +Z, feet on y = 0, about 1 unit tall like Tripo
exports).  The verified skeleton of the first character (template_skeleton.json) is scaled to the new height and then
refitted to the mesh:
  * depth (z): the body core (voxels well inside the solid, so capes / skirts / thin cloth are ignored) is compared
    with the same measurement on the template character and the torso, hips and shoulders are shifted accordingly
  * shoulder width follows the width of the chest core; neck / head heights follow the top of the head core
    (helmet wings, crests and other thin ornaments do not count)
  * knees and ankles are centred in the leg cores; heel / ball / toe follow the shoe extent
  * the arm chain follows the measured arm centre line from the shoulder to the fingertip (wrist = thinnest section)

  python3 tools/rig3d/auto_skeleton.py <src.glb> <out skeleton.json>

The result is a starting point, not a verdict: ALWAYS overlay it (preview_skeleton.py, slice_view.py) and correct what
is off.  Cloth chains are not guessed: add `Left/RightRibbon1-2 + Ribbon_End` (strips at the wrists),
`Sash1-2 + Sash_End` (cloth hanging from the belt; on the centre line it acts as a tabard) and
`Cape1-2 + Cape_End` (cape down the back) by hand when the character has them.
"""
import json, os, re, sys
import numpy as np
from scipy import ndimage as ndi
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from glb import GLB
from skin_weights import voxelize

H_VOX = 0.003
CORE = 0.010          # a voxel this deep inside the solid belongs to the body core


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
    ref = tpl["ref"]
    s = height / tpl["height"]
    pts = {k: np.array(v, float) * s for k, v in tpl["points"].items()}

    h = H_VOX * s
    solid, lo = voxelize(P, T, h)
    edt = ndi.distance_transform_edt(solid) * h
    core = edt > CORE * s
    C = (np.argwhere(solid) + 0.5) * h + lo                     # all solid voxel centres
    Cc = (np.argwhere(core) + 0.5) * h + lo                     # body core
    opened = ndi.binary_opening(solid, iterations=max(1, int(round(0.006 * s / h))))
    Co = (np.argwhere(opened) + 0.5) * h + lo                   # solid without thin strips (keeps hands)

    def blobs(y):
        """core blobs of the horizontal slice at height y: list of (cx, cz, xmin, xmax, n), largest first."""
        j = int((y - lo[1]) / h)
        lab, n = ndi.label(core[:, j, :])
        res = []
        for k in range(1, n + 1):
            idx = np.argwhere(lab == k)
            if len(idx) >= 25:
                c = (idx.mean(0) + 0.5) * h + lo[[0, 2]]
                res.append((c[0], c[1], idx[:, 0].min() * h + lo[0], (idx[:, 0].max() + 1) * h + lo[0], len(idx)))
        return sorted(res, key=lambda b: -b[4])

    def central(y):
        bl = [b for b in blobs(y) if b[2] < 0 < b[3]]
        return bl[0] if bl else None

    # ---- torso depth: shift relative to the template character's own core centroids
    dz = {}
    for key, zref in ref["core_z"].items():
        b = central(float(key) * s)
        dz[key] = (b[1] - zref * s) if b else 0.0
    groups = {"0.507": ["Hips", "LeftUpLeg", "RightUpLeg"], "0.584": ["Spine"], "0.656": ["Spine1"],
              "0.733": ["Spine2", "Neck", "Head", "Head_End", "LeftShoulder", "RightShoulder", "LeftArm", "RightArm"]}
    for key, names in groups.items():
        for n in names:
            pts[n][2] += dz[key]
    # ---- shoulder width from the chest core
    b = central(0.733 * s)
    if b:
        k = float(np.clip(max(-b[2], b[3]) / (ref["halfwidth_733"] * s), 0.8, 1.2))
        for n in ("LeftShoulder", "RightShoulder", "LeftArm", "RightArm"):
            pts[n][0] *= k
    # ---- neck / head heights from the top of the head core (thin ornaments excluded)
    crown = (np.nonzero(core.any(axis=(0, 2)))[0].max() + 1) * h + lo[1]
    ya = pts["LeftArm"][1]
    k = float(np.clip((crown - ya) / (ref["crown"] * s - ya), 0.7, 1.1))
    for n in ("Neck", "Head"):
        pts[n][1] = ya + (pts[n][1] - ya) * k
    pts["Head_End"][1] = height

    for S, sg in (("Left", 1), ("Right", -1)):
        # ---- leg: centre of the leg core at knee / ankle height
        for jn in (S + "Leg", S + "Foot"):
            cand = [b for b in blobs(pts[jn][1]) if b[0] * sg > 0.01 * s and abs(b[0]) < 0.2 * s]
            if cand:
                bb = min(cand, key=lambda b: abs(b[0] - pts[jn][0]))
                pts[jn][[0, 2]] = [bb[0], bb[1]]
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
        # ---- arm: fingertip = widest point of the body on this side, then centre line shoulder -> tip
        side = Co[(Co[:, 0] * sg > abs(pts[S + "Arm"][0]) * 0.9) & (Co[:, 1] > 0.3 * height)]
        if len(side) > 100:
            tip0 = side[np.argmax(side[:, 0] * sg)]
            sh = pts[S + "Arm"]
            near = C[np.linalg.norm(C - tip0, axis=1) < 0.03 * s]
            tip = near[np.argmax(np.linalg.norm(near - sh, axis=1))]
            axis = tip - sh; L = np.linalg.norm(axis); axis /= L
            fr = np.linspace(0.08, 0.97, 60)
            line, area = [], []
            for f in fr:
                c = sh + axis * L * f
                m = np.zeros(len(Co), bool)
                for _ in range(3):                                   # follow the bent arm
                    m = (np.abs((Co - c) @ axis) < 0.006 * s) & (np.linalg.norm(Co - c, axis=1) < 0.06 * s)
                    if m.sum() < 5:
                        break
                    c = Co[m].mean(0)
                line.append(c); area.append(int(m.sum()))
            line, area = np.array(line), np.array(area)
            win = (fr > 0.64) & (fr < 0.80)      # the wrist sits at ~0.72 of the shoulder-fingertip line
            iw = np.nonzero(win)[0][np.argmin(ndi.uniform_filter1d(area.astype(float), 5)[win])]   # wrist = thinnest
            t_up = np.linalg.norm(np.subtract(tpl["points"][S + "ForeArm"], tpl["points"][S + "Arm"]))
            t_fo = np.linalg.norm(np.subtract(tpl["points"][S + "Hand"], tpl["points"][S + "ForeArm"]))
            cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(np.vstack([sh, line[:iw + 1]]), axis=0), axis=1))])
            ie = int(np.argmin(np.abs(cum[1:] - cum[-1] * t_up / (t_up + t_fo))))
            pts[S + "ForeArm"], pts[S + "Hand"], pts[S + "Hand_End"] = line[ie], line[iw], tip
        else:
            notes.append(f"{S} arm not found: kept the scaled template")
    pts["Root"] = np.zeros(3)
    doc = {"status": "AUTO ESTIMATE - verify with preview_skeleton.py / slice_view.py and correct before skinning",
           "conventions": "+Y up, faces +Z, character's left = +X; identity rest rotations",
           "notes": notes, "points": {k: [round(float(c), 4) for c in v] for k, v in pts.items()}}
    txt = json.dumps(doc, indent=1)
    txt = re.sub(r"\[\s+([-\d.e]+),\s+([-\d.e]+),\s+([-\d.e]+)\s+\]", r"[\1, \2, \3]", txt)
    open(out, "w").write(txt + "\n")
    print(f"height {height:.3f}  scale {s:.3f}  head core top {crown:.3f}  torso depth shifts {dict((k, round(v, 3)) for k, v in dz.items())}  -> {out}")
    for n in notes:
        print("NOTE:", n)


if __name__ == "__main__":
    main()
