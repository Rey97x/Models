#!/usr/bin/env python3
"""
Skin weights by geodesic voxel binding.

  python3 tools/rig3d/skin_weights.py [--skeleton character3d/rig/skeleton.json] [--out character3d/rig/weights.npz]

Method
  1. weld UV-seam duplicates (identical positions get identical weights)
  2. voxelise the solid (surface sampling + outside flood fill), resolution H
  3. every joint owns one or more source segments ("bone" + helper bars that claim thick body parts: pelvis bar,
     rib bars, heel / sole segments).  Its sources are all solid voxels inside a capsule around those segments;
     the capsule radius is half the local body thickness, so thick and thin parts compete fairly
  4. geodesic distance from the sources THROUGH the solid (Dijkstra, 26-connectivity): separate limbs never
     influence each other because the path has to go around through the body
  5. weights = exp(-(d - dmin) / SIGMA), left/right exclusion, Laplacian smoothing over the surface,
     at most 4 influences, renormalised
"""
import argparse, os, sys, time
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from glb import GLB
from skeleton_def import JOINT_NAMES, load_skeleton

H = 0.004            # voxel size
SIGMA = 0.011        # blend falloff
CLOTH_RADIUS = 0.014 # capsule radius for ribbon / sash bones
CLOTH_GAIN = 1.8     # cloth distances are scaled so their influence dies quickly outside the cloth
SMOOTH_ITERS, SMOOTH_LAMBDA = 14, 0.5
CLOTH = ("Ribbon", "Sash", "Cape")
THIN_SHEET = 0.014   # a vertex is on a thin sheet if nothing within 0.012 of it is deeper than this inside the solid
CAPE_REACH = 0.22    # ... and it belongs to the cape if it is this close (through the solid) to the cape joints
CAPE_BAR = 0.07      # half width of the helper bars that spread the cape joints across the sheet


def lerp(a, b, t):
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


def source_segments(pts):
    """{joint: [(a, b), ...]} segments whose surrounding volume belongs to the joint."""
    P = {k: np.asarray(v, float) for k, v in pts.items()}
    seg = {}
    armx = abs(P["LeftArm"][0])
    hipx = abs(P["LeftUpLeg"][0])

    def bar(center, half, axis=0):
        d = np.zeros(3); d[axis] = half
        return (center - d, center + d)

    # torso: spine segment + a lateral bar ("ribs") so the trunk volume is claimed by the spine, not by the arms
    seg["Hips"] = [(P["Hips"], lerp(P["Hips"], P["Spine"], 0.9)),
                   bar(np.array([0, P["LeftUpLeg"][1], P["Hips"][2]]), hipx * 1.35),
                   bar(lerp(P["Hips"], P["Spine"], 0.5), hipx * 1.3)]
    seg["Spine"] = [(P["Spine"], P["Spine1"]), bar(lerp(P["Spine"], P["Spine1"], 0.5), armx * 0.55)]
    seg["Spine1"] = [(P["Spine1"], P["Spine2"]), bar(lerp(P["Spine1"], P["Spine2"], 0.5), armx * 0.68)]
    seg["Spine2"] = [(P["Spine2"], P["Neck"]), bar(lerp(P["Spine2"], P["Neck"], 0.3), armx * 0.72),
                     bar(lerp(P["Spine2"], P["Neck"], 0.3), 0.045, axis=2)]
    seg["Neck"] = [(P["Neck"], lerp(P["Neck"], P["Head"], 0.9))]
    hc = lerp(P["Head"], P["Head_End"], 0.5)
    seg["Head"] = [(P["Head"], P["Head_End"]), bar(hc, 0.035), bar(hc, 0.04, axis=2)]
    for s in ("Left", "Right"):
        seg[s + "Shoulder"] = [(P[s + "Shoulder"], lerp(P[s + "Shoulder"], P[s + "Arm"], 0.8))]
        seg[s + "Arm"] = [(P[s + "Arm"], P[s + "ForeArm"])]
        seg[s + "ForeArm"] = [(P[s + "ForeArm"], P[s + "Hand"])]
        seg[s + "Hand"] = [(P[s + "Hand"], P[s + "Hand_End"])]
        seg[s + "UpLeg"] = [(lerp(P[s + "UpLeg"], P[s + "Leg"], 0.18), P[s + "Leg"])]
        seg[s + "Leg"] = [(P[s + "Leg"], lerp(P[s + "Leg"], P[s + "Foot"], 0.88))]
        heel = P[s + "Heel"] + [0, 0.014, 0.008]                       # inside the heel pad
        ball = P[s + "ToeBase"]
        seg[s + "Foot"] = [(P[s + "Foot"], lerp(P[s + "Foot"], ball, 0.85)), (P[s + "Foot"], heel),
                           (heel, lerp(heel, ball, 0.8))]
        seg[s + "ToeBase"] = [(lerp(ball, P[s + "Toe_End"], 0.25), P[s + "Toe_End"])]
        if s + "Ribbon1" in P:
            seg[s + "Ribbon1"] = [(P[s + "Ribbon1"], P[s + "Ribbon2"])]
            seg[s + "Ribbon2"] = [(P[s + "Ribbon2"], P[s + "Ribbon_End"])]
    if "Cape1" in P:
        for a, b, fr in (("Cape1", "Cape2", (0.4, 0.85)), ("Cape2", "Cape_End", (0.3, 0.7, 0.95))):
            seg[a] = [(lerp(P[a], P[b], 0.15 if a == "Cape1" else 0.0), P[b])] + [bar(lerp(P[a], P[b], f), CAPE_BAR) for f in fr]
    if "Sash1" in P:
        seg["Sash1"] = [(P["Sash1"], P["Sash2"])]
        seg["Sash2"] = [(P["Sash2"], P["Sash_End"])]
    return seg


def voxelize(P, T, h, pad=3):
    lo = P.min(0) - pad * h
    dims = np.ceil((P.max(0) - lo) / h).astype(int) + pad
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    maxe = np.maximum.reduce([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1), np.linalg.norm(A - C, axis=1)])
    n = np.clip(np.ceil(maxe / (0.5 * h)).astype(int), 1, 80)
    surf = np.zeros(dims, bool)
    for k in np.unique(n):
        m = n == k
        ii, jj = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = ii + jj <= k
        u, v = ii[keep] / k, jj[keep] / k
        pts = (A[m][:, None, :] + (B[m] - A[m])[:, None, :] * u[None, :, None]
               + (C[m] - A[m])[:, None, :] * v[None, :, None]).reshape(-1, 3)
        idx = np.floor((pts - lo) / h).astype(int)
        surf[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    lab, _ = ndi.label(~surf)                      # 6-connected background
    solid = lab != lab[0, 0, 0]
    return solid, lo


def build_graph(solid):
    ids = -np.ones(solid.shape, np.int64)
    ids[solid] = np.arange(solid.sum())
    rows, cols, vals = [], [], []
    dims = solid.shape
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                off = (dx, dy, dz)
                sa = tuple(slice(max(0, -o), n - max(0, o)) for o, n in zip(off, dims))
                sb = tuple(slice(max(0, o), n - max(0, -o)) for o, n in zip(off, dims))
                a, b = ids[sa], ids[sb]
                m = (a >= 0) & (b >= 0)
                rows.append(a[m]); cols.append(b[m])
                vals.append(np.full(m.sum(), H * np.sqrt(dx * dx + dy * dy + dz * dz), np.float32))
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    n = int(solid.sum())
    g = csr_matrix((np.concatenate([v, v]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(n, n))
    return g, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "character3d/src/desert_assassin_tripo.glb"))
    ap.add_argument("--skeleton", default=os.path.join(ROOT, "character3d/rig/skeleton.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "character3d/rig/weights.npz"))
    a = ap.parse_args()
    t0 = time.time()

    md = GLB.load(a.src).mesh_data()
    P_all, T_all = md["positions"].astype(np.float64), md["triangles"].astype(np.int64)
    key = np.round(P_all * 1e5).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    P = P_all[first]                               # welded positions
    T = inv[T_all]
    print(f"vertices {len(P_all)} (welded {len(P)}), triangles {len(T)}")

    joints, ends, lms = load_skeleton(a.skeleton)
    pts = {j["name"]: j["head"] for j in joints}; pts.update(ends); pts.update(lms)
    segs = source_segments(pts)

    solid, lo = voxelize(P, T, H)
    g, ids = build_graph(solid)
    centers = (np.argwhere(solid) + 0.5) * H + lo
    tree = cKDTree(centers)
    edt = ndi.distance_transform_edt(solid)[solid] * H        # inside distance to the surface, per node
    print(f"voxels {solid.shape} solid {len(centers)}  ({time.time() - t0:.1f}s)")

    vnode = ids[tuple(np.floor((P - lo) / H).astype(int).T)]
    miss = vnode < 0
    if miss.any():
        vnode[miss] = tree.query(P[miss])[1]

    names = [j["name"] for j in joints]
    Draw = {}                                      # un-scaled distances of the cloth joints
    D = np.full((len(P), len(names)), np.inf, np.float32)
    for k, n in enumerate(names):
        if n == "Root":
            continue
        samples = np.concatenate([lerp(s[0], s[1], np.linspace(0, 1, max(2, int(np.linalg.norm(np.subtract(s[1], s[0])) / (0.5 * H)) + 1))[:, None])
                                  for s in segs[n]])
        near = tree.query(samples)[1]
        cloth = any(c in n for c in CLOTH)
        if cloth:
            rad = CLOTH_RADIUS
        else:
            rad = float(np.clip(0.5 * np.median(edt[near]), H, 0.032))
        src = np.unique(np.concatenate([np.asarray(b, np.int64) for b in tree.query_ball_point(samples, rad)] + [near]))
        d = dijkstra(g, directed=False, indices=src, min_only=True)
        dv = d[vnode].astype(np.float32)
        D[:, k] = dv * (CLOTH_GAIN if cloth else 1.0)
        if cloth:
            Draw[n] = dv
        print(f"  {n:<14} capsule r={rad:.3f}  sources {len(src):>6}")

    # a vertex on one side of the body never follows the other side's limbs
    x = P[:, 0]
    for k, n in enumerate(names):
        if n.startswith("Left"):
            D[x < -0.012, k] = np.inf
        elif n.startswith("Right"):
            D[x > 0.012, k] = np.inf
    W = np.exp(-(D - D.min(1, keepdims=True)) / SIGMA).astype(np.float64)
    W[~np.isfinite(D)] = 0
    W /= W.sum(1, keepdims=True)

    # A cape is a free-hanging thin sheet whose edges often lie against (and are fused with) hips and thighs.  Every
    # vertex of the free sheet must follow the cape joints only, otherwise the legs drag its edges along.
    cape = [k for k, n in enumerate(names) if n.startswith("Cape")]
    if cape:
        thick = ndi.maximum_filter(ndi.distance_transform_edt(solid) * H, size=2 * int(round(0.012 / H)) + 1)
        vi = np.clip(np.floor((P - lo) / H).astype(int), 0, np.array(solid.shape) - 1)
        thin = thick[vi[:, 0], vi[:, 1], vi[:, 2]] < THIN_SHEET
        dc = np.minimum.reduce([Draw[names[k]] for k in cape])
        other = [v for n, v in Draw.items() if not n.startswith("Cape")]
        d_other = np.minimum.reduce(other) if other else np.full(len(P), np.inf)
        free = thin & (dc < CAPE_REACH) & (dc <= d_other)
        Wc = np.zeros_like(W[free])
        Wc[:, cape] = np.exp(-(np.stack([Draw[names[k]][free] for k in cape], 1) - dc[free][:, None]) / SIGMA)
        W[free] = Wc / Wc.sum(1, keepdims=True)
        print(f"  cape: {int(free.sum())} free-sheet vertices follow the cape joints only")

    # Laplacian smoothing over the welded surface
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    A = coo_matrix((np.ones(len(e) * 2), (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))),
                   shape=(len(P), len(P))).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    for _ in range(SMOOTH_ITERS):
        W = (1 - SMOOTH_LAMBDA) * W + SMOOTH_LAMBDA * (A @ W) / deg[:, None]
    for k, n in enumerate(names):                  # re-apply the side exclusion after smoothing
        if n.startswith("Left"):
            W[x < -0.012, k] = 0
        elif n.startswith("Right"):
            W[x > 0.012, k] = 0

    order = np.argsort(-W, 1)[:, :4]
    w = np.take_along_axis(W, order, 1)
    w[w < 0.01] = 0
    w /= w.sum(1, keepdims=True)
    order = np.where(w > 0, order, 0)

    np.savez_compressed(a.out, joint_names=np.array(names), joint_idx=order[inv].astype(np.uint16), weights=w[inv].astype(np.float32))
    dom = order[:, 0]
    print("dominant-joint vertex counts:", {names[k]: int((dom == k).sum()) for k in range(len(names)) if (dom == k).any()})
    print(f"influences per vertex: {np.bincount((w > 0).sum(1), minlength=5)[1:].tolist()} (1..4)")
    print(f"wrote {a.out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
