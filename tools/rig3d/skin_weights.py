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
CAPE_REACH = 0.35    # ... and it belongs to the cape if it is this close (through the solid) to the cape joints
CAPE_BAR_LOW = 0.11
CAPE_ATTACH = ("Spine", "Spine1", "Spine2", "Neck", "Head", "LeftShoulder", "RightShoulder")   # where a cape stays welded
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
            hw = CAPE_BAR if a == "Cape1" else CAPE_BAR_LOW        # wider where the sheet hangs free of the body
            seg[a] = [(lerp(P[a], P[b], 0.15 if a == "Cape1" else 0.0), P[b])] + [bar(lerp(P[a], P[b], f), hw) for f in fr]
    if "Sash1" in P:
        seg["Sash1"] = [(P["Sash1"], P["Sash2"])]
        seg["Sash2"] = [(P["Sash2"], P["Sash_End"])]
    return seg


def cape_colour_mask(glb, md, first, T, seeds, n):
    """Colour distance of every welded vertex (surface-smoothed base colour) from the colour of the cape's sure
    vertices (`seeds`), in units of the cape's own colour spread: <= 1 means "cape coloured".
    None if the mesh has no base colour texture or there are too few seeds."""
    import io
    from PIL import Image
    try:
        js = glb.js
        mat = js["materials"][js["meshes"][0]["primitives"][0]["material"]]
        img = js["images"][js["textures"][mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]]
        bv = js["bufferViews"][img["bufferView"]]
        raw = bytes(glb.bin[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]])
        tex = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB").resize((1024, 1024), Image.BOX), float) / 255
        uv = md["uvs"][first]
    except (KeyError, IndexError):
        return None
    if seeds.sum() < 50:
        return None
    col = tex[np.clip((uv[:, 1] % 1.0) * 1024, 0, 1023).astype(int), np.clip((uv[:, 0] % 1.0) * 1024, 0, 1023).astype(int)]
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    A = coo_matrix((np.ones(len(e) * 2), (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    for _ in range(5):                               # average over the neighbourhood: texture noise, dark crevices
        col = 0.5 * col + 0.5 * (A @ col) / deg[:, None]
    sc = col[seeds]
    lum = sc.mean(1)
    cen = np.array([sc[lum <= np.percentile(lum, 33)].mean(0), sc.mean(0), sc[lum >= np.percentile(lum, 67)].mean(0)])
    for _ in range(10):                              # 3-means on the seed colours
        lab = np.argmin(np.linalg.norm(sc[:, None] - cen[None], axis=2), 1)
        cen = np.array([sc[lab == k].mean(0) if (lab == k).any() else cen[k] for k in range(3)])
    tol = np.array([max(0.045, 1.6 * np.percentile(np.linalg.norm(sc[lab == k] - cen[k], axis=1), 97)) if (lab == k).any() else 0.045
                    for k in range(3)])
    return (np.linalg.norm(col[:, None] - cen[None], axis=2) / tol[None]).min(1)      # <= 1: cape colour


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

    src_glb = GLB.load(a.src)
    md = src_glb.mesh_data()
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

    # ---------------------------------------------------------------------------------------------- cape
    # A cape is a free-hanging thin sheet.  Its edges usually lie against hips, thighs or armour pieces and are fused
    # with them in a single-shell mesh.  Two things are needed for it to move like cloth:
    #   1. every vertex of the free sheet follows the cape joints only (otherwise the legs drag its edges along);
    #      thin ARMOUR pieces near the cape (knee wings, fins) are told apart by their texture colour
    #   2. where the sheet is fused with legs / hips / arms the mesh is CUT (border vertices are duplicated), so the
    #      two sides slide past each other instead of stretching a membrane between knee and cape.
    #      The attachment along spine / shoulders / neck stays welded and blends smoothly.
    cape = [k for k, n in enumerate(names) if n.startswith("Cape")]
    n_src = len(P_all)
    T_out, extra_src, extra_wid = T_all, [], []
    if cape:
        thick = ndi.maximum_filter(ndi.distance_transform_edt(solid) * H, size=2 * int(round(0.012 / H)) + 1)
        vi = np.clip(np.floor((P - lo) / H).astype(int), 0, np.array(solid.shape) - 1)
        thin = thick[vi[:, 0], vi[:, 1], vi[:, 2]] < THIN_SHEET
        dc = np.minimum.reduce([Draw[names[k]] for k in cape])
        other = [v for n, v in Draw.items() if not n.startswith("Cape")]
        d_other = np.minimum.reduce(other) if other else np.full(len(P), np.inf)
        cand = thin & (dc < CAPE_REACH) & (dc <= d_other)
        score = cape_colour_mask(src_glb, md, first, T, cand & (dc < 0.035), len(P))
        col_ok = None if score is None else score <= 1.0
        near_ok = np.ones(len(P), bool) if score is None else score <= 1.8
        free = cand & col_ok if col_ok is not None else cand
        # keep the sheet coherent: drop speckles (dark crevices of thin armour), fill pin-holes, and keep only the
        # connected pieces that contain sure cape vertices
        e0 = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
        A0 = coo_matrix((np.ones(len(e0) * 2), (np.concatenate([e0[:, 0], e0[:, 1]]), np.concatenate([e0[:, 1], e0[:, 0]]))),
                        shape=(len(P), len(P))).tocsr()
        A0.data[:] = 1.0
        deg0 = np.maximum(np.asarray(A0.sum(1)).ravel(), 1)
        frac = (A0 @ free.astype(float)) / deg0
        free = (free & (frac >= 0.5)) | (cand & (frac >= 0.75))
        # frayed, slightly off-colour or folded (thicker) bits next to the sheet join it
        soft = (dc < CAPE_REACH) & (dc <= d_other) & near_ok & (thin | (col_ok if col_ok is not None else False))
        for _ in range(8):
            free |= soft & ((A0 @ free.astype(float)) / deg0 >= 0.34)
        from scipy.sparse.csgraph import connected_components
        sub = A0[free][:, free]
        ncomp, lab = connected_components(sub, directed=False)
        seed_in = (dc[free] < 0.035)
        good = np.zeros(ncomp, bool)
        good[np.unique(lab[seed_in])] = True
        good |= np.bincount(lab, minlength=ncomp) > 300
        keep = np.zeros(len(P), bool)
        keep[np.nonzero(free)[0][good[lab]]] = True
        free = keep
        print(f"  cape: {int(free.sum())} free-sheet vertices follow the cape joints only"
              + (f" ({int((cand & ~free).sum())} thin pieces near the cape kept on the body by colour)" if col_ok is not None else ""))

        def cape_rows(idx):
            r = np.zeros((len(idx), len(names)))
            r[:, cape] = np.exp(-(np.stack([Draw[names[k]][idx] for k in cape], 1) - dc[idx][:, None]) / SIGMA)
            return r / r.sum(1, keepdims=True)

        body_w = W.copy(); body_w[:, cape] = 0
        bs = body_w.sum(1)
        attach_cols = [k for k, n in enumerate(names) if n in CAPE_ATTACH]
        attach = np.where(bs > 1e-6, body_w[:, attach_cols].sum(1) / np.maximum(bs, 1e-9), 1.0)
        body_w /= np.maximum(bs, 1e-9)[:, None]
        W[free] = cape_rows(np.nonzero(free)[0])

        cnt = free[T].sum(1)
        T, T_out = T.copy(), T_all.copy()
        dup_id, orig_dup, new_rows, new_pos = {}, {}, [], []
        for t in np.nonzero((cnt > 0) & (cnt < 3))[0]:
            vs = T[t].copy(); isc = free[vs]
            if attach[vs[~isc]].mean() > 0.5:
                continue                                       # attachment to the torso: stays welded
            tri_cape = isc.sum() >= 2
            for c in range(3):
                v = int(vs[c])
                if isc[c] == tri_cape:
                    continue
                key = (v, bool(tri_cape))
                if key not in dup_id:
                    dup_id[key] = len(P) + len(new_rows)
                    if tri_cape:
                        row = cape_rows(np.array([v]))[0]
                    else:
                        row = body_w[v] if bs[v] > 1e-6 else body_w[vs[~isc]].mean(0)
                    new_rows.append(row); new_pos.append(P[v])
                T[t, c] = dup_id[key]
                k2 = (int(T_all[t, c]), dup_id[key])
                if k2 not in orig_dup:
                    orig_dup[k2] = n_src + len(extra_src)
                    extra_src.append(k2[0]); extra_wid.append(k2[1])
                T_out[t, c] = orig_dup[k2]
        if new_rows:
            W = np.vstack([W, np.array(new_rows)]); P = np.vstack([P, np.array(new_pos)])
        print(f"  cape: mesh cut along {len(new_rows)} fused border vertices ({len(extra_src)} vertices added)")

    # Laplacian smoothing over the welded surface
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    A = coo_matrix((np.ones(len(e) * 2), (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))),
                   shape=(len(P), len(P))).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    lam = np.where(deg > 0, SMOOTH_LAMBDA, 0.0)[:, None]     # vertices left without triangles by the cut keep their weights
    deg = np.maximum(deg, 1)
    x = P[:, 0]
    for _ in range(SMOOTH_ITERS):
        W = (1 - lam) * W + lam * (A @ W) / deg[:, None]
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

    if extra_src:
        # the cut added vertices: copies of source vertices `dup_src` (appended after the source ones) + new triangles
        sel = np.concatenate([inv, np.array(extra_wid)])
        np.savez_compressed(a.out, joint_names=np.array(names), joint_idx=order[sel].astype(np.uint16), weights=w[sel].astype(np.float32),
                            dup_src=np.array(extra_src, np.uint32), triangles=T_out.astype(np.uint32))
    else:
        np.savez_compressed(a.out, joint_names=np.array(names), joint_idx=order[inv].astype(np.uint16), weights=w[inv].astype(np.float32))
    dom = order[:, 0]
    print("dominant-joint vertex counts:", {names[k]: int((dom == k).sum()) for k in range(len(names)) if (dom == k).any()})
    print(f"influences per vertex: {np.bincount((w > 0).sum(1), minlength=5)[1:].tolist()} (1..4)")
    print(f"wrote {a.out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
