"""
Procedural weapons and hand sockets.

  socket  : where a weapon sits in a hand.  It is measured on the mesh: the hand vertices give the finger direction,
            the palm normal and the thumb side; the grip lies across the palm, the blade leaves on the thumb side and
            the cutting edge faces the knuckles (hammer grip).  The fingers are not articulated, so the hand stays open
            around the grip.
  weapons : low-poly meshes built here (no external assets): "sword" (straight, cross-guard, pommel) and "dagger"
            (short curved blade).  Weapon space: origin = middle of the grip, +Y = towards the tip, +X = cutting edge.

Each weapon is a separate mesh node parented to a `Weapon_<Side>` node under the hand joint, so an engine can hide or
swap it.  Materials are plain PBR factors (steel, brass, leather): they survive FBX export.
"""
import numpy as np
from glb import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, q_from_matrix

MATERIALS = {
    "steel":   {"baseColorFactor": [0.78, 0.80, 0.83, 1.0], "metallicFactor": 1.0, "roughnessFactor": 0.32},
    "brass":   {"baseColorFactor": [0.72, 0.53, 0.22, 1.0], "metallicFactor": 1.0, "roughnessFactor": 0.42},
    "leather": {"baseColorFactor": [0.20, 0.13, 0.09, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.85},
    "wrap":    {"baseColorFactor": [0.55, 0.44, 0.30, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.9},
}

# dimensions in model units (1 unit ~ 1.8 m)
SPECS = {
    "sword":  dict(blade=0.46, width=0.030, thick=0.0075, curve=0.0, grip=0.085, grip_r=0.0085, guard=0.075,
                   guard_t=0.011, pommel=0.014, metal="brass", grip_mat="leather", tip=0.16),
    "dagger": dict(blade=0.150, width=0.024, thick=0.0055, curve=0.022, grip=0.062, grip_r=0.0075, guard=0.036,
                   guard_t=0.008, pommel=0.010, metal="brass", grip_mat="wrap", tip=0.45),
}


class _Mesh:
    def __init__(self):
        self.parts = {}                                   # material -> list of triangles (3,3)

    def tri(self, mat, a, b, c):
        self.parts.setdefault(mat, []).append((a, b, c))

    def quad(self, mat, a, b, c, d):
        self.tri(mat, a, b, c); self.tri(mat, a, c, d)

    def loft(self, mat, rings, cap_start=True, cap_end=True):
        """rings: list of (k,3) loops with the same vertex count, connected in order"""
        for r0, r1 in zip(rings[:-1], rings[1:]):
            k = len(r0)
            for i in range(k):
                j = (i + 1) % k
                self.quad(mat, r0[i], r0[j], r1[j], r1[i])
        for ring, flip, on in ((rings[0], True, cap_start), (rings[-1], False, cap_end)):
            if on:
                c = np.mean(ring, 0)
                for i in range(len(ring)):
                    j = (i + 1) % len(ring)
                    self.tri(mat, *((c, ring[j], ring[i]) if flip else (c, ring[i], ring[j])))


def _ring(center, ax_x, ax_z, rx, rz, k=8):
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return [np.asarray(center) + rx * np.cos(t) * np.asarray(ax_x) + rz * np.sin(t) * np.asarray(ax_z) for t in a]


def build_weapon(kind):
    """returns {material: (positions (n,3), normals (n,3))} flat shaded, in weapon space"""
    s = SPECS[kind]
    m = _Mesh()
    X, Y, Z = np.eye(3)
    g2 = s["grip"] / 2
    # grip: slightly barrelled cylinder
    m.loft(s["grip_mat"], [_ring(Y * y, X, Z, s["grip_r"] * f, s["grip_r"] * f * 0.85)
                           for y, f in ((-g2, 0.92), (-g2 / 2, 1.0), (0, 1.03), (g2 / 2, 1.0), (g2, 0.92))], False, False)
    # pommel
    p0 = -g2
    m.loft(s["metal"], [_ring(Y * (p0 - s["pommel"] * d), X, Z, s["pommel"] * r, s["pommel"] * r)
                        for d, r in ((0.0, 0.55), (0.45, 1.0), (1.2, 1.0), (1.75, 0.5))])
    # guard: bar along the edge direction, a little curved towards the blade
    gt = s["guard_t"]
    for sign in (-1, 1):
        rings = []
        for u in (0.0, 0.5, 1.0):
            c = Y * (g2 + gt / 2 + 0.25 * gt * u * u) + X * sign * s["guard"] / 2 * u
            f = 1.0 - 0.35 * u
            rings.append([c + a * gt / 2 * f * Y + b * gt * 0.62 * f * Z for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))][::sign])
        m.loft(s["metal"], rings, cap_start=False)
    # blade: diamond section swept along a (possibly curved) spine, tapering to the tip
    y0 = g2 + gt
    n = 14
    rings = []
    for i in range(n + 1):
        u = i / n
        w = s["width"] / 2 * (1.0 if u < 1 - s["tip"] else max(0.0, (1 - u) / s["tip"]) ** 0.75)
        th = s["thick"] / 2 * (1.0 - 0.45 * u)
        c = Y * (y0 + s["blade"] * u) + X * (-s["curve"] * 4 * (u * u) * 0.25 + s["curve"] * u * u)
        ang = 2.0 * s["curve"] * u / max(s["blade"], 1e-6) * 1.0          # the section follows the bend
        ax = np.cos(ang) * X + np.sin(ang) * Y
        if i == n:
            rings.append([c] * 4)
        else:
            rings.append([c + ax * w, c + Z * th, c - ax * w * (0.82 if s["curve"] else 1.0), c - Z * th])
    m.loft("steel", rings, cap_start=True, cap_end=False)
    out = {}
    for mat, tris in m.parts.items():
        T = np.array(tris, float)
        T = T[np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) > 1e-12]
        nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        out[mat] = (T.reshape(-1, 3).astype(np.float32), np.repeat(nrm, 3, 0).astype(np.float32))
    return out


def compute_socket(P, hand_w, joints, ends, side, kind, cfg=None):
    """
    P       : (N,3) mesh vertices,  hand_w : (N,) skin weight of the hand joint
    Returns dict(offset (3,), rot (3,3) columns = weapon X (edge), Y (blade), Z in the hand joint's rest frame).
    cfg may override:  tilt (deg, blade tilted from the thumb axis towards the forearm), flip_palm, flip_thumb.
    """
    cfg = cfg or {}
    spec = SPECS[kind]
    heads = {j["name"]: np.asarray(j["head"], float) for j in joints}
    W = heads[side + "Hand"]
    tip = np.asarray(ends[side + "Hand_End"], float)
    f = (tip - W) / np.linalg.norm(tip - W)
    L = np.linalg.norm(tip - W)
    sg = 1.0 if side == "Left" else -1.0
    hv = P[hand_w > 0.6]
    t_along = (hv - W) @ f
    palm = hv[(t_along > 0.18 * L) & (t_along < 0.62 * L)]
    res = palm - np.outer((palm - W) @ f, f) - W
    _, vecs = np.linalg.eigh(np.cov(res.T))
    width = vecs[:, 2] - (vecs[:, 2] @ f) * f                # largest spread across the palm = thumb / pinky axis
    width /= np.linalg.norm(width)
    n = np.cross(f, width)
    if (n[0] * -sg < 0) != bool(cfg.get("flip_palm")):      # the palm faces the body's midline in an A-pose
        n = -n
    # thumb side: where the hand vertices spread further from the palm centre line (the thumb sticks out)
    c = palm.mean(0)
    proj = (hv - c) @ width
    thumb = width if (np.percentile(proj, 97) > -np.percentile(proj, 3)) else -width
    if cfg.get("flip_thumb"):
        thumb = -thumb
    tilt = np.radians(cfg.get("tilt", 12.0))
    blade = np.cos(tilt) * thumb - np.sin(tilt) * f           # leaves on the thumb side, tilted back to the forearm
    blade /= np.linalg.norm(blade)
    edge = f - (f @ blade) * blade                            # the cutting edge faces the knuckles
    edge /= np.linalg.norm(edge)
    z = np.cross(edge, blade)
    half_thick = np.percentile(np.abs((palm - c) @ n), 85)
    center = c + n * (half_thick + spec["grip_r"] * 0.9)
    return dict(offset=center - W, rot=np.stack([edge, blade, z], 1), palm_normal=n, thumb=thumb)


def add_weapon(glb, hand_node, side, kind, socket):
    """adds `Weapon_<side>` (socket transform) under the hand joint node and the weapon mesh below it"""
    js = glb.js
    mats = js.setdefault("materials", [])
    spec_mesh = build_weapon(kind)
    prims = []
    for mat, (pos, nrm) in spec_mesh.items():
        name = "weapon_" + mat
        idx = next((i for i, m in enumerate(mats) if m.get("name") == name), None)
        if idx is None:
            mats.append({"name": name, "pbrMetallicRoughness": dict(MATERIALS[mat])})
            idx = len(mats) - 1
        prims.append({"attributes": {"POSITION": glb.add_accessor(pos, "VEC3", ARRAY_BUFFER, minmax=True),
                                     "NORMAL": glb.add_accessor(nrm, "VEC3", ARRAY_BUFFER)},
                      "indices": glb.add_accessor(np.arange(len(pos), dtype=np.uint16), "SCALAR", ELEMENT_ARRAY_BUFFER),
                      "material": idx})
    js["meshes"].append({"name": f"{kind}_{side}", "primitives": prims})
    q = q_from_matrix(socket["rot"])
    js["nodes"].append({"name": f"Weapon_{side}", "translation": [float(v) for v in socket["offset"]],
                        "rotation": [float(v) for v in q], "mesh": len(js["meshes"]) - 1,
                        "extras": {"weapon": kind}})
    js["nodes"][hand_node].setdefault("children", []).append(len(js["nodes"]) - 1)
