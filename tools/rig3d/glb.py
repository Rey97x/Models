"""
Minimal GLB (glTF 2.0 binary) reader / writer with skin + animation helpers.  numpy only.

Conventions used by the whole rig3d pipeline
  * +Y up, model faces +Z, character's LEFT is +X   (glTF convention)
  * every joint has IDENTITY rest rotation (joint axes are world-aligned in the bind pose);
    a joint's rest local translation is  head - parent.head
  * quaternions are glTF order  [x, y, z, w]
  * a joint's animated local rotation q therefore means:  world = parent_world * q
"""
import json
import struct

import numpy as np

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
COMP_ID = {np.dtype(v): k for k, v in COMP.items()}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


class GLB:
    def __init__(self, js, binary):
        self.js = js
        self.bin = bytearray(binary)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            magic, version, _length = struct.unpack("<4sII", f.read(12))
            assert magic == b"glTF" and version == 2, "not a GLB v2 file"
            clen, ctype = struct.unpack("<II", f.read(8))
            assert ctype == 0x4E4F534A
            js = json.loads(f.read(clen))
            hdr = f.read(8)
            binary = b""
            if len(hdr) == 8:
                blen, btype = struct.unpack("<II", hdr)
                assert btype == 0x004E4942
                binary = f.read(blen)
        return cls(js, binary)

    def save(self, path):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.js.setdefault("buffers", [{}])
        self.js["buffers"][0] = {"byteLength": len(self.bin)}
        j = json.dumps(self.js, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        j += b" " * ((4 - len(j) % 4) % 4)
        total = 12 + 8 + len(j) + 8 + len(self.bin)
        with open(path, "wb") as f:
            f.write(struct.pack("<4sII", b"glTF", 2, total))
            f.write(struct.pack("<II", len(j), 0x4E4F534A))
            f.write(j)
            f.write(struct.pack("<II", len(self.bin), 0x004E4942))
            f.write(self.bin)

    # ------------------------------------------------------------ accessors
    def accessor(self, idx):
        """Return accessor `idx` as a numpy array of shape (count, ncomp) (or (count,) for SCALAR)."""
        a = self.js["accessors"][idx]
        bv = self.js["bufferViews"][a["bufferView"]]
        dt = np.dtype(COMP[a["componentType"]])
        n = NCOMP[a["type"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", 0) or dt.itemsize * n
        if stride == dt.itemsize * n:
            arr = np.frombuffer(self.bin, dt, a["count"] * n, off).reshape(a["count"], n)
        else:
            raw = np.frombuffer(self.bin, np.uint8, stride * (a["count"] - 1) + dt.itemsize * n, off)
            idxs = (np.arange(a["count"])[:, None] * stride + np.arange(dt.itemsize * n)[None, :])
            arr = raw[idxs].copy().view(dt).reshape(a["count"], n)
        arr = arr.copy()
        return arr[:, 0] if n == 1 else arr

    def add_view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        off = len(self.bin)
        self.bin.extend(data)
        bv = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            bv["target"] = target
        self.js.setdefault("bufferViews", []).append(bv)
        return len(self.js["bufferViews"]) - 1

    def add_accessor(self, arr, type_, target=None, normalized=False, minmax=False):
        arr = np.ascontiguousarray(arr)
        n = NCOMP[type_]
        arr = arr.reshape(-1, n) if n > 1 else arr.reshape(-1)
        view = self.add_view(arr.tobytes(), target)
        acc = {"bufferView": view, "componentType": COMP_ID[arr.dtype], "count": int(arr.shape[0]), "type": type_}
        if normalized:
            acc["normalized"] = True
        if minmax:
            a2 = arr.reshape(arr.shape[0], -1)
            acc["min"] = [float(v) for v in a2.min(0)]
            acc["max"] = [float(v) for v in a2.max(0)]
        self.js.setdefault("accessors", []).append(acc)
        return len(self.js["accessors"]) - 1

    # ----------------------------------------------------------------- mesh
    def mesh_data(self, mesh=0, prim=0):
        """positions (N,3) float32, normals, uvs, triangles (M,3) uint32 of one primitive."""
        p = self.js["meshes"][mesh]["primitives"][prim]
        at = p["attributes"]
        out = {"positions": self.accessor(at["POSITION"]).astype(np.float32)}
        if "NORMAL" in at:
            out["normals"] = self.accessor(at["NORMAL"]).astype(np.float32)
        if "TEXCOORD_0" in at:
            out["uvs"] = self.accessor(at["TEXCOORD_0"]).astype(np.float32)
        if "indices" in p:
            out["triangles"] = self.accessor(p["indices"]).astype(np.uint32).reshape(-1, 3)
        return out

    def set_mesh_arrays(self, positions, normals, uvs, triangles, mesh=0, prim=0):
        """Replace the geometry of one primitive (call BEFORE add_skin / add_animation: the mesh accessors must be the
        only ones).  The old vertex / index data is dropped from the buffer, images are kept."""
        js = self.js
        p = js["meshes"][mesh]["primitives"][prim]
        used = set(p["attributes"].values()) | {p["indices"]}
        assert used == set(range(len(js["accessors"]))), "set_mesh_arrays: unexpected extra accessors"
        imgs = [(im, bytes(self.bin[js["bufferViews"][im["bufferView"]].get("byteOffset", 0):
                                    js["bufferViews"][im["bufferView"]].get("byteOffset", 0) + js["bufferViews"][im["bufferView"]]["byteLength"]]))
                for im in js.get("images", []) if "bufferView" in im]
        js["bufferViews"], js["accessors"], self.bin = [], [], bytearray()
        at = {"POSITION": self.add_accessor(np.asarray(positions, np.float32), "VEC3", ARRAY_BUFFER, minmax=True)}
        if normals is not None:
            at["NORMAL"] = self.add_accessor(np.asarray(normals, np.float32), "VEC3", ARRAY_BUFFER)
        if uvs is not None:
            at["TEXCOORD_0"] = self.add_accessor(np.asarray(uvs, np.float32), "VEC2", ARRAY_BUFFER)
        tri = np.asarray(triangles).reshape(-1)
        p["attributes"] = at
        p["indices"] = self.add_accessor(tri.astype(np.uint16 if len(positions) <= 65535 else np.uint32), "SCALAR", ELEMENT_ARRAY_BUFFER)
        for im, data in imgs:
            im["bufferView"] = self.add_view(data)

    def node_index(self, name):
        for i, n in enumerate(self.js["nodes"]):
            if n.get("name") == name:
                return i
        raise KeyError(name)

    # ----------------------------------------------------------------- skin
    def add_skin(self, joints, joint_idx, weights, mesh_node=0, mesh=0, prim=0, name="Armature"):
        """
        joints    : list of {"name", "parent" (name or None), "head": [x,y,z]}  parents before children
        joint_idx : (N,4) int   indices into `joints`
        weights   : (N,4) float, rows sum to 1
        Adds joint nodes (identity rest rotation), the skin with inverse bind matrices and
        JOINTS_0 / WEIGHTS_0 attributes.  Returns {joint name: node index}.
        """
        js = self.js
        nodes = js["nodes"]
        heads = {j["name"]: np.asarray(j["head"], np.float64) for j in joints}
        node_of = {}
        roots = []
        for j in joints:
            ph = heads[j["parent"]] if j["parent"] else np.zeros(3)
            node = {"name": j["name"], "translation": [float(v) for v in heads[j["name"]] - ph]}
            nodes.append(node)
            node_of[j["name"]] = len(nodes) - 1
            if j["parent"]:
                nodes[node_of[j["parent"]]].setdefault("children", []).append(node_of[j["name"]])
            else:
                roots.append(node_of[j["name"]])
        ibm = np.tile(np.eye(4, dtype=np.float32).reshape(1, 16), (len(joints), 1))
        for k, j in enumerate(joints):
            ibm[k, 12:15] = -heads[j["name"]]          # column-major: translation in elements 12..14
        ibm_acc = self.add_accessor(ibm, "MAT4")
        skin = {"name": name, "joints": [node_of[j["name"]] for j in joints], "inverseBindMatrices": ibm_acc,
                "skeleton": roots[0]}
        js.setdefault("skins", []).append(skin)
        mn = nodes[mesh_node]
        mn["skin"] = len(js["skins"]) - 1
        for k in ("translation", "rotation", "scale", "matrix"):   # ignored for skinned meshes anyway
            mn.pop(k, None)
        scene = js["scenes"][js.get("scene", 0)]
        for r in roots:
            if r not in scene["nodes"]:
                scene["nodes"].append(r)
        w = np.asarray(weights, np.float32)
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
        ji = np.asarray(joint_idx)
        ji = np.where(w > 0, ji, 0)
        jdt = np.uint8 if len(joints) <= 255 else np.uint16
        at = js["meshes"][mesh]["primitives"][prim]["attributes"]
        at["JOINTS_0"] = self.add_accessor(ji.astype(jdt), "VEC4", ARRAY_BUFFER)
        at["WEIGHTS_0"] = self.add_accessor(w.astype(np.float32), "VEC4", ARRAY_BUFFER)
        return node_of

    # ------------------------------------------------------------ animation
    def add_animation(self, name, tracks, extras=None):
        """
        tracks : {joint/node name: {"t": (K,) seconds, "rotation": (K,4) xyzw [optional],
                                     "translation": (K,3) [optional], "scale": (K,3) [optional]}}
        All samplers use LINEAR interpolation.  Time accessors are shared between identical tracks.
        """
        js = self.js
        anim = {"name": name, "samplers": [], "channels": []}
        if extras:
            anim["extras"] = extras
        tcache = {}
        for jname, tr in tracks.items():
            node = self.node_index(jname)
            t = np.asarray(tr["t"], np.float32)
            key = t.tobytes()
            if key not in tcache:
                tcache[key] = self.add_accessor(t, "SCALAR", minmax=True)
            for path, typ in (("translation", "VEC3"), ("rotation", "VEC4"), ("scale", "VEC3")):
                if tr.get(path) is None:
                    continue
                v = np.asarray(tr[path], np.float32)
                assert v.shape[0] == t.shape[0], f"{name}/{jname}/{path}: {v.shape[0]} keys vs {t.shape[0]} times"
                if path == "rotation":
                    v = v / np.linalg.norm(v, axis=1, keepdims=True)
                    # keep neighbouring keys in the same hemisphere so LINEAR (slerp) takes the short way
                    for i in range(1, len(v)):
                        if np.dot(v[i], v[i - 1]) < 0:
                            v[i] = -v[i]
                out = self.add_accessor(v, typ)
                anim["samplers"].append({"input": tcache[key], "output": out, "interpolation": "LINEAR"})
                anim["channels"].append({"sampler": len(anim["samplers"]) - 1, "target": {"node": node, "path": path}})
        js.setdefault("animations", []).append(anim)
        return len(js["animations"]) - 1


# ------------------------------------------------------------------ quaternion helpers (xyzw)
def q_mul(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], -1)


def q_conj(q):
    q = np.asarray(q, float)
    return q * np.array([-1, -1, -1, 1.0])


def q_axis_angle(axis, angle_rad):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle_rad / 2)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle_rad / 2)])


def q_rotate(q, v):
    q, v = np.asarray(q, float), np.asarray(v, float)
    u, w = q[..., :3], q[..., 3:4]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def q_from_matrix(m):
    m = np.asarray(m, float)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def q_to_matrix(q):
    x, y, z, w = np.asarray(q, float)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def fk(joints, local_rot, local_pos=None):
    """
    Forward kinematics for the identity-rest skeleton.
    joints    : list of {"name","parent","head"} (parents first)
    local_rot : {name: quat xyzw}   (missing = identity)
    local_pos : {name: absolute local translation} (missing = rest offset)
    Returns {name: (world_pos (3,), world_quat (4,))}
    """
    heads = {j["name"]: np.asarray(j["head"], float) for j in joints}
    out = {}
    for j in joints:
        n, p = j["name"], j["parent"]
        ql = np.asarray(local_rot.get(n, [0, 0, 0, 1.0]), float)
        rest = heads[n] - (heads[p] if p else 0)
        tl = np.asarray(local_pos[n], float) if local_pos and n in local_pos else rest
        if p:
            pp, pq = out[p]
            out[n] = (pp + q_rotate(pq, tl), q_mul(pq, ql))
        else:
            out[n] = (tl, ql)
    return out
