"""
Build a 2D skinned-mesh rig from a single front-facing character image.

Pipeline
  1. Background removal  (white-background flood fill from image borders)
  2. Skeleton definition (joint positions in image pixels)
  3. Mesh generation      (regular grid clipped to the silhouette, triangulated)
  4. Skin weights         (geodesic distance from every bone, computed inside the
                           silhouette so separated limbs never influence each other)
  5. Export               (texture PNG with alpha + rig.json for the viewer)

Usage:  python3 tools/build_rig.py
Requires: numpy, pillow, scipy
"""
import json
import os

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "character", "src", "reference.jpg")
OUT = os.path.join(ROOT, "character", "rig")

# Image-space origin of the rig (feet centre on the ground) and pixel -> unit scale
ORIGIN = (900.0, 2330.0)
PX_PER_UNIT = 1000.0
GRID = 12          # mesh cell size, in source pixels
GEO_DOWN = 4       # down-sampling for the geodesic distance field
SIGMA = 7.0        # weight falloff (in down-sampled pixels)
MAX_INF = 4        # influences per vertex

# ---------------------------------------------------------------------------
# Skeleton.  "R" = character's right = screen LEFT, "L" = screen RIGHT.
# head/tail are image pixel coordinates.  `seg` optionally overrides the segment
# used for skin weighting (e.g. thighs only own the trousers below the tunic).
# `z` is the draw-order layer (bigger = closer to camera).
# ---------------------------------------------------------------------------
BONES = [
    # name,          parent,        head,          tail,          z,    seg
    ("root",        None,          (900, 2330),   (900, 2250),   0.0,  None),
    ("hips",        "root",        (900, 1250),   (900, 1100),   0.0,  ((900, 1290), (900, 1100))),
    ("spine",       "hips",        (900, 1100),   (900, 850),    0.0,  None),
    ("chest",       "spine",       (900, 850),    (900, 600),    0.0,  ((900, 850), (900, 640))),
    ("neck",        "chest",       (900, 600),    (900, 440),    0.1,  ((900, 560), (900, 440))),
    ("head",        "neck",        (900, 440),    (900, 130),    0.1,  None),

    ("upperarm_R",  "chest",       (630, 585),    (480, 830),    0.4,  ((610, 615), (480, 830))),
    ("forearm_R",   "upperarm_R",  (480, 830),    (340, 1090),   0.5,  None),
    ("hand_R",      "forearm_R",   (340, 1090),   (285, 1275),   0.6,  ((335, 1110), (285, 1275))),
    ("ribbon_R1",   "forearm_R",   (408, 1110),   (412, 1310),   0.45, ((410, 1150), (412, 1310))),
    ("ribbon_R2",   "ribbon_R1",   (412, 1310),   (420, 1525),   0.45, None),

    ("upperarm_L",  "chest",       (1170, 585),   (1320, 830),   0.4,  ((1190, 615), (1320, 830))),
    ("forearm_L",   "upperarm_L",  (1320, 830),   (1455, 1090),  0.5,  None),
    ("hand_L",      "forearm_L",   (1455, 1090),  (1575, 1275),  0.6,  ((1465, 1110), (1575, 1275))),
    ("ribbon_L1",   "forearm_L",   (1435, 1110),  (1445, 1300),  0.45, ((1437, 1150), (1445, 1300))),
    ("ribbon_L2",   "ribbon_L1",   (1445, 1300),  (1465, 1480),  0.45, None),

    ("skirt_R",     "hips",        (800, 1230),   (740, 1480),   0.2,  ((790, 1300), (735, 1470))),
    ("skirt_L",     "hips",        (1000, 1230),  (1040, 1480),  0.2,  ((1010, 1300), (1040, 1470))),
    ("sash1",       "hips",        (1090, 1100),  (1135, 1380),  0.3,  ((1100, 1140), (1135, 1380))),
    ("sash2",       "sash1",       (1135, 1380),  (1195, 1660),  0.3,  None),

    ("thigh_R",     "hips",        (790, 1320),   (740, 1780),   -0.2, ((760, 1520), (740, 1780))),
    ("shin_R",      "thigh_R",     (740, 1780),   (735, 2180),   -0.2, None),
    ("foot_R",      "shin_R",      (735, 2180),   (660, 2310),   -0.1, ((735, 2200), (650, 2310))),
    ("thigh_L",     "hips",        (1010, 1320),  (1075, 1780),  -0.2, ((1050, 1520), (1075, 1780))),
    ("shin_L",      "thigh_L",     (1075, 1780),  (1100, 2180),  -0.2, None),
    ("foot_L",      "shin_L",      (1100, 2180),  (1160, 2310),  -0.1, ((1100, 2200), (1170, 2310))),
]


def to_world(p):
    return [round((p[0] - ORIGIN[0]) / PX_PER_UNIT, 5), round((ORIGIN[1] - p[1]) / PX_PER_UNIT, 5)]


def build_mask(img):
    a = np.asarray(img).astype(np.float32)
    mn, mx = a.min(2), a.max(2)
    bg = (mn > 225) & ((mx - mn) < 20)
    lab, _ = ndi.label(bg)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    fg = ~np.isin(lab, list(border))
    fg = ndi.binary_opening(fg, iterations=2)
    lab, n = ndi.label(fg)
    sizes = ndi.sum(fg, lab, range(1, n + 1))
    fg = lab == (np.argmax(sizes) + 1)
    # fill only small enclosed holes (keep the white gaps between ribbons and hands)
    holes = ndi.binary_fill_holes(fg) & ~fg
    hl, hn = ndi.label(holes)
    if hn:
        hs = ndi.sum(holes, hl, range(1, hn + 1))
        small = np.isin(hl, np.nonzero(hs < 400)[0] + 1)
        fg |= small
    # remove whitish floor shadow below the feet
    shadow = (mn > 200) & ((mx - mn) < 25)
    shadow[:2250] = False
    fg &= ~shadow
    return fg


def segment_points(a, b, step=1.0):
    a, b = np.array(a, float), np.array(b, float)
    n = max(2, int(np.linalg.norm(b - a) / step) + 1)
    t = np.linspace(0, 1, n)[:, None]
    return a + (b - a) * t


def geodesic_fields(mask_s):
    """Return a function bone_segment -> distance field (inside mask_s)."""
    h, w = mask_s.shape
    idx = -np.ones((h, w), np.int64)
    ys, xs = np.nonzero(mask_s)
    idx[ys, xs] = np.arange(len(ys))
    rows, cols, vals = [], [], []
    for dy, dx, c in [(0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.4142), (1, -1, 1.4142)]:
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < h) & (x2 >= 0) & (x2 < w)
        ok[ok] &= mask_s[y2[ok], x2[ok]]
        a, b = idx[ys[ok], xs[ok]], idx[y2[ok], x2[ok]]
        rows += [a, b]; cols += [b, a]; vals += [np.full(len(a), c)] * 2
    g = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(len(ys), len(ys)))
    # nearest in-mask pixel for every pixel (to snap sources / vertices)
    _, (ny, nx) = ndi.distance_transform_edt(~mask_s, return_indices=True)

    def field(seg):
        pts = segment_points(seg[0], seg[1], GEO_DOWN) / GEO_DOWN
        px = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        py = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        sy, sx = ny[py, px], nx[py, px]
        src = np.unique(idx[sy, sx])
        d = dijkstra(g, indices=src, min_only=True)
        out = np.full((h, w), np.inf)
        out[ys, xs] = d
        return out

    return field, ny, nx


def main():
    os.makedirs(OUT, exist_ok=True)
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    fg = build_mask(img)

    # ---------------- texture (cropped, with soft alpha) ----------------
    ys, xs = np.nonzero(fg)
    pad = 24
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
    alpha = Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    rgba = img.copy()
    rgba.putalpha(alpha)
    # bleed colours into transparent area to avoid white fringes when filtering
    arr = np.asarray(rgba).copy()
    _, (iy, ix) = ndi.distance_transform_edt(~fg, return_indices=True)
    arr[..., :3] = np.asarray(img)[iy, ix]
    tex = Image.fromarray(arr).crop((x0, y0, x1, y1))
    tex_scale = 0.75
    tex = tex.resize((int(tex.width * tex_scale), int(tex.height * tex_scale)), Image.LANCZOS)
    tex.save(os.path.join(OUT, "texture.png"), optimize=True)

    # ---------------- mesh ----------------
    cw = GRID
    gx = np.arange(x0, x1 + cw, cw)
    gy = np.arange(y0, y1 + cw, cw)
    # a cell is kept if it touches the silhouette (dilated by 1 cell for safety)
    fgd = ndi.binary_dilation(fg, iterations=4)
    cells = np.zeros((len(gy) - 1, len(gx) - 1), bool)
    for j in range(len(gy) - 1):
        ya, yb = gy[j], min(H, gy[j + 1])
        for i in range(len(gx) - 1):
            xa, xb = gx[i], min(W, gx[i + 1])
            if ya < H and xa < W and fgd[ya:yb, xa:xb].any():
                cells[j, i] = True
    vid = -np.ones((len(gy), len(gx)), np.int64)
    verts, tris = [], []

    def v(j, i):
        if vid[j, i] < 0:
            vid[j, i] = len(verts)
            verts.append((float(gx[i]), float(gy[j])))
        return int(vid[j, i])

    for j, i in zip(*np.nonzero(cells)):
        a, b, c, d = v(j, i), v(j, i + 1), v(j + 1, i + 1), v(j + 1, i)
        tris += [(a, d, b), (b, d, c)]  # CCW once y is flipped to world space
    verts = np.array(verts)
    print(f"mesh: {len(verts)} vertices, {len(tris)} triangles")

    # ---------------- weights ----------------
    small = Image.fromarray((fg * 255).astype(np.uint8)).resize((W // GEO_DOWN, H // GEO_DOWN), Image.BOX)
    mask_s = np.asarray(small) > 60
    field, ny, nx = geodesic_fields(mask_s)
    hs, ws = mask_s.shape
    vx = np.clip(np.round(verts[:, 0] / GEO_DOWN).astype(int), 0, ws - 1)
    vy = np.clip(np.round(verts[:, 1] / GEO_DOWN).astype(int), 0, hs - 1)
    sy, sx = ny[vy, vx], nx[vy, vx]
    snap_extra = np.hypot(sy - vy, sx - vx)

    skinnable = [k for k, b in enumerate(BONES) if b[0] != "root"]
    D = np.full((len(verts), len(BONES)), np.inf)
    for k in skinnable:
        name, parent, head, tail, z, seg = BONES[k]
        f = field(seg or (head, tail))
        D[:, k] = f[sy, sx] + snap_extra
        print(f"  weights: {name}")
    dmin = D.min(1, keepdims=True)
    Wt = np.exp(-(D - dmin) / SIGMA)
    Wt[~np.isfinite(D)] = 0
    order = np.argsort(-Wt, 1)[:, :MAX_INF]
    top = np.take_along_axis(Wt, order, 1)
    top[top < 0.02] = 0
    top /= top.sum(1, keepdims=True)

    # z per vertex (draw order) blended by weights
    zl = np.array([b[4] for b in BONES])
    vz = (top * zl[order]).sum(1)

    # ---------------- export ----------------
    bones_out = []
    for name, parent, head, tail, z, seg in BONES:
        bones_out.append({"name": name, "parent": parent,
                          "head": to_world(head), "tail": to_world(tail)})
    pos = []
    uv = []
    for (px, py), z in zip(verts, vz):
        wx, wy = to_world((px, py))
        pos += [wx, wy, round(float(z) * 0.01, 5)]
        uv += [round((px - x0) / (x1 - x0), 5), round(1 - (py - y0) / (y1 - y0), 5)]
    rig = {
        "name": "Nomade del deserto",
        "source": "character/src/reference.jpg",
        "units": "1 unit = 1000 source pixels; origin = feet centre on the ground",
        "texture": "texture.png",
        "bones": bones_out,
        "mesh": {
            "position": pos,
            "uv": uv,
            "index": [int(i) for t in tris for i in t],
            "skinIndex": order.astype(int).flatten().tolist(),
            "skinWeight": [round(float(w), 4) for w in top.flatten()],
        },
    }
    with open(os.path.join(OUT, "rig.json"), "w") as fh:
        json.dump(rig, fh, separators=(",", ":"))
    print("written", OUT)


if __name__ == "__main__":
    main()
