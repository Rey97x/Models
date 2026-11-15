"""
Crude preview skinning (Euclidean distance to bone segments).  NOT the final skin: it only exists so that
animations can be previewed on the mesh before the real weights (character3d/rig/weights.npz) are available.
"""
import numpy as np
from skeleton_def import HIERARCHY, END_SITES

PRIMARY_CHILD = {"Hips": "Spine", "Spine2": "Neck", "LeftForeArm": "LeftHand", "RightForeArm": "RightHand"}


def bone_segments(points):
    """{joint: (head, tail)}; tail = primary child head or end site (Root gets a tiny stub)."""
    kids = {}
    for n, p in HIERARCHY:
        kids.setdefault(p, []).append(n)
    end_of = {v: k for k, v in END_SITES.items()}
    segs = {}
    for n, _ in HIERARCHY:
        h = np.asarray(points[n], float)
        if n in PRIMARY_CHILD:
            t = points[PRIMARY_CHILD[n]]
        elif n in end_of and end_of[n] in points:
            t = points[end_of[n]]
        elif kids.get(n):
            t = points[kids[n][0]]
        else:
            t = h + [0, 0.02, 0]
        segs[n] = (h, np.asarray(t, float))
    return segs


def seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def quick_weights(P, points, sigma=0.012):
    names = [n for n, _ in HIERARCHY]
    segs = bone_segments(points)
    D = np.full((len(P), len(names)), 1e9)
    for k, n in enumerate(names):
        if n == "Root":
            continue
        d = seg_dist(P, *segs[n])
        if n.startswith("Left"):
            d = np.where(P[:, 0] < -0.02, 1e9, d)
        if n.startswith("Right"):
            d = np.where(P[:, 0] > 0.02, 1e9, d)
        D[:, k] = d
    W = np.exp(-(D - D.min(1, keepdims=True)) / sigma)
    idx = np.argsort(-W, 1)[:, :4]
    w = np.take_along_axis(W, idx, 1)
    w[w < 0.03] = 0
    w /= w.sum(1, keepdims=True)
    return idx, w
