#!/usr/bin/env python3
"""
Independent foot-contact audit of a rigged + animated GLB.  Reads ONLY the GLB (plus skeleton.json for the sole
geometry), evaluates every clip like a glTF player (LINEAR / slerp, also between keys) and measures:

  * sliding   : for each sole point (heel, ball, toe tip) and each contact (consecutive 30 fps keys on which it lies
                on the ground), the horizontal distance it travels, sampled also between the keys.  In-place clips are judged after applying
                the root motion they declare in `extras` (speed along +Z, turnRate about +Y).
  * ground    : lowest sole point (penetration) per clip
  * knees     : interior knee angle range, largest change per frame, leg stretch ratio
  * loop seam : pose difference between first and last key
  * root      : displacement / yaw of the Root joint over the clip

  python3 tools/rig3d/audit_feet.py character3d/nomade3d_rigged.glb [--skeleton character3d/rig/skeleton.json] [--hz 120]
Exit code 1 if a threshold in CONTRACT.md is violated.
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from glb import GLB, q_mul, q_rotate, q_axis_angle

CONTACT_Y = 0.0008     # a sole point this close to the ground (about 1.5 mm on a real person) counts as touching
SLIDE_TOL = 0.0005     # CONTRACT.md
PEN_TOL = -0.001
STRETCH_TOL = 0.995


def slerp(a, b, u):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + u * (b - a)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - u) * th) * a + np.sin(u * th) * b) / np.sin(th)
    return q / np.linalg.norm(q)


class Player:
    def __init__(self, glb):
        self.g = glb
        js = glb.js
        self.nodes = js["nodes"]
        self.parent = {}
        for i, n in enumerate(self.nodes):
            for c in n.get("children", []):
                self.parent[c] = i
        self.name = {n.get("name"): i for i, n in enumerate(self.nodes)}
        self.joints = js["skins"][0]["joints"]
        self.order = sorted(self.joints, key=self.depth)
        self.clips = {}
        for a in js.get("animations", []):
            ch = {}
            for c in a["channels"]:
                s = a["samplers"][c["sampler"]]
                ch[(c["target"]["node"], c["target"]["path"])] = (glb.accessor(s["input"]).astype(float), glb.accessor(s["output"]).astype(float))
            dur = max(t[-1] for t, _ in ch.values())
            self.clips[a["name"]] = {"ch": ch, "dur": float(dur), "extras": a.get("extras", {})}

    def depth(self, i):
        d = 0
        while i in self.parent:
            i = self.parent[i]; d += 1
        return d

    def local(self, clip, node, t):
        n = self.nodes[node]
        tr = np.array(n.get("translation", [0, 0, 0]), float)
        rot = np.array(n.get("rotation", [0, 0, 0, 1]), float)
        for path in ("translation", "rotation"):
            c = clip["ch"].get((node, path))
            if c is None:
                continue
            ts, vs = c
            if t <= ts[0]:
                v = vs[0]
            elif t >= ts[-1]:
                v = vs[-1]
            else:
                k = int(np.searchsorted(ts, t, side="right") - 1)
                u = (t - ts[k]) / (ts[k + 1] - ts[k])
                v = slerp(vs[k], vs[k + 1], u) if path == "rotation" else vs[k] + u * (vs[k + 1] - vs[k])
            if path == "translation":
                tr = v
            else:
                rot = v / np.linalg.norm(v)
        return tr, rot

    def pose(self, clip, t):
        """{node index: (world position, world quaternion)} for all joints."""
        out = {}
        for j in self.order:
            tr, rot = self.local(clip, j, t)
            p = self.parent.get(j)
            if p in out:
                pp, pq = out[p]
                out[j] = (pp + q_rotate(pq, tr), q_mul(pq, rot))
            else:
                out[j] = (tr, rot)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("--skeleton", default=os.path.join(HERE, "..", "..", "character3d", "rig", "skeleton.json"))
    ap.add_argument("--hz", type=float, default=120)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    import json
    pts = {k: np.array(v, float) for k, v in json.load(open(a.skeleton))["points"].items()}
    P = Player(GLB.load(a.glb))
    N = P.name
    ok_all = True
    # sole points: (label, joint they are rigid with, rest position)
    sole = {}
    for s, S in (("L", "Left"), ("R", "Right")):
        ball = pts[S + "ToeBase"].copy(); ball[1] = 0.0
        tip = pts[S + "Toe_End"].copy(); tip[1] = 0.0
        # the ball point belongs to the toe segment: during heel-off the foot pitches about the ToeBase joint
        # while the toes stay flat, so the sole under that joint is carried by ToeBase, not by Foot
        sole[s] = [("heel", S + "Foot", pts[S + "Heel"]), ("ball", S + "ToeBase", ball), ("toe", S + "ToeBase", tip)]
    print(f"{'clip':<18}{'dur':>6}  {'max slide':>9}  {'min y':>8}  {'knee deg':>13}  {'stretch':>7}  {'seam':>8}  root motion")
    for name, clip in P.clips.items():
        ex = clip["extras"] or {}
        dur = clip["dur"]
        step = max(1, int(round(a.hz / 30)))
        n = int(round(dur * 30)) * step + 1
        ts = np.linspace(0, dur, n)
        speed = float(ex.get("speed", 0) or 0) if not ex.get("rootMotion") else 0.0
        turn = float(ex.get("turnRate", 0) or 0) if not ex.get("rootMotion") else 0.0
        turn = abs(turn) * (-1 if "Right" in name else 1)
        tracks = {s: {lab: np.zeros((n, 3)) for lab, _, _ in sole[s]} for s in sole}
        knee = {s: np.zeros(n) for s in sole}
        stretch = {s: np.zeros(n) for s in sole}
        root = np.zeros((n, 4))
        for i, t in enumerate(ts):
            w = P.pose(clip, t)
            yaw_q = q_axis_angle([0, 1, 0], np.radians(turn) * t)      # what the game engine would add
            shift = np.array([0, 0, speed * t])
            rp, rq = w[N["Root"]]
            root[i] = [rp[0], rp[2], 2 * np.arctan2(rq[1], rq[3]), rp[1]]
            for s, S in (("L", "Left"), ("R", "Right")):
                for lab, jn, rest in sole[s]:
                    jp, jq = w[N[jn]]
                    world = jp + q_rotate(jq, rest - pts[jn])
                    tracks[s][lab][i] = q_rotate(yaw_q, world) + shift
                h, k, f = w[N[S + "UpLeg"]][0], w[N[S + "Leg"]][0], w[N[S + "Foot"]][0]
                u, v = h - k, f - k
                knee[s][i] = np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
                stretch[s][i] = np.linalg.norm(f - h) / (np.linalg.norm(u) + np.linalg.norm(v))
        worst, worst_desc, miny = 0.0, "", 1e9
        for s in sole:
            for lab in tracks[s]:
                tr = tracks[s][lab]
                miny = min(miny, tr[:, 1].min())
                # a point is planted on the 30 fps KEYS where it lies on the ground; its drift is measured over all
                # samples (also between keys) from the first to the last planted key of each contact.  The landing
                # and lift-off transitions (the key interval before / after) are motion, not sliding.
                keys = np.arange(0, n, step)
                on = tr[keys, 1] < CONTACT_Y
                i = 0
                while i < len(keys):
                    if not on[i]:
                        i += 1; continue
                    j = i
                    while j + 1 < len(keys) and on[j + 1]:
                        j += 1
                    seg = tr[keys[i]:keys[j] + 1][:, [0, 2]]
                    d = float(np.linalg.norm(seg - seg[0], axis=1).max()) if j > i else 0.0
                    if a.verbose and j > i:
                        print(f"    {name} {s}-{lab}: planted {ts[keys[i]]:.3f}-{ts[keys[j]]:.3f}s slide {d:.5f}")
                    if d > worst:
                        worst, worst_desc = d, f"{s}-{lab} {ts[keys[i]]:.2f}-{ts[keys[j]]:.2f}s"
                    i = j + 1
        # loop seam: local rotations of all joints except Root, first vs last key
        seam = 0.0
        if ex.get("loop", True) or name.startswith("Turn360"):
            for j in P.joints:
                if P.nodes[j].get("name") == "Root":
                    continue
                t0, r0 = P.local(clip, j, 0.0); t1, r1 = P.local(clip, j, dur)
                seam = max(seam, 1 - abs(float(np.dot(r0, r1))), float(np.abs(t0 - t1).max()) if P.nodes[j].get("name") != "Root" else 0)
        yaw = np.unwrap(root[:, 2])
        kmin = min(knee[s].min() for s in sole); kmax = max(knee[s].max() for s in sole)
        kjump = max(np.abs(np.diff(knee[s])).max() * step for s in sole)       # deg per 1/30 s
        smax = max(stretch[s].max() for s in sole)
        rm = f"dz={root[-1,1]-root[0,1]:+.4f} dx={root[-1,0]-root[0,0]:+.4f} yaw={np.degrees(yaw[-1]-yaw[0]):+.1f}deg"
        extra = (f" | in-place speed {speed:.4f}/s" if speed else "") + (f" | in-place turn {turn:+.1f} deg/s" if turn else "")
        bad = []
        if worst > SLIDE_TOL: bad.append("SLIDE")
        if miny < PEN_TOL: bad.append("PENETRATION")
        if smax > STRETCH_TOL: bad.append("STRETCH")
        if seam > 1e-4: bad.append("SEAM")
        ok_all &= not bad
        print(f"{name:<18}{dur:>6.2f}  {worst:>9.5f}  {miny:>8.4f}  {kmin:>5.0f}-{kmax:<3.0f} d{kjump:>3.0f}  {smax:>7.4f}  {seam:>8.1e}  {rm}{extra}"
              f"   {'FAIL ' + ','.join(bad) + ' (' + worst_desc + ')' if bad else 'ok'}")
    print("FOOT AUDIT:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
