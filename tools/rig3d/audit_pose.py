#!/usr/bin/env python3
"""
Pose quality audit of a rigged + animated GLB: self-penetration, joint naturalness, smoothness.

  python3 tools/rig3d/audit_pose.py <file.glb> --skeleton <skeleton.json> [--clips Attack,Dodge_Back] [--verbose]

What it measures (per clip, sampled at the clip's key rate):
  * penetration : the body is approximated by capsules measured on the mesh (limbs along their bones, the trunk as
                  lateral bars, the head), each weapon blade by a thin capsule.  For every non-adjacent pair the
                  clearance (distance minus radii) is tracked; a pair FAILS when it goes deeper than TOL below what
                  the same pair shows in the Idle start pose (so tight but harmless rest contacts are not reported).
  * wrists      : bend between forearm and hand (deg), limit WRIST_MAX
  * elbows      : interior angle range (a straight arm is 180; below ELBOW_MIN is over-folded)
  * shoulders   : how far the upper arm points behind the chest / across the chest (deg)
  * smoothness  : largest angular speed (deg/s) and the largest speed jump between consecutive keys (deg/s)
Exit code 1 if something fails.
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from glb import GLB, q_rotate, q_mul, q_conj
from audit_feet import Player
from quick_skin import bone_segments
from weapons import SPECS

TOL = 0.006          # allowed extra penetration w.r.t. the Idle start pose (model units, ~1 cm on a real person)
WRIST_MAX = 62.0
ELBOW_MIN = 32.0
SPEED_MAX = 2200.0   # deg/s, any joint (a real sword cut reaches 1500-2000 at the wrist)
JUMP_MAX = 420.0     # deg/s change between consecutive keys at 120 Hz (scaled for other rates)

LIMBS = ["Arm", "ForeArm", "Hand", "UpLeg", "Leg", "Foot"]
TRUNK = ["Hips", "Spine", "Spine1", "Spine2"]


def seg_seg(p1, q1, p2, q2):
    """distance between segments p1q1 and p2q2 (vectorised over the first axis)"""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = np.sum(d1 * d1, -1), np.sum(d2 * d2, -1), np.sum(d2 * r, -1)
    b, c = np.sum(d1 * d2, -1), np.sum(d1 * r, -1)
    den = a * e - b * b
    s = np.where(den > 1e-12, np.clip((b * f - c * e) / np.maximum(den, 1e-12), 0, 1), 0.0)
    t = np.clip((b * s + f) / np.maximum(e, 1e-12), 0, 1)
    s = np.clip((b * t - c) / np.maximum(a, 1e-12), 0, 1)
    return np.linalg.norm((p1 + d1 * s[..., None]) - (p2 + d2 * t[..., None]), axis=-1)


def build_capsules(g, pts):
    """{name: (joint, a_local, b_local, radius)} measured on the bind-pose mesh of the GLB"""
    js = g.js
    skin = js["skins"][0]
    names = [js["nodes"][j]["name"] for j in skin["joints"]]
    prim = next(p for m in js["meshes"] for p in m["primitives"] if "JOINTS_0" in p["attributes"])
    P = g.accessor(prim["attributes"]["POSITION"]).astype(float)
    ji, w = g.accessor(prim["attributes"]["JOINTS_0"]), g.accessor(prim["attributes"]["WEIGHTS_0"])
    dom = ji[np.arange(len(P)), np.argmax(w, 1)]
    return capsules_from_arrays(P, dom, names, pts), names


def capsules_from_arrays(P, dom, names, pts):
    """body capsules from bind-pose vertices `P`, their dominant joint index `dom` and the skeleton points"""
    segs = bone_segments(pts)
    caps = {}
    for n in names:
        v = P[dom == names.index(n)]
        if len(v) < 30 or n not in pts:
            continue
        head = np.asarray(pts[n], float)
        base = n.replace("Left", "").replace("Right", "")
        if base in LIMBS:
            a, b = segs[n]
            ab = b - a
            t = np.clip(((v - a) @ ab) / (ab @ ab), 0, 1)
            r = float(np.percentile(np.linalg.norm(v - (a + t[:, None] * ab), axis=1), 55))
            if base == "Foot":
                r *= 0.8
            caps[n] = (n, a - head, b - head, r)
        elif n in TRUNK:
            x0, x1 = np.percentile(v[:, 0], [8, 92])
            z0, z1 = np.percentile(v[:, 2], [12, 88])
            r = 0.5 * (z1 - z0) * 0.92
            cy = float(np.median(v[:, 1]))
            cz = 0.5 * (z0 + z1)
            hw = max(0.5 * (x1 - x0) - r, 0.005)
            cx = 0.5 * (x0 + x1)
            caps[n] = (n, np.array([cx - hw, cy, cz]) - head, np.array([cx + hw, cy, cz]) - head, r)
        elif n == "Head":
            c = np.median(v, 0)
            r = float(np.percentile(np.linalg.norm(v - c, axis=1), 45))
            caps[n] = (n, c - head - [0, 0.01, 0], c - head + [0, 0.01, 0], r)
    return caps


def adjacent(a, b):
    chain = [["Hips", "Spine", "Spine1", "Spine2", "Head"]]
    for s in ("Left", "Right"):
        chain.append([s + "Arm", s + "ForeArm", s + "Hand"])
        chain.append([s + "UpLeg", s + "Leg", s + "Foot"])
    for c in chain:
        if a in c and b in c and abs(c.index(a) - c.index(b)) <= 1:
            return True
    pairs = {("Spine2", "Arm"), ("Spine1", "Arm"), ("Hips", "UpLeg"), ("Spine", "UpLeg"), ("Spine2", "Head"), ("Spine1", "Head")}
    ba, bb = a.replace("Left", "").replace("Right", ""), b.replace("Left", "").replace("Right", "")
    return (ba, bb) in pairs or (bb, ba) in pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb"); ap.add_argument("--skeleton", required=True)
    ap.add_argument("--clips"); ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    g = GLB.load(a.glb)
    pts = {k: np.array(v, float) for k, v in json.load(open(a.skeleton))["points"].items()}
    P = Player(g)
    N = P.name
    caps, names = build_capsules(g, pts)
    # weapons: blade capsule in the hand joint's frame
    blades = {}
    for i, n in enumerate(g.js["nodes"]):
        if n.get("name", "").startswith("Weapon_"):
            side = n["name"].split("_")[1]
            spec = SPECS[n["extras"]["weapon"]]
            off, q = np.array(n["translation"], float), np.array(n["rotation"], float)
            y0 = spec["grip"] / 2 + spec["guard_t"]
            blades["Blade" + side] = (side + "Hand", off + q_rotate(q, np.array([0, y0, 0.0])),
                                      off + q_rotate(q, np.array([0, y0 + spec["blade"], 0.0])), 0.006)
    allc = dict(caps); allc.update(blades)
    pairs = []
    keys = list(allc)
    for i, x in enumerate(keys):
        for y in keys[i + 1:]:
            if x.startswith("Blade") and y.startswith("Blade"):
                pairs.append((x, y)); continue
            bx = x.startswith("Blade"); by = y.startswith("Blade")
            if bx or by:
                bl, other = (x, y) if bx else (y, x)
                side = bl[5:]
                if other in (side + "Hand", side + "ForeArm"):
                    continue
                pairs.append((x, y)); continue
            if adjacent(x, y):
                continue
            arm = lambda n: any(t in n for t in ("Arm", "Hand"))
            leg = lambda n: any(t in n for t in ("UpLeg", "Leg", "Foot")) and "Arm" not in n
            if arm(x) or arm(y) or (leg(x) and leg(y) and x[:4] != y[:4]):
                pairs.append((x, y))

    def world(clip, t):
        w = P.pose(clip, t)
        out = {}
        for k, (jn, la, lb, r) in allc.items():
            jp, jq = w[N[jn]]
            out[k] = (jp + q_rotate(jq, la), jp + q_rotate(jq, lb), r)
        return w, out

    idle = P.clips.get("Idle")
    base = {}
    if idle:
        _, c0 = world(idle, 0.0)
        for x, y in pairs:
            base[(x, y)] = float(seg_seg(c0[x][0][None], c0[x][1][None], c0[y][0][None], c0[y][1][None])[0] - c0[x][2] - c0[y][2])
    want = a.clips.split(",") if a.clips else list(P.clips)
    ok_all = True
    rest_bad = sorted((v, x, y) for (x, y), v in base.items() if v < -TOL and ("Blade" in x or "Blade" in y))
    for v, x, y in rest_bad:
        print(f"Idle start pose: {x} / {y} overlap by {-v:.3f}")
        ok_all = False
    for name in want:
        clip = P.clips[name]
        kt = max((t for t, _ in clip["ch"].values()), key=len)
        fps = int(round(1.0 / np.median(np.diff(kt)))) if len(kt) > 1 else 30
        n = int(round(clip["dur"] * fps)) + 1
        ts = np.linspace(0, clip["dur"], n)
        worst = {}
        wrist = {s: [] for s in ("Left", "Right")}
        elbow = {s: [] for s in ("Left", "Right")}
        behind = {s: [] for s in ("Left", "Right")}
        local = {j: [] for j in P.joints}
        for t in ts:
            w, c = world(clip, t)
            for x, y in pairs:
                d = float(seg_seg(c[x][0][None], c[x][1][None], c[y][0][None], c[y][1][None])[0] - c[x][2] - c[y][2])
                ref = min(0.0, base.get((x, y), 0.0))
                if d - ref < worst.get((x, y), (1e9, 0))[0]:
                    worst[(x, y)] = (d - ref, t)
            for s in ("Left", "Right"):
                sh, el, wr = w[N[s + "Arm"]][0], w[N[s + "ForeArm"]][0], w[N[s + "Hand"]][0]
                tip = wr + q_rotate(w[N[s + "Hand"]][1], pts[s + "Hand_End"] - pts[s + "Hand"])
                u, v, h = sh - el, wr - el, tip - wr
                ang = lambda p, q: float(np.degrees(np.arccos(np.clip(p @ q / (np.linalg.norm(p) * np.linalg.norm(q)), -1, 1))))
                # rest bend between forearm and hand is subtracted (the modelled hand is not in line with the forearm)
                rest = ang(pts[s + "Hand"] - pts[s + "ForeArm"], pts[s + "Hand_End"] - pts[s + "Hand"])
                wrist[s].append(abs(ang(v, h) - rest) if False else ang(q_rotate(q_conj(w[N[s + "ForeArm"]][1]), h), pts[s + "Hand_End"] - pts[s + "Hand"]))
                elbow[s].append(ang(u, v))
                cq = w[N["Spine2"]][1]
                d = q_rotate(q_conj(cq), el - sh); d /= np.linalg.norm(d)
                behind[s].append(float(np.degrees(np.arcsin(np.clip(-d[2], -1, 1)))))
            for j in P.joints:
                local[j].append(P.local(clip, j, t)[1])
        sp_max, jump_max, sp_j, jump_j = 0.0, 0.0, "", ""
        for j, qs in local.items():
            qs = np.array(qs)
            dots = np.clip(np.abs(np.sum(qs[1:] * qs[:-1], -1)), -1, 1)
            sp = np.degrees(2 * np.arccos(dots)) * fps
            if len(sp) and sp.max() > sp_max:
                sp_max, sp_j = float(sp.max()), P.nodes[j]["name"]
            if len(sp) > 1:
                jm = np.abs(np.diff(sp)).max() * (fps / 120.0)
                if jm > jump_max:
                    jump_max, jump_j = float(jm), P.nodes[j]["name"]
        bad = sorted([(d, x, y, t) for (x, y), (d, t) in worst.items() if d < -TOL])
        wmax = max(max(wrist[s]) for s in wrist)
        emin = min(min(elbow[s]) for s in elbow)
        bmax = max(max(behind[s]) for s in behind)
        flags = []
        if bad: flags.append("PENETRATION")
        if wmax > WRIST_MAX: flags.append("WRIST")
        if emin < ELBOW_MIN: flags.append("ELBOW")
        if sp_max > SPEED_MAX: flags.append("SPEED")
        if jump_max > JUMP_MAX: flags.append("JERK")
        ok_all &= not flags
        print(f"{name:<16} penetrations {len(bad):2d}  wrist max {wmax:5.1f}  elbow min {emin:5.1f}  arm behind max {bmax:5.1f}  "
              f"speed {sp_max:6.0f} ({sp_j})  jump {jump_max:5.0f} ({jump_j})   {'FAIL ' + ','.join(flags) if flags else 'ok'}")
        for d, x, y, t in (bad if a.verbose else bad[:6]):
            print(f"     {x} / {y}: {-d:.3f} deeper than in Idle at t={t:.2f}s")
    print("POSE AUDIT:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
