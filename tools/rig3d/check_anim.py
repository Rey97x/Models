#!/usr/bin/env python3
"""
Animation acceptance check for the rigged nomad GLB (criteria: tools/rig3d/CONTRACT.md).

    python3 tools/rig3d/check_anim.py file.glb [--skeleton character3d/rig/skeleton.json] [--clip NAME] [--verbose]

The check reads ONLY the GLB (nodes, skin, animation samplers) and plays it like a glTF player would: LINEAR
interpolation of translations, slerp of rotations between the 30 fps keys, its own forward kinematics.  The skeleton
file is needed only for the foot geometry that the GLB does not store (heel landmarks and toe end sites).
Every clip is sampled at the keys and at `--sub` positions inside every frame interval.

Per clip (PASS / FAIL, exit code 1 if anything fails):
  timing      30 fps keys, first key at 0, last key at the clip duration, duration a multiple of 1/30 s
  quaternions finite, unit length, neighbouring keys in the same hemisphere, no joint jumps by more than 40 deg / frame
  stance      while a foot is in flat stance (heel, ball and toe tip on the ground) the Foot and ToeBase joints do not
              move by more than 0.0005; a sole point that is on the ground does not slide (heel during the heel-strike
              roll, ball / toe during heel-off).  For in-place clips the declared root motion is applied first:
              Walk -> translation extras.speed * t along +Z, TurnLoop_* -> rotation extras.turnRate * t about the
              vertical axis through Root.  There must always be a supporting foot and the expected number of steps.
  ground      heel landmark, ball and toe tip never below y = -0.001
  knees       IK reach |hip - ankle| <= 0.995 (thigh + shin), the knee never bends backwards, no pops
              (knee angle velocity / acceleration limits)
  loop        looping clips: pose at t = duration equals pose at t = 0 (1e-4) and the motion is smooth across the seam
              (second difference at the seam not larger than inside the clip)
  root        Walk: Root stays at the origin;  Walk_RootMotion: Root z = speed * t, stride = speed * duration, the pose
              tracks are identical to Walk;  Turn360_*: pure yaw 0 -> +-360 deg on Root, monotonic, first and last pose
              equal to the first Idle frame;  TurnLoop_*: no motion on Root, turnRate declared
  footsteps   extras.footsteps match the contact times found in the motion
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from glb import GLB, q_mul, q_rotate  # noqa: E402

FPS = 30.0
TOL_DRIFT = 0.0005          # planted foot drift (model units)
TOL_GROUND = -0.001         # lowest allowed sole point
TOL_REACH = 0.995           # IK reach / leg length
TOL_POSE = 1e-4             # loop seam / start pose equality
EPS_CONTACT = 0.0003        # a sole point lower than this is "on the ground"
KNEE_VEL = 16.0             # deg / frame
KNEE_ACC = 12.5             # deg / frame^2  (human gait reaches ~9-10 in late swing)
SIDES = ("Left", "Right")


# ------------------------------------------------------------------------------------------------ glTF scene
class Scene:
    def __init__(self, g):
        self.g = g
        nodes = g.js["nodes"]
        self.n = len(nodes)
        self.names = [nd.get("name", f"node{i}") for i, nd in enumerate(nodes)]
        self.parent = [-1] * self.n
        for i, nd in enumerate(nodes):
            for c in nd.get("children", []):
                self.parent[c] = i
        self.rest_t = np.array([nd.get("translation", [0, 0, 0]) for nd in nodes], float)
        self.rest_r = np.array([nd.get("rotation", [0, 0, 0, 1]) for nd in nodes], float)
        self.rest_s = np.array([nd.get("scale", [1, 1, 1]) for nd in nodes], float)
        skin = g.js["skins"][0]
        self.joints = list(skin["joints"])
        self.idx = {self.names[j]: j for j in self.joints}
        order, seen = [], set()

        def visit(i):
            if i in seen:
                return
            if self.parent[i] >= 0:
                visit(self.parent[i])
            seen.add(i)
            order.append(i)
        for j in self.joints:
            visit(j)
        self.order = order
        self.clips = []
        for a in g.js.get("animations", []):
            ch = {}
            for c in a["channels"]:
                s = a["samplers"][c["sampler"]]
                ch[(c["target"]["node"], c["target"]["path"])] = (
                    g.accessor(s["input"]).astype(float), g.accessor(s["output"]).astype(float), s.get("interpolation", "LINEAR"))
            self.clips.append(dict(name=a.get("name", "?"), extras=a.get("extras", {}), channels=ch))

    def bind_positions(self):
        t = np.zeros((1,))
        P, _ = self.fk({}, t)
        return {self.names[j]: P[j][0] for j in self.order}

    def sample(self, clip, times):
        """local TRS of the animated nodes at `times` (LINEAR / slerp, like a glTF player)"""
        out = {}
        for (node, path), (kt, kv, interp) in clip["channels"].items():
            i = np.clip(np.searchsorted(kt, times, "right") - 1, 0, max(len(kt) - 2, 0))
            if len(kt) == 1:
                out[(node, path)] = np.repeat(kv[:1], len(times), 0)
                continue
            u = np.clip((times - kt[i]) / np.maximum(kt[i + 1] - kt[i], 1e-12), 0, 1)
            if interp == "STEP":
                u = np.floor(u)
            a, b = kv[i], kv[i + 1]
            if path == "rotation":
                d = np.sum(a * b, -1)
                b = np.where(d[:, None] < 0, -b, b)
                d = np.clip(np.abs(d), -1, 1)
                th = np.arccos(d)
                sn = np.sin(th)
                small = sn < 1e-6
                wa = np.where(small, 1 - u, np.sin((1 - u) * th) / np.where(small, 1, sn))
                wb = np.where(small, u, np.sin(u * th) / np.where(small, 1, sn))
                q = wa[:, None] * a + wb[:, None] * b
                out[(node, path)] = q / np.linalg.norm(q, axis=-1, keepdims=True)
            else:
                out[(node, path)] = a + (b - a) * u[:, None]
        return out

    def fk(self, local, times):
        """world position / rotation of every joint (and its ancestors) for all sample times"""
        T = len(times)
        P, Q = {}, {}
        for i in self.order:
            t = local.get((i, "translation"), np.repeat(self.rest_t[i:i + 1], T, 0))
            r = local.get((i, "rotation"), np.repeat(self.rest_r[i:i + 1], T, 0))
            p = self.parent[i]
            if p < 0 or p not in P:
                P[i], Q[i] = t, r
            else:
                P[i] = P[p] + q_rotate(Q[p], t)
                Q[i] = q_mul(Q[p], r)
        return P, Q


def q_yaw(a):
    a = np.asarray(a, float)
    z = np.zeros_like(a)
    return np.stack([z, np.sin(a / 2), z, np.cos(a / 2)], -1)


def rot_angle(a, b):
    """angle between two rotations (rad)"""
    d = np.clip(np.abs(np.sum(a * b, -1)), 0, 1)
    return 2 * np.arccos(d)


def runs(mask, cyclic=False):
    """[(start, end)] inclusive index runs where mask is True (cyclic: a run may wrap, then end >= len)"""
    n = len(mask)
    out, i = [], 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    if cyclic and len(out) > 1 and out[0][0] == 0 and out[-1][1] == n - 1:
        first = out.pop(0)
        s, _ = out.pop()
        out.append((s, n + first[1]))
    return out


# ------------------------------------------------------------------------------------------------ clip check
class Report:
    def __init__(self, name):
        self.name, self.rows, self.failed = name, [], False

    def add(self, ok, topic, msg):
        self.rows.append(("PASS" if ok else "FAIL", topic, msg))
        self.failed |= not ok

    def info(self, topic, msg):
        self.rows.append(("info", topic, msg))

    def show(self):
        print(f"== {self.name}: {'FAIL' if self.failed else 'PASS'}")
        for st, topic, msg in self.rows:
            print(f"   [{st}] {topic:<11} {msg}")


def clip_kind(name):
    if name == "Idle":
        return "idle"
    if name == "Walk":
        return "walk"
    if name == "Walk_RootMotion":
        return "walk_rm"
    if name.startswith("Turn360"):
        return "turn360"
    if name.startswith("TurnLoop"):
        return "turnloop"
    return "other"


def check_clip(sc, clip, feet, sub, verbose, idle_pose=None, walk_ref=None):
    name, ex = clip["name"], clip["extras"]
    kind = clip_kind(name)
    rep = Report(name)
    loop = bool(ex.get("loop", kind in ("idle", "walk", "walk_rm", "turnloop")))
    sgn = -1.0 if name.endswith("Right") else 1.0
    ix = sc.idx

    # ---------------------------------------------------------------- timing
    durs = [kt[-1] for kt, _, _ in clip["channels"].values()]
    dur = max(durs)
    N = int(round(dur * FPS))
    ok = abs(N / FPS - dur) < 1e-5 and N >= 1
    bad = 0
    for (node, path), (kt, kv, interp) in clip["channels"].items():
        if len(kt) != N + 1 or np.max(np.abs(kt - np.arange(N + 1) / FPS)) > 1e-5 or interp != "LINEAR":
            bad += 1
    rep.add(ok and bad == 0, "timing", f"duration {dur:.4f} s = {N} frames at 30 fps, {len(clip['channels'])} channels"
            + (f", {bad} channels not on the uniform 30 fps grid" if bad else ""))
    for key in ("loop", "rootMotion"):
        if key not in ex:
            rep.add(False, "extras", f"extras.{key} missing")

    # ---------------------------------------------------------------- quaternion sanity
    worst_norm, worst_step, flips, nonfinite = 0.0, 0.0, 0, 0
    worst_joint = ""
    for (node, path), (kt, kv, _) in clip["channels"].items():
        if not np.all(np.isfinite(kv)):
            nonfinite += 1
            continue
        if path != "rotation":
            continue
        worst_norm = max(worst_norm, float(np.max(np.abs(np.linalg.norm(kv, axis=1) - 1))))
        d = np.sum(kv[1:] * kv[:-1], -1)
        flips += int(np.sum(d < 0))
        st = float(np.degrees(np.max(rot_angle(kv[1:], kv[:-1])))) if len(kv) > 1 else 0.0
        if st > worst_step:
            worst_step, worst_joint = st, sc.names[node]
    rep.add(nonfinite == 0 and worst_norm < 1e-4 and flips == 0 and worst_step < 40.0, "quaternions",
            f"|q|-1 max {worst_norm:.1e}, hemisphere flips {flips}, largest step {worst_step:.1f} deg/frame ({worst_joint})"
            + (f", {nonfinite} non-finite channels" if nonfinite else ""))

    # ---------------------------------------------------------------- sampling + FK
    T = N * sub + 1
    times = np.arange(T) / (FPS * sub)
    keys = np.arange(0, T, sub)
    local = sc.sample(clip, times)
    P, Q = sc.fk(local, times)
    root = ix["Root"]

    # declared root motion of in-place clips
    speed, rate = ex.get("speed"), ex.get("turnRate")
    if kind == "walk" and speed is None:
        rep.add(False, "extras", "extras.speed missing")
        speed = 0.0
    if kind == "turnloop" and rate is None:
        rep.add(False, "extras", "extras.turnRate missing")
        rate = 0.0

    # ---------------------------------------------------------------- feet
    # loops are analysed over two periods (the second period continues the declared / accumulated root motion),
    # so that stances that run across the seam are measured in one piece
    reps = 2 if loop else 1
    Te = reps * (T - 1) + 1
    src = np.arange(Te) % (T - 1) if loop else np.arange(T)
    lap = (np.arange(Te) // (T - 1)) if loop else np.zeros(Te, int)
    te = np.arange(Te) / (FPS * sub)
    keys_e = np.arange(0, Te, sub)
    carry_t = np.zeros((Te, 3))
    carry_q = None
    if kind == "walk":
        carry_t[:, 2] = speed * te
    elif kind == "walk_rm":
        carry_t = lap[:, None] * (P[root][-1] - P[root][0])[None, :]
    elif kind == "turnloop":
        carry_q = q_yaw(sgn * math.radians(rate) * te)

    def world(i):
        p, q = P[i][src], Q[i][src]
        if carry_q is not None:
            p, q = q_rotate(carry_q, p), q_mul(carry_q, q)
        return p + carry_t, q

    foot = {}
    for s in SIDES:
        fg = feet[s]
        pf, qf = world(ix[s + "Foot"])
        pt, qt = world(ix[s + "ToeBase"])
        foot[s] = dict(Foot=pf, ToeBase=pt, heel=pf + q_rotate(qf, fg["heel"]), ball=pt + q_rotate(qt, fg["ball"]),
                       tip=pt + q_rotate(qt, fg["tip"]), qf=qf)
    where = min(((float(foot[s][k][:, 1].min()), s, k) for s in SIDES for k in ("heel", "ball", "tip")))
    lowest = where[0]
    rep.add(lowest >= TOL_GROUND, "ground", f"lowest sole point y = {lowest:+.5f} ({where[1]} {where[2]}), limit {TOL_GROUND}")

    drift_key = drift_sub = slide_key = slide_sub = 0.0
    n_flat_runs, con, step_count, stance_frac, contact_starts = {}, {}, {}, {}, {}
    for s in SIDES:
        f = foot[s]
        c = {k: f[k][:, 1] <= EPS_CONTACT for k in ("heel", "ball", "tip")}
        flat = c["heel"] & c["ball"] & c["tip"]
        anyc = c["heel"] | c["ball"] | c["tip"]
        con[s] = dict(flat=flat, any=anyc, **c)
        # flat stance: Foot and ToeBase joints (and the sole) fixed; drift measured from the first key of the stance
        for a, b in runs(flat[keys_e]):
            if b == a:
                continue
            for jn in ("Foot", "ToeBase", "heel", "tip"):
                p = f[jn][a * sub:b * sub + 1]
                d = np.linalg.norm(p - p[0], axis=1)
                drift_sub = max(drift_sub, float(d.max()))
                drift_key = max(drift_key, float(d[::sub].max()))
        # single sole points on the ground must not slide either (heel-strike roll, heel-off about the ball)
        for jn in ("heel", "ball", "tip"):
            for a, b in runs(c[jn][keys_e]):
                if b == a:
                    continue
                p = f[jn][a * sub:b * sub + 1][:, [0, 2]]
                d = np.linalg.norm(p - p[0], axis=1)
                slide_sub = max(slide_sub, float(d.max()))
                slide_key = max(slide_key, float(d[::sub].max()))
        k1 = keys[:-1] if loop else keys                    # one period (keys of the un-extended timeline)
        ak, fk1 = anyc[k1], flat[k1]
        n_flat_runs[s] = len(runs(fk1, loop))
        sw_runs = runs(~ak, loop)
        step_count[s] = len(sw_runs)
        stance_frac[s] = float(np.mean(ak))
        contact_starts[s] = sorted((b + 1) % len(ak) if loop else b + 1 for a, b in sw_runs if loop or b + 1 < len(ak))
    support_gap = int(np.sum(~(con["Left"]["any"] | con["Right"]["any"])))
    rep.add(drift_key <= TOL_DRIFT and drift_sub <= TOL_DRIFT, "stance",
            f"flat-stance drift of Foot / ToeBase: {drift_key:.5f} at keys, {drift_sub:.5f} between keys (limit {TOL_DRIFT})"
            + (f"  [after extras.speed {speed:.4f}]" if kind == "walk" else "")
            + (f"  [after extras.turnRate {rate:.3f} deg/s]" if kind == "turnloop" else ""))
    rep.add(slide_key <= TOL_DRIFT and slide_sub <= TOL_DRIFT, "contact",
            f"sliding of grounded sole points: {slide_key:.5f} at keys, {slide_sub:.5f} between keys (limit {TOL_DRIFT})")
    msg = ", ".join(f"{s[0]}: {step_count[s]} steps, on the ground {100 * stance_frac[s]:.0f} %, flat stances {n_flat_runs[s]}" for s in SIDES)
    okc = support_gap == 0
    if kind == "idle":
        okc &= all(con[s]["flat"].all() for s in SIDES)
        msg += "  (both feet must stay flat)"
    elif kind in ("walk", "walk_rm"):
        okc &= all(step_count[s] == 1 and 0.55 <= stance_frac[s] <= 0.68 and n_flat_runs[s] >= 1 for s in SIDES)
        msg += "  (expected 1 step per foot, stance 55-68 %)"
    elif kind == "turn360":
        tot = sum(step_count.values())
        okc &= all(step_count[s] >= 3 and con[s]["flat"][0] and con[s]["flat"][-1] for s in SIDES) and 6 <= tot <= 12
        msg += f"  (expected 6-12 steps, got {tot}; planted at both ends)"
    elif kind == "turnloop":
        okc &= all(step_count[s] >= 1 and n_flat_runs[s] >= 1 for s in SIDES)
    rep.add(okc, "support", msg + (f", {support_gap} samples without a supporting foot" if support_gap else ""))

    # ---------------------------------------------------------------- knees
    reach_max, knee_rng, back, kv_max, ka_max = 0.0, {}, 0.0, 0.0, 0.0
    for s in SIDES:
        H, K, A = P[ix[s + "UpLeg"]], P[ix[s + "Leg"]], P[ix[s + "Foot"]]
        L = np.linalg.norm(sc.rest_t[ix[s + "Leg"]]) + np.linalg.norm(sc.rest_t[ix[s + "Foot"]])
        reach = np.linalg.norm(A - H, axis=1) / L
        reach_max = max(reach_max, float(reach.max()))
        td = (K - H) / np.linalg.norm(K - H, axis=1, keepdims=True)
        sd = (A - K) / np.linalg.norm(A - K, axis=1, keepdims=True)
        hinge = q_rotate(Q[ix[s + "UpLeg"]], np.array([1.0, 0, 0]))
        sign = np.sign(np.sum(np.cross(td, sd) * hinge, 1))
        ang = np.degrees(np.arccos(np.clip(np.sum(td * sd, 1), -1, 1))) * sign
        back = min(back, float(ang.min()))
        knee_rng[s] = (float(ang.min()), float(ang.max()))
        ak = ang[keys]
        if loop:
            ak = np.concatenate([ak[-2:-1], ak, ak[1:2]])
        d1 = np.diff(ak)
        kv_max = max(kv_max, float(np.abs(d1).max()))
        if len(d1) > 1:
            ka_max = max(ka_max, float(np.abs(np.diff(d1)).max()))
        clip.setdefault("_dbg", {})[s] = dict(reach=reach, knee=ang)
    rep.add(reach_max <= TOL_REACH + 1e-4 and back > -1.0 and kv_max <= KNEE_VEL and ka_max <= KNEE_ACC, "knees",
            f"max reach {reach_max:.4f} (limit {TOL_REACH}), flexion L {knee_rng['Left'][0]:.1f}..{knee_rng['Left'][1]:.1f} "
            f"R {knee_rng['Right'][0]:.1f}..{knee_rng['Right'][1]:.1f} deg, max speed {kv_max:.1f} deg/frame (limit {KNEE_VEL}), "
            f"max accel {ka_max:.1f} deg/frame^2 (limit {KNEE_ACC})")

    # ---------------------------------------------------------------- loop seam
    def pose_diff(la, ia, lb, ib, skip_root_t=False, skip_root=False):
        worst_r = worst_t = 0.0
        for (node, path) in set(la) | set(lb):
            if skip_root and node == root:
                continue
            va = la[(node, path)][ia] if (node, path) in la else (sc.rest_r[node] if path == "rotation" else sc.rest_t[node])
            vb = lb[(node, path)][ib] if (node, path) in lb else (sc.rest_r[node] if path == "rotation" else sc.rest_t[node])
            if path == "rotation":
                worst_r = max(worst_r, float(np.max(rot_angle(va, vb))))
            elif path == "translation" and not (skip_root_t and node == root):
                worst_t = max(worst_t, float(np.max(np.linalg.norm(va - vb, axis=-1))))
        return worst_r, worst_t

    if loop:
        wr, wt = pose_diff(local, 0, local, T - 1, skip_root_t=(kind == "walk_rm"))
        # smoothness across the seam: second differences of the joint positions relative to Root, at the keys
        seam_ratio, seam_abs, worst = 0.0, 0.0, ""
        for j in sc.joints:
            if j == root:
                continue
            rel = (P[j] - P[root])[keys]
            if N < 4:
                break
            acc = np.linalg.norm(rel[2:] - 2 * rel[1:-1] + rel[:-2], axis=1)
            seam = float(np.linalg.norm(rel[1] - 2 * rel[0] + rel[N - 1]))
            inner = float(acc.max())
            if seam > 1.3 * inner + 1e-4 and seam / max(inner, 1e-9) > seam_ratio:
                seam_ratio, worst = seam / max(inner, 1e-9), sc.names[j]
            seam_abs = max(seam_abs, seam - 1.3 * inner)
        rep.add(wr <= TOL_POSE and wt <= TOL_POSE and seam_ratio == 0.0, "loop",
                f"pose(t=end) - pose(t=0): rotation {wr:.1e} rad, translation {wt:.1e} (limit {TOL_POSE}); seam acceleration "
                + ("not larger than inside the clip" if seam_ratio == 0.0 else f"{seam_ratio:.1f}x the largest interior value ({worst})"))

    # ---------------------------------------------------------------- root motion
    rt, rq = P[root], Q[root]
    ident = float(np.degrees(rot_angle(rq, np.array([0, 0, 0, 1.0])).max()))
    if kind in ("idle", "walk", "turnloop", "other"):
        okr = float(np.abs(rt).max()) < 1e-6 and ident < 1e-3
        msg = f"Root stays at the origin (|t| max {np.abs(rt).max():.1e}, rotation max {ident:.1e} deg)"
        if kind == "walk":
            msg += f", speed {speed:.4f} units/s"
            if walk_ref is not None:
                okr &= abs(walk_ref["extras"].get("speed", -1) - speed) < 1e-6
        if kind == "turnloop":
            okr &= rate is not None and rate > 0
            msg += f", turnRate {rate:.3f} deg/s ({'left' if sgn > 0 else 'right'}), {rate * dur:.1f} deg per loop"
        rep.add(okr, "root", msg)
    elif kind == "walk_rm":
        speed = ex.get("speed")
        if speed is None:
            rep.add(False, "extras", "extras.speed missing")
            speed = 0.0
        lin = float(np.abs(rt[:, 2] - rt[0, 2] - speed * times).max())
        lat = float(np.abs(rt[:, :2]).max())
        stride = float(rt[-1, 2] - rt[0, 2])
        okr = lin < 1e-4 and lat < 1e-6 and ident < 1e-3 and abs(stride - speed * dur) < 1e-4
        msg = f"Root z = speed * t within {lin:.1e}; stride {stride:.4f} = speed {speed:.4f} x duration {dur:.4f} = {speed * dur:.4f}"
        if "strideLength" in ex:
            okr &= abs(ex["strideLength"] - stride) < 1e-4
        if walk_ref is not None:
            wl = sc.sample(walk_ref, times)
            wr2, wt2 = pose_diff(local, slice(None), wl, slice(None), skip_root=True)
            okr &= wr2 < 1e-5 and wt2 < 1e-6 and abs(walk_ref["extras"].get("speed", -1) - speed) < 1e-6
            msg += f"; pose tracks identical to Walk ({wr2:.0e} rad, {wt2:.0e})"
        rep.add(okr, "root", msg)
    elif kind == "turn360":
        yaw = np.unwrap(2 * np.arctan2(rq[:, 1], rq[:, 3]))
        yaw = yaw - yaw[0]
        tilt = float(np.abs(rq[:, [0, 2]]).max())
        total = float(np.degrees(yaw[-1]))
        dy = np.diff(yaw) * sgn
        okr = abs(total - sgn * 360.0) < 0.05 and tilt < 1e-6 and float(np.abs(rt).max()) < 1e-6 and float(dy.min()) > -1e-4 \
            and abs(float(np.degrees(2 * np.arctan2(rq[0, 1], rq[0, 3])))) < 1e-3
        msg = (f"Root yaw 0 -> {total:+.3f} deg (pure yaw, monotonic: {float(dy.min()) > -1e-4}), peak rate "
               f"{float(np.degrees(np.abs(np.diff(yaw[keys])).max()) * FPS):.0f} deg/s, Root translation {np.abs(rt).max():.1e}")
        rep.add(okr, "root", msg)
        if idle_pose is not None:
            il = sc.sample(idle_pose, np.zeros(1))
            r0, t0 = pose_diff(local, 0, il, 0, skip_root=True)
            r1, t1 = pose_diff(local, T - 1, il, 0, skip_root=True)
            rep.add(max(r0, r1) <= TOL_POSE and max(t0, t1) <= TOL_POSE, "idle pose",
                    f"first / last pose vs Idle frame 0: rotation {r0:.1e} / {r1:.1e} rad, translation {t0:.1e} / {t1:.1e}")
        else:
            rep.add(False, "idle pose", "no Idle clip in the file to compare the start / end pose with")
    if kind in ("turn360", "turnloop"):
        # comfortable hip rotation: yaw of the foot relative to the pelvis
        worst = 0.0
        hq = world(ix["Hips"])[1]
        fwd_h = q_rotate(hq, np.array([0, 0, 1.0]))
        for s in SIDES:
            fwd_f = q_rotate(foot[s]["qf"], np.array([0, 0, 1.0]))
            d = np.arctan2(fwd_f[:, 0], fwd_f[:, 2]) - np.arctan2(fwd_h[:, 0], fwd_h[:, 2])
            d = np.degrees(np.arctan2(np.sin(d), np.cos(d)))
            worst = max(worst, float(np.abs(d).max()))
        rep.add(worst <= 45.0, "hip yaw", f"largest foot-to-pelvis yaw {worst:.1f} deg (comfortable <= ~40, limit 45)")

    # ---------------------------------------------------------------- footsteps
    fs = ex.get("footsteps")
    if kind != "idle":
        if not fs:
            rep.add(False, "footsteps", "extras.footsteps missing")
        else:
            miss = []
            for st in fs:
                s = "Left" if st.get("foot") == "L" else "Right"
                k = st.get("t", -1) * FPS
                starts = contact_starts[s]
                per = N if loop else 10 ** 9
                if not any(min(abs(k - c), abs(k - c + per), abs(k - c - per)) <= 1.01 for c in starts):
                    miss.append(f"{st.get('foot')}@{st.get('t'):.3f}")
            n_found = sum(len(v) for v in contact_starts.values())
            rep.add(not miss and len(fs) == n_found, "footsteps",
                    f"{len(fs)} footsteps declared, {n_found} contacts found" + (f", unmatched: {', '.join(miss)}" if miss else ", all matched"))

    if verbose:
        print(f"-- {name}: per key  [foot: heel/ball/tip height (1e-3 units), c=contact F=flat]  reach  knee")
        for k in range(N + 1):
            i = k * sub
            row = f"   {k:3d} "
            for s in SIDES:
                f = foot[s]
                tag = "F" if con[s]["flat"][i] else ("c" if con[s]["any"][i] else " ")
                row += (f"| {s[0]} {tag} h{f['heel'][i, 1] * 1e3:6.1f} b{f['ball'][i, 1] * 1e3:6.1f} t{f['tip'][i, 1] * 1e3:6.1f} "
                        f"z{f['Foot'][i, 2]:+.3f} r{clip['_dbg'][s]['reach'][i]:.4f} k{clip['_dbg'][s]['knee'][i]:5.1f} ")
            hp = world(ix["Hips"])[0][i]
            row += f"| hips {hp[0]:+.4f} {hp[1]:.4f} {hp[2]:+.4f}"
            print(row)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb")
    ap.add_argument("--skeleton", default=os.path.join(ROOT, "character3d", "rig", "skeleton.json"))
    ap.add_argument("--clip", help="check only this clip")
    ap.add_argument("--sub", type=int, default=4, help="samples per frame interval (default 4)")
    ap.add_argument("--verbose", action="store_true", help="print a per-key table of the feet / knees")
    a = ap.parse_args()

    g = GLB.load(a.glb)
    if not g.js.get("skins") or not g.js.get("animations"):
        print("FAIL: the file has no skin / no animations")
        sys.exit(1)
    sc = Scene(g)
    import json
    pts = {k: np.array(v, float) for k, v in json.load(open(a.skeleton))["points"].items()}
    bind = sc.bind_positions()
    failed = False
    # the skeleton file must describe the skeleton that is in the GLB
    mism = max((float(np.linalg.norm(bind[n] - pts[n])) for n in bind if n in pts), default=1.0)
    if mism > 1e-4:
        print(f"FAIL: --skeleton {a.skeleton} does not match the joints of the GLB (max difference {mism:.4f})")
        sys.exit(1)
    rest_rot = max(float(np.degrees(rot_angle(sc.rest_r[j], np.array([0, 0, 0, 1.0])))) for j in sc.joints)
    feet = {}
    for s in SIDES:
        need = [s + "Foot", s + "ToeBase", s + "Heel", s + "Toe_End"]
        miss = [n for n in need if n not in pts]
        if miss:
            print(f"FAIL: skeleton file lacks {miss}")
            sys.exit(1)
        F, B = pts[s + "Foot"], pts[s + "ToeBase"]
        feet[s] = dict(heel=pts[s + "Heel"] - F, ball=np.array([0.0, -B[1], 0.0]),
                       tip=np.array([pts[s + "Toe_End"][0] - B[0], -B[1], pts[s + "Toe_End"][2] - B[2]]))
    print(f"{os.path.basename(a.glb)}: {len(sc.joints)} joints, {len(sc.clips)} clips, skeleton {a.skeleton}"
          f" (bind match {mism:.1e}, rest rotations {rest_rot:.1e} deg)")
    by_name = {c["name"]: c for c in sc.clips}
    want = ["Idle", "Walk", "Walk_RootMotion", "Turn360_Left", "Turn360_Right", "TurnLoop_Left", "TurnLoop_Right"]
    for w in want:
        if w not in by_name and not a.clip:
            print(f"== {w}: FAIL\n   [FAIL] missing     clip not found in the file")
            failed = True
    for c in sc.clips:
        if a.clip and c["name"] != a.clip:
            continue
        rep = check_clip(sc, c, feet, max(a.sub, 1), a.verbose, idle_pose=by_name.get("Idle"),
                         walk_ref=by_name.get("Walk") if c["name"] != "Walk" else by_name.get("Walk_RootMotion"))
        rep.show()
        failed |= rep.failed
    print("RESULT:", "FAIL" if failed else "PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
