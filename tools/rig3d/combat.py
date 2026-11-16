"""
Combat clips: attack and dodge / dash, built on the control rig of animate.py (same IK feet, same foot locking).

  build_combat_clips(skeleton_path, cfg, sockets, caps) -> list of clip dicts (same format as animate.build_clips)

`cfg`     : <character>/rig/character.json  {"weightClass": "heavy" | "light", "weapons": [{"hand", "type"}, ...]}
`sockets` : weapons.compute_socket results (where each weapon sits in its hand)
`caps`    : body capsules measured on the mesh (audit_pose.capsules_from_arrays), used to keep arms and blades
            out of the body while the clips are generated

Clips (one-shots that start and end exactly in the Idle start pose, so they blend from / to Idle):
    Attack        feet step into the strike and come back; `extras.hits` = times at which the blade connects
    Dodge_Back    evasive back-step, root motion on Root (-Z);  `extras.distance`
    Dash_Forward  forward burst,      root motion on Root (+Z);  `extras.distance`

Weight classes (the logic the clips follow):
    heavy (armour, long sword) : one committed diagonal cut: step in, long wind-up over the shoulder, hips then chest
                                 then arm, heavy follow-through and a slow recovery; the dodge is a short two-step
                                 retreat behind a guard, always one foot on the ground
    light (cloth, two daggers) : two fast alternating slashes on a quick lunge; the dodge is a low hop with a
                                 real flight phase and an almost instant recovery

How the weapon arm is animated
    * a cut is a swing in a plane (authored in "strike space": arm lengths from the shoulder, x = outwards, y = up,
      z = forward, in the chest frame, so the trunk rotation carries the blade): the hand travels on an arc, the
      blade lies in the swing plane and trails the arm by a wrist angle, the cutting edge always leads
    * the transitions (guard, recovery) are smooth curves through a few poses; the first and last one are the Idle
      pose of the hand, measured from the Idle clip
    * the grip is kept outside the trunk, head and thighs, the blade is turned away from them, the wrist is limited
      to what a wrist can do, and the elbow swivels outwards whenever it would enter the trunk
"""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

import animate as A
from animate import DEG, SIDES, SG, Stance, foot_track, controls_from, controls_at, pelvis_height, \
    qmul, qconj, qrot, q_from_m, _norm
from weapons import SPECS

COMBAT_FPS = 120      # key rate of the combat clips: explosive moves keyed at 30 fps let a glTF player's linear
                      # interpolation drag a planted foot by several millimetres between keys

PRESETS = {
    "heavy": dict(
        attack=dict(dur=1.60, step=0.14, lunge=0.070, crouch=0.050, twist=40.0, hips=22.0, lean=14.0),
        dodge=dict(dur=1.00, dist=0.38, crouch=0.030, land=0.040, clear=0.026, flight=False, lean=7.0, hop=0.0),
        dash=dict(dur=0.95, dist=0.46, crouch=0.030, land=0.040, clear=0.026, flight=False, lean=13.0, hop=0.0),
    ),
    "light": dict(
        attack=dict(dur=1.05, step=0.16, lunge=0.060, crouch=0.045, twist=30.0, hips=14.0, lean=11.0),
        dodge=dict(dur=0.70, dist=0.60, crouch=0.050, land=0.045, clear=0.040, flight=True, lean=10.0, hop=0.022),
        dash=dict(dur=0.63, dist=0.72, crouch=0.050, land=0.045, clear=0.036, flight=True, lean=19.0, hop=0.014),
    ),
}

DEBUG = None                    # set to a dict to capture the arm solver's internals
WRIST_SWING = 52.0 * DEG       # how far the hand may bend away from the forearm
WRIST_TWIST = 80.0 * DEG       # pronation / supination
# strike space: where the elbow goes relative to the shoulder-wrist line.  One reference for every pose: the elbow
# points back when the arm hangs, down when it reaches forward or sideways, out and back when it is raised
POLE = np.array([0.35, -0.75, -0.55])
HAND_MARGIN, ELBOW_MARGIN, BLADE_MARGIN = 0.046, 0.030, 0.014


def kf(t, keys):
    """smooth, overshoot-free curve through (time, value) keys; value may be a vector"""
    ts = np.array([k[0] for k in keys], float)
    vs = np.array([k[1] for k in keys], float)
    return PchipInterpolator(ts, vs, axis=0)(np.clip(t, ts[0], ts[-1]))


def frames(dur):
    n = int(round(dur * A.FPS)) + 1
    return n, np.arange(n) / A.FPS


def _rot_about(v, axis, ang):
    """Rodrigues rotation of vectors v (n,3) about unit axis (3,) by angles ang (n,)"""
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return v * c + np.cross(axis, v) * s + axis * (v @ axis)[:, None] * (1 - c)


# ------------------------------------------------------------------------------------------------ body capsules
class Body:
    """world-space capsules of the solved pose, to keep hands, elbows and blades outside the body"""
    def __init__(self, sol, caps, names):
        self.c = []
        self.radius = {k: v[3] for k, v in caps.items()}
        for name in names:
            if name in caps and caps[name][0] in sol["Wp"]:
                jn, la, lb, r = caps[name]
                self.c.append((sol["Wp"][jn] + qrot(sol["Wq"][jn], np.asarray(la)), sol["Wp"][jn] + qrot(sol["Wq"][jn], np.asarray(lb)), r))

    def push(self, p, margin):
        """(n,3) points moved out of every capsule (+ margin); returns the pushed points"""
        p = p.copy()
        for _ in range(2):
            for a, b, r in self.c:
                ab = b - a
                t = np.clip(np.sum((p - a) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-12), 0, 1)
                c = a + ab * t[:, None]
                d = p - c
                dist = np.linalg.norm(d, axis=-1)
                inside = dist < r + margin
                if inside.any():
                    dirn = np.where(dist[:, None] > 1e-6, d / np.maximum(dist, 1e-6)[:, None], np.array([0, 0, 1.0]))
                    p[inside] = (c + dirn * (r + margin))[inside]
        return p

    def depth(self, p, margin):
        """(n,) how far each point is inside the nearest capsule (+ margin); <= 0 means clear"""
        worst = np.full(len(p), -1e9)
        for a, b, r in self.c:
            ab = b - a
            t = np.clip(np.sum((p - a) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-12), 0, 1)
            worst = np.maximum(worst, r + margin - np.linalg.norm(p - (a + ab * t[:, None]), axis=-1))
        return worst


def _smooth_push(body, p, margin, sigma):
    """push out of the body, low-pass the correction so the hand does not snap, push again"""
    q = body.push(p, margin)
    q = p + gaussian_filter1d(q - p, sigma, axis=0, mode="nearest")
    return body.push(q, margin)


# ------------------------------------------------------------------------------------------------ weapon arm
def _arm_frames(rig, side):
    """rest frames [hinge, bone, hinge x bone] of the upper arm and the forearm (the elbow folds forward)"""
    out = []
    for v in (rig.off[side + "ForeArm"], rig.off[side + "Hand"]):
        a = v / np.linalg.norm(v)
        fold = np.array([0, 0, 1.0]) - a[2] * a
        fold /= np.linalg.norm(fold)
        h = np.cross(a, fold)
        out.append(np.stack([h, a, np.cross(h, a)], 1))
    return out


def _ik(S, Wr, w, L1, L2):
    v = Wr - S
    d = np.linalg.norm(v, axis=-1)
    dc = np.clip(d, 0.52 * (L1 + L2), (L1 + L2) * 0.997)        # never fold the elbow tighter than ~60 degrees
    u = v / d[:, None]
    cosA = np.clip((L1 ** 2 + dc ** 2 - L2 ** 2) / (2 * L1 * dc), -1, 1)
    sinA = np.sqrt(1 - cosA ** 2)
    w = _norm(w - np.sum(w * u, -1, keepdims=True) * u)
    return S + L1 * (cosA[:, None] * u + sinA[:, None] * w), S + u * dc[:, None], u, w


def _swing_twist_clamp(q, axis, swing_max, twist_max):
    """clamp unit quaternions q (n,4) = swing * twist(about axis) to the given limits"""
    v = q[:, :3]
    proj = (v @ axis)[:, None] * axis
    tw = np.concatenate([proj, q[:, 3:4]], 1)
    nrm = np.linalg.norm(tw, axis=1, keepdims=True)
    tw = np.where(nrm > 1e-9, tw / np.maximum(nrm, 1e-9), np.array([0, 0, 0, 1.0]))
    sw = qmul(q, qconj(tw))

    def clamp(x, lim):
        x = np.where(x[:, 3:4] < 0, -x, x)
        ang = 2 * np.arccos(np.clip(x[:, 3], -1, 1))
        k = np.where(ang > lim, lim / np.maximum(ang, 1e-9), 1.0)
        ax = _norm(np.where(np.linalg.norm(x[:, :3], axis=1, keepdims=True) > 1e-9, x[:, :3], np.array([1.0, 0, 0])))
        h = 0.5 * ang * k
        return np.concatenate([ax * np.sin(h)[:, None], np.cos(h)[:, None]], 1)
    return qmul(clamp(sw, swing_max), clamp(tw, twist_max))


def weapon_arm(rig, sol, side, socket, t, traj, weight, body, blade_len=0.0):
    """Overrides Arm / ForeArm / Hand of `side` in the solved pose `sol` (in place).
       traj   : dict(pos, blade, edge) arrays (n,3) in strike space
       weight : (n,) 0 = keep the solver's relaxed arm, 1 = weapon pose"""
    n = len(t)
    sg = SG[side]
    mirror = np.array([sg, 1.0, 1.0])
    L1, L2 = np.linalg.norm(rig.off[side + "ForeArm"]), np.linalg.norm(rig.off[side + "Hand"])
    L = L1 + L2
    sig = 0.02 * A.FPS
    Wp, Wq, lq = sol["Wp"], sol["Wq"], sol["lq"]
    Qc, S = Wq["Spine2"], Wp[side + "Arm"]
    grip = S + qrot(Qc, traj["pos"] * mirror * L)
    blade = qrot(Qc, _norm(traj["blade"]) * mirror)
    edge = qrot(Qc, traj["edge"] * mirror)
    Rs, off = np.asarray(socket["rot"], float), np.asarray(socket["offset"], float)
    # 1. the hand stays outside the body
    grip = _smooth_push(body, grip, HAND_MARGIN, sig)
    # 2. the blade is turned away from the body (checked at its middle and at its tip)
    if blade_len > 0:
        for frac in (0.25, 0.55, 0.8, 1.0):
            pt = grip + blade * blade_len * frac
            blade = _norm(body.push(pt, BLADE_MARGIN) - grip)
        blade = _norm(gaussian_filter1d(blade, sig, axis=0, mode="nearest"))
    edge = _norm(edge - np.sum(edge * blade, -1, keepdims=True) * blade)
    Rw = np.stack([edge, blade, np.cross(edge, blade)], -1)
    Qh = q_from_m(Rw @ Rs.T)                                           # wanted hand orientation
    E0u, E0f = _arm_frames(rig, side)
    a_f = rig.off[side + "Hand"] / L2
    pole0 = qrot(Qc, _norm(traj["pole"]) * mirror)
    alt0 = qrot(Qc, np.tile(np.array([0.9, 0.3, -0.3]) * mirror, (n, 1)))
    swivel = np.zeros(n)

    def ref_pole(u):
        """elbow direction perpendicular to the shoulder-wrist line; a second reference takes over smoothly when
           the wanted pole runs along the arm (no flips)"""
        a = pole0 - np.sum(pole0 * u, -1, keepdims=True) * u
        b = alt0 - np.sum(alt0 * u, -1, keepdims=True) * u
        return _norm(a + 0.10 * b)

    def solve_arm(Qh, swivel):
        wrist = grip - qrot(Qh, np.tile(off, (n, 1)))
        v = wrist - S
        u = _norm(v)
        w0 = ref_pole(u)
        w = w0 * np.cos(swivel)[:, None] + np.cross(u, w0) * (sg * np.sin(swivel))[:, None]     # + = elbow outwards / up
        E, Wr, u, w = _ik(S, wrist, w, L1, L2)
        du, df = (E - S) / L1, _norm(Wr - E)
        hinge = _norm(np.cross(du, -w))
        Qu = q_from_m(np.stack([hinge, du, np.cross(hinge, du)], -1) @ E0u.T)
        Qf = q_from_m(np.stack([hinge, df, np.cross(hinge, df)], -1) @ E0f.T)
        return Qu, Qf, E, Wr

    # 3. wrist limits (the grip stays where it is, the blade gives way) and elbow clearance
    for _ in range(3):
        Qu, Qf, E, Wr = solve_arm(Qh, swivel)
        Qh = qmul(Qf, _swing_twist_clamp(qmul(qconj(Qf), Qh), a_f, WRIST_SWING, WRIST_TWIST))
    # elbow clearance: smallest outward swivel that keeps the elbow and the middle of both arm bones off the trunk
    cand = np.arange(0.0, 85.0, 4.0) * DEG
    r_fore, r_up = body.radius.get(side + "ForeArm", ELBOW_MARGIN), body.radius.get(side + "Arm", ELBOW_MARGIN)
    wrist = grip - qrot(Qh, np.tile(off, (n, 1)))
    u = _norm(wrist - S)
    w0 = ref_pole(u)
    need = np.full(n, cand[-1])
    done = np.zeros(n, bool)
    for dlt in cand:
        w = w0 * math.cos(dlt) + np.cross(u, w0) * (sg * math.sin(dlt))
        Ei, Wri, _, _ = _ik(S, wrist, w, L1, L2)
        dep = np.maximum.reduce([body.depth(Ei, r_fore + 0.006), body.depth(0.5 * (Ei + Wri), r_fore + 0.006),
                                 body.depth(0.5 * (Ei + S), r_up * 0.6)])
        hit = (dep <= 0) & ~done
        need[hit] = dlt
        done |= hit
    swivel = gaussian_filter1d(maximum_filter1d(need, int(0.12 * A.FPS) | 1, mode="nearest"), 0.035 * A.FPS, mode="nearest")
    for _ in range(2):
        Qu, Qf, E, Wr = solve_arm(Qh, swivel)
        Qh = qmul(Qf, _swing_twist_clamp(qmul(qconj(Qf), Qh), a_f, WRIST_SWING, WRIST_TWIST))
    Qu, Qf, E, Wr = solve_arm(Qh, swivel)
    hand_local = _swing_twist_clamp(qmul(qconj(Qf), Qh), a_f, WRIST_SWING, WRIST_TWIST)     # always within the limits
    for i in range(1, n):                                               # the clamp engages with a corner: round it
        if hand_local[i] @ hand_local[i - 1] < 0:
            hand_local[i] = -hand_local[i]
    hand_local = _norm(gaussian_filter1d(hand_local, 0.022 * A.FPS, axis=0, mode="nearest"))
    Qh = qmul(Qf, hand_local)
    if DEBUG is not None:
        DEBUG[side] = dict(need=need, swivel=swivel, E=E, Wr=Wr, S=S.copy(), grip=grip, Qu=Qu, Qf=Qf, Qh=Qh, u=u, w0=w0)
    new = {side + "Arm": qmul(qconj(Wq[side + "Shoulder"]), Qu), side + "ForeArm": qmul(qconj(Qu), Qf),
           side + "Hand": hand_local}
    for jn in (side + "Arm", side + "ForeArm"):                        # take the last corners out of the arm
        q = new[jn]
        for i in range(1, n):
            if q[i] @ q[i - 1] < 0:
                q[i] = -q[i]
        new[jn] = _norm(gaussian_filter1d(q, 0.014 * A.FPS, axis=0, mode="nearest"))
    for jn, q in new.items():                                           # blend with the relaxed arm at the ends
        q0 = lq[jn]
        q = np.where(np.sum(q0 * q, -1, keepdims=True) < 0, -q, q)
        lq[jn] = _norm((1 - weight)[:, None] * q0 + weight[:, None] * q)
    for jn in (side + "Arm", side + "ForeArm", side + "Hand"):
        par = dict(rig.hier)[jn]
        Wp[jn] = Wp[par] + qrot(Wq[par], rig.off[jn])
        Wq[jn] = qmul(Wq[par], lq[jn])


def idle_weapon_pose(rig, idle, side, socket):
    """the Idle start pose of the weapon hand in strike space: (pos, blade, edge)"""
    sol = idle["_sol"]
    sg = SG[side]
    mirror = np.array([sg, 1.0, 1.0])
    L = np.linalg.norm(rig.off[side + "ForeArm"]) + np.linalg.norm(rig.off[side + "Hand"])
    Qc, S = sol["Wq"]["Spine2"][0], sol["Wp"][side + "Arm"][0]
    qh, ph = sol["Wq"][side + "Hand"][0], sol["Wp"][side + "Hand"][0]
    Rs, off = np.asarray(socket["rot"], float), np.asarray(socket["offset"], float)
    loc = lambda v: qrot(qconj(Qc), v) * mirror
    return (loc(ph + qrot(qh, off) - S) / L, loc(qrot(qh, Rs[:, 1])), loc(qrot(qh, Rs[:, 0])))


def cut_keys(t0, tc, te, start, contact, past, r, psi, step=None):
    """dense keys of a planar cut: the hand goes from direction `start` through `contact` and `past` degrees beyond,
       at distances r = (start, contact, end) arm lengths; the blade trails the arm by psi = (start, contact, end) deg"""
    step = step or 1.0 / A.FPS
    s, c = _norm(np.asarray(start, float)), _norm(np.asarray(contact, float))
    nrm = _norm(np.cross(s, c))
    th_c = math.acos(float(np.clip(s @ c, -1, 1)))
    th_e = th_c + past * DEG
    ts = np.arange(t0, te + 0.5 * step, step)
    d1, d2 = tc - t0, te - tc
    th = kf(ts, [(t0, 0.0), (t0 + 0.42 * d1, 0.10 * th_c), (tc, th_c), (tc + 0.5 * d2, th_c + 0.74 * (th_e - th_c)), (te, th_e)])
    rr = kf(ts, [(t0, r[0]), (tc, r[1]), (te, r[2])])
    ps = kf(ts, [(t0, psi[0] * DEG), (tc, psi[1] * DEG), (te, psi[2] * DEG)])
    sv = np.tile(s, (len(ts), 1))
    u = _rot_about(sv, nrm, th)
    blade = _rot_about(sv, nrm, th - ps)
    edge = np.cross(nrm, blade)
    return [(float(tt), u[i] * rr[i], blade[i], edge[i]) for i, tt in enumerate(ts)]


def trajectory(t, keys):
    keys = sorted(keys, key=lambda k: k[0])
    pos = kf(t, [(k[0], np.asarray(k[1], float)) for k in keys])
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    pos = pos * np.maximum(1.0, 0.56 / np.maximum(r, 1e-6))          # the hand never passes through the shoulder
    blade = _norm(kf(t, [(k[0], _norm(np.asarray(k[2], float))) for k in keys]))
    edge = kf(t, [(k[0], _norm(np.asarray(k[3], float))) for k in keys])
    pole = _norm(kf(t, [(k[0], _norm(np.asarray(k[4] if len(k) > 4 else POLE, float))) for k in keys]))
    return dict(pos=pos, blade=blade, edge=edge, pole=pole)


def _smooth_cloth(clip, sigma):
    """fast hands make the hanging strips whip: low-pass the cloth joints (the ends keep the Idle pose)"""
    n = len(next(iter(clip["tracks"].values()))["t"])
    k = np.minimum(np.arange(n), n - 1 - np.arange(n)) / (4 * sigma)
    win = (np.clip(k, 0, 1) ** 2 * (3 - 2 * np.clip(k, 0, 1)))[:, None]
    for jn, tr in clip["tracks"].items():
        if any(c in jn for c in ("Ribbon", "Sash")):
            q = np.asarray(tr["rotation"], float).copy()
            for i in range(1, n):
                if q[i] @ q[i - 1] < 0:
                    q[i] = -q[i]
            s = _norm(gaussian_filter1d(q, sigma, axis=0, mode="nearest"))
            tr["rotation"] = _norm(q * (1 - win) + s * win)


def make_clip(rig, name, C, extras, root_motion, idle, arms=None):
    """animate.make_clip with weapon-arm overrides, then the exact Idle start pose at both ends"""
    orig = A.solve

    def solve(rig_, C_):
        sol = orig(rig_, C_)
        for fn in arms or []:
            fn(sol)
        return sol
    A.solve = solve
    try:
        clip = A.make_clip(rig, name, C, extras, loop=False, root_motion=root_motion, cloth_init=idle["_cloth0"],
                           settle=int(round(0.33 * A.FPS)))
    finally:
        A.solve = orig
    _smooth_cloth(clip, 0.035 * A.FPS)
    for jn, tr in clip["tracks"].items():
        if jn == "Root":
            continue
        q0 = idle["tracks"][jn]["rotation"][0]
        for i in (0, -1):
            tr["rotation"][i] = q0 if np.dot(q0, tr["rotation"][i]) >= 0 else -q0
        if "translation" in tr:
            tr["translation"][0] = tr["translation"][-1] = idle["tracks"][jn]["translation"][0]
    return {k: v for k, v in clip.items() if not k.startswith("_")}


def _apply_feet(rig, C, stances, t):
    for s in SIDES:
        ft = foot_track(rig.leg[s], stances[s], t)
        k = s[0]
        C[k + "_ank"], C[k + "_fyaw"], C[k + "_fpitch"], C[k + "_tpitch"] = ft["ank"], ft["yaw"], ft["pitch"], ft["tpitch"]


def _home(idle0):
    return {s: (idle0[s[0] + "_ank"].copy(), float(idle0[s[0] + "_fyaw"])) for s in SIDES}


def _blade_len(sockets, side):
    if side not in sockets:
        return 0.0
    sp = SPECS[sockets[side]["kind"]]
    return sp["grip"] / 2 + sp["guard_t"] + sp["blade"]


BODY_PARTS = ["Hips", "Spine", "Spine1", "Spine2", "Head", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg"]

# poses in strike space: (grip position, blade direction, edge direction)
GUARD_SWORD = ((0.20, -0.52, 0.46), (-0.05, 0.62, 0.78), (0.0, -0.78, 0.62))       # hand in front of the hip, point up
GUARD_DAGGER = ((0.16, -0.40, 0.50), (-0.12, 0.50, 0.86), (0.0, -0.86, 0.50))


# ------------------------------------------------------------------------------------------------ attack
def build_attack(rig, idle, cfg, sockets, caps):
    heavy = cfg.get("weightClass", "light") == "heavy"
    P = PRESETS["heavy" if heavy else "light"]["attack"]
    D = P["dur"]
    n, t = frames(D)
    p = t / D
    idle0 = controls_at(idle["_C"], 0)
    C = controls_from(rig, idle0, n)
    hands = [w["hand"] for w in cfg.get("weapons", [])] or ["Right"]
    lead = "Right" if "Right" in hands else hands[0]          # the hand that strikes first
    sg = SG[lead]                                             # +1 left, -1 right
    other = "Left" if lead == "Right" else "Right"
    dual = other in hands
    tw, hp, ln = P["twist"] * DEG, P["hips"] * DEG, P["lean"] * DEG
    if heavy:
        # step in while winding up, then hips -> chest -> arm; `turn` is -1 fully wound up, +1 at the end of the cut
        hips = kf(p, [(0, 0), (0.14, -0.15), (0.38, -1.0), (0.455, 0.2), (0.53, 0.9), (0.66, 0.85), (0.84, 0.25), (1, 0)])
        turn = kf(p, [(0, 0), (0.14, -0.10), (0.40, -1.0), (0.485, 0.15), (0.565, 0.9), (0.68, 0.88), (0.86, 0.25), (1, 0)])
        fwd = kf(p, [(0, 0), (0.18, -0.10), (0.38, -0.30), (0.52, 1.0), (0.66, 0.95), (0.86, 0.25), (1, 0)])
        low = kf(p, [(0, 0), (0.20, 0.20), (0.38, 0.35), (0.52, 1.0), (0.66, 0.90), (0.86, 0.25), (1, 0)])
        lean = kf(p, [(0, 0), (0.20, -0.10), (0.38, -0.30), (0.53, 1.0), (0.66, 0.90), (0.86, 0.2), (1, 0)])
        step_on, step_off, back_on, back_off = 0.20, 0.385, 0.72, 0.90
        hits = [0.50 * D]
    else:
        hips = kf(p, [(0, 0), (0.15, -0.8), (0.24, 0.3), (0.31, 0.9), (0.40, -0.2), (0.49, -0.9), (0.60, -0.7), (0.80, -0.15), (1, 0)])
        turn = kf(p, [(0, 0), (0.17, -0.85), (0.26, 0.3), (0.335, 0.9), (0.42, -0.2), (0.515, -0.9), (0.62, -0.75), (0.82, -0.15), (1, 0)])
        fwd = kf(p, [(0, 0), (0.10, 0.05), (0.26, 1.0), (0.53, 0.95), (0.72, 0.4), (1, 0)])
        low = kf(p, [(0, 0), (0.12, 0.45), (0.26, 1.0), (0.53, 0.9), (0.74, 0.35), (1, 0)])
        lean = kf(p, [(0, 0), (0.12, 0.2), (0.26, 1.0), (0.53, 0.85), (0.76, 0.25), (1, 0)])
        step_on, step_off, back_on, back_off = 0.07, 0.215, 0.66, 0.85
        hits = [0.26 * D, 0.45 * D]
    # a strike with the right hand turns the body to the left (+yaw)
    C["pel_yaw"] = idle0["pel_yaw"] + (-sg) * hp * hips
    C["chest_yaw"] = idle0["chest_yaw"] + (-sg) * tw * turn
    C["chest_pitch"] = idle0["chest_pitch"] + ln * lean
    C["pel_pitch"] = idle0["pel_pitch"] + 0.30 * ln * lean
    C["chest_roll"] = idle0["chest_roll"] + (-sg) * 3.5 * DEG * turn * np.clip(lean, 0, 1)
    C["head_pitch"] = idle0["head_pitch"] + 3 * DEG * lean
    C["head_yaw"] = idle0["head_yaw"] + (-sg) * 0.18 * tw * turn                 # the eyes stay on the target
    C["pel_pos"][:, 2] = idle0["pel_pos"][2] + P["lunge"] * fwd
    C["pel_pos"][:, 0] = idle0["pel_pos"][0] + sg * 0.014 * np.minimum(hips, 0.0)     # weight on the rear leg while winding up
    # feet: heavy = the foot opposite to the weapon steps in; light = the weapon-side foot lunges
    step_foot = other if heavy else lead
    home = _home(idle0)
    st = {}
    for s in SIDES:
        G, yaw = home[s]
        if s == step_foot:
            G1 = G + np.array([0, 0, P["step"]])
            st[s] = [Stance(G, yaw, th=(step_on - 0.06) * D, tl=step_on * D, lift=13 * DEG),
                     Stance(G1, yaw, tc=step_off * D, tf=(step_off + 0.035) * D, th=(back_on - 0.06) * D, tl=back_on * D,
                            land=-9 * DEG, lift=9 * DEG, clear=0.020),
                     Stance(G, yaw, tc=back_off * D, tf=(back_off + 0.035) * D, land=7 * DEG, clear=0.015)]
        else:
            st[s] = [Stance(G, yaw)]
    _apply_feet(rig, C, st, t)
    y_des = idle0["pel_pos"][1] - P["crouch"] * low
    pelvis_height(rig, C, y_des, {s: 0.990 for s in SIDES}, periodic=False, sigma=0.6 * A.FPS / 30)

    arms = []
    wgt = kf(p, [(0, 0), (0.12, 1), (0.88, 1), (1, 0)])

    def add(side, keys):
        rest = idle_weapon_pose(rig, idle, side, sockets[side])
        tr = trajectory(t, [(0.0,) + rest] + keys + [(D,) + rest])
        arms.append(lambda sol: weapon_arm(rig, sol, side, sockets[side], t, tr, wgt,
                                           Body(sol, caps, BODY_PARTS), _blade_len(sockets, side)))

    if heavy or not dual:
        if lead in sockets:
            keys = [(0.15 * D,) + GUARD_SWORD,
                    (0.28 * D, (0.52, 0.06, 0.36), (0.10, 0.96, -0.12), (0.0, 0.12, 0.99))] \
                + cut_keys(0.40 * D, 0.50 * D, 0.60 * D, start=(0.46, 0.74, -0.28), contact=(-0.14, 0.00, 1.0), past=52,
                           r=(0.60, 0.93, 0.80), psi=(92, 44, 36)) \
                + [(0.69 * D, (-0.33, -0.56, 0.52), (-0.50, -0.58, 0.64), (-0.6, -0.5, -0.62)),
                   (0.84 * D, (0.06, -0.60, 0.48), (-0.25, 0.05, 0.97), (-0.1, -0.99, 0.02))]
            add(lead, keys)
        # free arm: counterbalance, driven through the relaxed-arm controls
        k = other[0]
        bal = kf(p, [(0, 0), (0.38, 0.7), (0.53, 1.0), (0.68, 0.8), (1, 0)])
        C[k + "_abd"] = idle0[k + "_abd"] + 14 * DEG * bal
        C[k + "_flex"] = idle0[k + "_flex"] + 14 * DEG * kf(p, [(0, 0), (0.38, 1.0), (0.53, -1.2), (0.68, -1.0), (1, 0)])
        C[k + "_elbow"] = idle0[k + "_elbow"] + 34 * DEG * bal
    else:
        # two daggers: lead hand slashes outside -> inside, the other hand follows a beat later
        k1 = [(0.09 * D,) + GUARD_DAGGER] \
            + cut_keys(0.17 * D, 0.26 * D, 0.335 * D, start=(0.90, 0.26, 0.20), contact=(0.02, 0.06, 1.0), past=46,
                       r=(0.62, 0.93, 0.78), psi=(86, 48, 40)) \
            + [(0.45 * D, (0.26, -0.40, 0.36), (0.05, 0.45, 0.89), (0.0, -0.89, 0.45)),
               (0.70 * D,) + GUARD_DAGGER, (0.86 * D, (0.18, -0.62, 0.34), (-0.05, 0.30, 0.95), (0.0, -0.95, 0.30))]
        k2 = [(0.12 * D,) + GUARD_DAGGER, (0.26 * D, (0.30, -0.22, 0.42), (0.20, 0.50, 0.84), (-0.2, -0.80, 0.56))] \
            + cut_keys(0.36 * D, 0.45 * D, 0.525 * D, start=(0.90, 0.28, 0.22), contact=(0.02, 0.08, 1.0), past=46,
                       r=(0.62, 0.93, 0.78), psi=(86, 48, 40)) \
            + [(0.64 * D, (0.24, -0.40, 0.38), (0.05, 0.45, 0.89), (0.0, -0.89, 0.45)),
               (0.80 * D,) + GUARD_DAGGER, (0.92 * D, (0.18, -0.66, 0.30), (-0.05, 0.28, 0.96), (0.0, -0.96, 0.28))]
        add(lead, k1)
        add(other, k2)
    ex = dict(hits=[round(float(h), 3) for h in hits], weightClass="heavy" if heavy else "light",
              footsteps=[dict(foot=step_foot[0], t=round(step_off * D, 3)), dict(foot=step_foot[0], t=round(back_off * D, 3))])
    return make_clip(rig, "Attack", C, ex, root_motion=False, idle=idle, arms=arms)


# ------------------------------------------------------------------------------------------------ dodge / dash
def build_dash(rig, idle, cfg, sockets, caps, name, direction, key):
    """direction = -1 back (Dodge_Back) or +1 forward (Dash_Forward).  Root travels `dist` along Z."""
    heavy = cfg.get("weightClass", "light") == "heavy"
    P = PRESETS["heavy" if heavy else "light"][key]
    D, dist = P["dur"], P["dist"] * direction
    n, t = frames(D)
    p = t / D
    idle0 = controls_at(idle["_C"], 0)
    C = controls_from(rig, idle0, n)
    if P["flight"]:      # light: sink, push off with both legs, short low flight, land rear foot first
        a = 0.17
        f1 = dict(off=0.15, lift=0.22, on=0.60, flat=0.66)
        f2 = dict(off=0.21, lift=0.29, on=0.69, flat=0.75)
        go = kf(p, [(0, 0), (0.10, 0.0), (a, 0.02), (0.29, 0.20), (0.60, 0.84), (0.75, 0.985), (0.86, 1.0), (1, 1.0)])
    else:                # heavy: two deliberate steps, never airborne
        a = 0.16
        f1 = dict(off=0.12, lift=0.18, on=0.44, flat=0.50)
        f2 = dict(off=0.42, lift=0.48, on=0.72, flat=0.78)
        go = kf(p, [(0, 0), (0.10, 0.0), (a, 0.02), (0.44, 0.52), (0.72, 0.96), (0.84, 1.0), (1, 1.0)])
    z = dist * go
    C["root_pos"][:, 2] = z
    C["pel_pos"][:, 2] = idle0["pel_pos"][2] + z
    first, second = ("Right", "Left") if direction < 0 else ("Left", "Right")
    home = _home(idle0)
    shift = np.array([0, 0, dist])
    land = (11 if direction < 0 else -8) * DEG                 # backwards: toes first; forwards: heel first
    lift = (14 if direction > 0 else 8) * DEG
    st = {}
    for s, f in ((first, f1), (second, f2)):
        G, yaw = home[s]
        st[s] = [Stance(G, yaw, th=f["off"] * D, tl=f["lift"] * D, lift=lift),
                 Stance(G + shift, yaw, tc=f["on"] * D, tf=f["flat"] * D, land=land, clear=P["clear"])]
    _apply_feet(rig, C, st, t)
    # body: sink before leaving, lean into the push, small ballistic rise in flight, absorb the landing, stand up
    tB = 0.5 * (f1["lift"] + f2["on"])                     # middle of the travel
    tC = f2["flat"]                                        # both feet down again
    low = P["crouch"] * kf(p, [(0, 0), (a, 1.0), (tB, 0.50), (f2["on"], 0.35), (1, 0)]) \
        + P["land"] * kf(p, [(0, 0), (f2["on"], 0.0), (tC + 0.03, 1.0), (min(0.96, tC + 0.14), 0.45), (1, 0)])
    if P["hop"]:
        low = low - P["hop"] * kf(p, [(0, 0), (f2["lift"], 0.0), (0.5 * (f2["lift"] + f1["on"]), 1.0), (f1["on"], 0.0), (1, 0)])
    lean = kf(p, [(0, 0), (a, 0.7), (tB, 1.0), (f2["on"], 0.6), (tC + 0.03, 0.5 if heavy else 0.4), (1, 0)])
    pitch = P["lean"] * DEG * lean * (1.0 if direction > 0 else 0.8)        # guarded and slightly forward both ways
    C["chest_pitch"] = idle0["chest_pitch"] + pitch
    C["pel_pitch"] = idle0["pel_pitch"] + 0.40 * pitch
    C["head_pitch"] = idle0["head_pitch"] - 0.30 * pitch                    # eyes stay on the opponent
    # the pelvis leads the push (hips go first), then the feet catch up
    C["pel_pos"][:, 2] += direction * 0.022 * kf(p, [(0, 0), (a, 0.5), (tB, 1.0), (f2["on"], 0.3), (tC, 0.0), (1, 0)])
    sway = kf(p, [(0, 0), (f1["lift"], 0.0), (0.5 * (f1["lift"] + f1["on"]), 0.8), (f1["flat"], -0.5), (f2["flat"], 0.0), (1, 0)])
    C["pel_pos"][:, 0] = idle0["pel_pos"][0] + SG[second] * 0.010 * sway
    C["pel_yaw"] = idle0["pel_yaw"] + SG[first] * direction * 5 * DEG * sway
    C["chest_yaw"] = idle0["chest_yaw"] - SG[first] * direction * 3 * DEG * sway
    y_des = idle0["pel_pos"][1] - low
    pelvis_height(rig, C, y_des, {s: 0.988 for s in SIDES}, periodic=False, sigma=0.5 * A.FPS / 30)
    # arms: weapon hands come to a compact guard, a bare hand only balances a little
    arms = []
    wgt = kf(p, [(0, 0), (0.20, 1), (0.80, 1), (1, 0)])
    for s in SIDES:
        if s in sockets:
            guard = GUARD_SWORD if sockets[s]["kind"] == "sword" else GUARD_DAGGER
            g2 = (tuple(np.add(guard[0], (0.03, 0.05 if direction < 0 else -0.04, -0.06))), guard[1], guard[2])
            rest = idle_weapon_pose(rig, idle, s, sockets[s])
            tg = max(a + 0.10, 0.30)
            tr = trajectory(t, [(0.0,) + rest, (tg * D,) + guard, (f2["on"] * D,) + g2, (min(0.90, f2["flat"] + 0.08) * D,) + guard, (D,) + rest])
            arms.append(lambda sol, s=s, tr=tr: weapon_arm(rig, sol, s, sockets[s], t, tr, wgt,
                                                           Body(sol, caps, BODY_PARTS), _blade_len(sockets, s)))
        else:
            k = s[0]
            bal = kf(p, [(0, 0), (a, 0.4), (f2["lift"] + 0.1, 1.0), (f2["flat"], 0.5), (1, 0)])
            C[k + "_abd"] = idle0[k + "_abd"] + 9 * DEG * bal
            C[k + "_elbow"] = idle0[k + "_elbow"] + 32 * DEG * bal
            C[k + "_flex"] = idle0[k + "_flex"] + 6 * DEG * bal
    ex = dict(distance=round(abs(P["dist"]), 4), direction="back" if direction < 0 else "forward",
              weightClass="heavy" if heavy else "light", airborne=bool(P["flight"]),
              footsteps=[dict(foot=first[0], t=round(f1["on"] * D, 3)), dict(foot=second[0], t=round(f2["on"] * D, 3))])
    return make_clip(rig, name, C, ex, root_motion=True, idle=idle, arms=arms)


def build_combat_clips(skeleton_path, cfg, sockets=None, caps=None):
    rig, base = A.build_clips(skeleton_path, _raw=True)
    idle = base[0]
    sockets, caps = sockets or {}, caps or {}
    base_fps = A.FPS
    A.FPS = COMBAT_FPS
    try:
        return [build_attack(rig, idle, cfg, sockets, caps),
                build_dash(rig, idle, cfg, sockets, caps, "Dodge_Back", -1, "dodge"),
                build_dash(rig, idle, cfg, sockets, caps, "Dash_Forward", +1, "dash")]
    finally:
        A.FPS = base_fps
