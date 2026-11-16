"""
Combat clips: attack and dodge / dash, built on the control rig of animate.py (same IK feet, same foot locking).

  build_combat_clips(skeleton_path, cfg, sockets) -> list of clip dicts (same format as animate.build_clips)

`cfg` is <character>/rig/character.json:
    {"weightClass": "heavy" | "light", "weapons": [{"hand": "Right", "type": "sword"}, ...]}
`sockets` come from weapons.compute_socket (where each weapon sits in its hand).

Clips (one-shots that start and end exactly in the Idle start pose, so they blend from / to Idle):
    Attack        feet step into the strike and come back; `extras.hits` = times at which the blade connects
    Dodge_Back    evasive back-step, root motion on Root (-Z);  `extras.distance`
    Dash_Forward  forward burst,      root motion on Root (+Z);  `extras.distance`

Weight classes (the logic the clips follow):
    heavy (armour, long sword) : one big committed diagonal cut with a long wind-up and recovery; the dodge is a
                                 short two-step shuffle, always one foot on the ground, deep landing, slow to recover
    light (cloth, two daggers) : two fast alternating slashes with a quick lunge; the dodge is a long low leap
                                 with a real flight phase and an almost instant recovery

Weapon motion is authored in "strike space": positions in arm lengths from the shoulder of the weapon hand,
x = outwards, y = up, z = forward, expressed in the chest frame, so the torso twist carries the blade.
The arm is solved by two-bone IK towards the grip, the hand is oriented so the blade and its cutting edge point
where the key says.
"""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator

import animate as A
from animate import FPS, DEG, SIDES, SG, Stance, foot_track, controls_from, controls_at, pelvis_height, \
    qmul, qconj, qrot, q_from_m, _norm

# ------------------------------------------------------------------------------------------------ presets
PRESETS = {
    "heavy": dict(
        attack=dict(dur=1.50, step=0.13, lunge=0.060, crouch=0.050, twist=38.0, lean=17.0),
        dodge=dict(dur=1.10, dist=0.40, antic=0.20, crouch=0.035, land=0.055, clear=0.030, flight=False, lean=7.0),
        dash=dict(dur=1.00, dist=0.48, antic=0.18, crouch=0.030, land=0.050, clear=0.028, flight=False, lean=14.0),
    ),
    "light": dict(
        attack=dict(dur=1.00, step=0.16, lunge=0.055, crouch=0.045, twist=30.0, lean=12.0),
        dodge=dict(dur=0.60, dist=0.75, antic=0.14, crouch=0.050, land=0.040, clear=0.050, flight=True, lean=9.0),
        dash=dict(dur=0.5667, dist=0.85, antic=0.13, crouch=0.050, land=0.040, clear=0.045, flight=True, lean=20.0),
    ),
}

ELBOW_POLE = np.array([0.55, -0.60, -0.58])       # strike space: the elbow points out, down and back


def kf(t, keys):
    """smooth, overshoot-free curve through (time, value) keys; value may be a vector"""
    ts = np.array([k[0] for k in keys], float)
    vs = np.array([k[1] for k in keys], float)
    return PchipInterpolator(ts, vs, axis=0)(np.clip(t, ts[0], ts[-1]))


COMBAT_FPS = 120      # key rate of the combat clips: explosive moves keyed at 30 fps let a glTF player's linear
                      # interpolation drag a planted foot by several millimetres between keys


def frames(dur):
    n = int(round(dur * A.FPS)) + 1
    return n, np.arange(n) / A.FPS


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


def weapon_arm(rig, sol, side, socket, t, keys, weight):
    """Overrides Arm / ForeArm / Hand of `side` in the solved pose `sol` (in place).
       keys   : [(time, grip (3), blade (3), edge (3))] in strike space
       weight : (n,) 0 = keep the solver's relaxed arm, 1 = weapon pose"""
    n = len(t)
    sg = SG[side]
    mirror = np.array([sg, 1.0, 1.0])
    L1, L2 = np.linalg.norm(rig.off[side + "ForeArm"]), np.linalg.norm(rig.off[side + "Hand"])
    L = L1 + L2
    grip = kf(t, [(k[0], np.asarray(k[1], float)) for k in keys]) * mirror * L
    blade = _norm(kf(t, [(k[0], _norm(np.asarray(k[2], float))) for k in keys])) * mirror
    edge = kf(t, [(k[0], _norm(np.asarray(k[3], float))) for k in keys]) * mirror
    edge = _norm(edge - np.sum(edge * blade, -1, keepdims=True) * blade)
    Wp, Wq, lq = sol["Wp"], sol["Wq"], sol["lq"]
    Qc, S = Wq["Spine2"], Wp[side + "Arm"]
    grip_w = S + qrot(Qc, grip)
    Yb, Xe = qrot(Qc, blade), qrot(Qc, edge)
    Rw = np.stack([Xe, Yb, np.cross(Xe, Yb)], -1)                       # weapon frame in the world
    if socket is not None:
        Rs, off = np.asarray(socket["rot"], float), np.asarray(socket["offset"], float)
    else:                                                               # bare hand: "blade" = fingers, grip = palm
        f = rig.arm[side].hand_end / np.linalg.norm(rig.arm[side].hand_end)
        x = np.cross(f, [0, 0, 1.0]); x /= np.linalg.norm(x)
        Rs, off = np.stack([x, f, np.cross(x, f)], 1), 0.5 * rig.arm[side].hand_end
    Mh = Rw @ Rs.T                                                      # hand joint orientation in the world
    wrist = grip_w - np.einsum("nij,j->ni", Mh, off)
    v = wrist - S
    d = np.linalg.norm(v, axis=-1)
    dc = np.clip(d, abs(L1 - L2) + 1e-4, L * 0.998)
    u = v / d[:, None]
    cosA = np.clip((L1 ** 2 + dc ** 2 - L2 ** 2) / (2 * L1 * dc), -1, 1)
    sinA = np.sqrt(1 - cosA ** 2)
    pole = qrot(Qc, np.tile(ELBOW_POLE * mirror, (n, 1)))
    w = _norm(pole - np.sum(pole * u, -1, keepdims=True) * u)
    E = S + L1 * (cosA[:, None] * u + sinA[:, None] * w)
    Wr = S + u * dc[:, None]
    du, df = (E - S) / L1, _norm(Wr - E)
    hinge = _norm(np.cross(du, -w))
    E0u, E0f = _arm_frames(rig, side)
    Qu = q_from_m(np.stack([hinge, du, np.cross(hinge, du)], -1) @ E0u.T)
    Qf = q_from_m(np.stack([hinge, df, np.cross(hinge, df)], -1) @ E0f.T)
    Qh = q_from_m(Mh)
    new = {side + "Arm": qmul(qconj(Wq[side + "Shoulder"]), Qu), side + "ForeArm": qmul(qconj(Qu), Qf),
           side + "Hand": qmul(qconj(Qf), Qh)}
    for jn, q in new.items():                                           # blend with the relaxed arm
        q0 = lq[jn]
        q = np.where(np.sum(q0 * q, -1, keepdims=True) < 0, -q, q)
        lq[jn] = _norm((1 - weight)[:, None] * q0 + weight[:, None] * q)
    for jn in (side + "Arm", side + "ForeArm", side + "Hand"):
        par = dict(rig.hier)[jn]
        Wp[jn] = Wp[par] + qrot(Wq[par], rig.off[jn])
        Wq[jn] = qmul(Wq[par], lq[jn])


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


# ------------------------------------------------------------------------------------------------ attack
# strike-space keys: (fraction of the clip, grip, blade direction, edge direction)
SWORD_CUT = [                                   # one committed diagonal cut, from over the shoulder to the far hip
    (0.00, (0.16, -0.86, 0.14), (0.00, 0.20, 1.00), (0.0, -1.0, 0.2)),
    (0.14, (0.30, -0.35, 0.20), (0.15, 0.75, 0.60), (0.0, -0.5, 0.9)),
    (0.29, (0.22, 0.52, -0.08), (0.25, 0.80, -0.55), (0.0, 0.55, 0.85)),      # wound up over the shoulder
    (0.355, (0.05, 0.50, 0.55), (-0.05, 0.75, 0.65), (0.0, -0.65, 0.75)),
    (0.405, (-0.30, -0.05, 0.86), (-0.30, -0.10, 0.95), (-0.15, -0.95, -0.2)),  # contact
    (0.46, (-0.62, -0.50, 0.50), (-0.60, -0.62, 0.50), (-0.5, -0.2, -0.85)),
    (0.60, (-0.60, -0.60, 0.42), (-0.55, -0.66, 0.50), (-0.5, -0.2, -0.85)),    # follow-through held
    (0.84, (0.10, -0.80, 0.20), (0.00, 0.10, 1.00), (0.0, -1.0, 0.1)),
    (1.00, (0.16, -0.86, 0.14), (0.00, 0.20, 1.00), (0.0, -1.0, 0.2)),
]
DAGGER_LEAD = [                                 # fast horizontal slash, outside -> inside
    (0.00, (0.16, -0.86, 0.14), (0.00, 0.20, 1.00), (0.0, -1.0, 0.2)),
    (0.14, (0.62, 0.02, 0.30), (0.75, 0.10, 0.65), (-0.6, 0.0, 0.8)),          # cocked out to the side
    (0.23, (0.10, 0.06, 0.88), (-0.10, 0.08, 1.00), (-1.0, 0.0, -0.1)),        # contact
    (0.29, (-0.52, 0.00, 0.62), (-0.80, 0.02, 0.60), (-0.6, 0.0, -0.8)),
    (0.44, (-0.30, -0.12, 0.60), (-0.55, 0.10, 0.80), (-0.7, 0.0, -0.6)),
    (0.72, (0.14, -0.70, 0.22), (0.00, 0.15, 1.00), (0.0, -1.0, 0.1)),
    (1.00, (0.16, -0.86, 0.14), (0.00, 0.20, 1.00), (0.0, -1.0, 0.2)),
]
DAGGER_FOLLOW = [                               # second slash with the other hand, a beat later
    (0.00, (0.16, -0.86, 0.14), (0.00, 0.20, 1.00), (0.0, -1.0, 0.2)),
    (0.16, (0.30, -0.30, 0.40), (0.10, 0.40, 0.90), (-0.3, -0.6, 0.6)),         # guard in front of the chest
    (0.31, (0.60, 0.10, 0.34), (0.75, 0.15, 0.65), (-0.6, 0.0, 0.8)),
    (0.40, (0.08, 0.12, 0.88), (-0.10, 0.10, 1.00), (-1.0, 0.0, -0.1)),        # contact
    (0.46, (-0.50, 0.04, 0.60), (-0.80, 0.04, 0.60), (-0.6, 0.0, -0.8)),
    (0.60, (-0.25, -0.15, 0.58), (-0.50, 0.10, 0.85), (-0.7, 0.0, -0.6)),
    (0.82, (0.14, -0.72, 0.22), (0.00, 0.15, 1.00), (0.0, -1.0, 0.1)),
    (1.00, (0.16, -0.86, 0.14), (0.00, 0.20, 1.00), (0.0, -1.0, 0.2)),
]


def _scaled(keys, dur):
    return [(k[0] * dur,) + tuple(k[1:]) for k in keys]


def build_attack(rig, idle, cfg, sockets):
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
    tw, ln = P["twist"] * DEG, P["lean"] * DEG
    if heavy:
        # wind-up: turn away and back, load the rear leg; strike: unwind through the target, drop the weight forward
        turn = kf(p, [(0, 0), (0.29, -1.0), (0.405, 0.55), (0.50, 0.85), (0.62, 0.80), (1, 0)])
        fwd = kf(p, [(0, 0), (0.27, -0.35), (0.42, 1.0), (0.62, 0.9), (1, 0)])
        low = kf(p, [(0, 0), (0.27, 0.30), (0.42, 1.0), (0.62, 0.85), (1, 0)])
        lean = kf(p, [(0, 0), (0.27, -0.35), (0.42, 1.0), (0.62, 0.9), (1, 0)])
        step_on, step_off, back_on, back_off = 0.27, 0.385, 0.70, 0.86
        hits = [0.405 * D]
    else:
        turn = kf(p, [(0, 0), (0.14, -0.8), (0.24, 0.6), (0.31, 0.9), (0.40, -0.75), (0.48, -0.9), (0.62, -0.3), (1, 0)])
        fwd = kf(p, [(0, 0), (0.12, 0.1), (0.24, 1.0), (0.48, 0.9), (0.66, 0.4), (1, 0)])
        low = kf(p, [(0, 0), (0.12, 0.5), (0.24, 1.0), (0.48, 0.85), (1, 0)])
        lean = kf(p, [(0, 0), (0.24, 1.0), (0.48, 0.8), (1, 0)])
        step_on, step_off, back_on, back_off = 0.08, 0.215, 0.58, 0.76
        hits = [0.23 * D, 0.40 * D]
    # a strike with the right hand turns the chest to the left (+yaw); `turn` is +1 at the end of the first cut
    C["chest_yaw"] = idle0["chest_yaw"] + (-sg) * tw * turn
    C["pel_yaw"] = idle0["pel_yaw"] + (-sg) * 0.36 * tw * turn
    C["chest_pitch"] = idle0["chest_pitch"] + ln * lean
    C["pel_pitch"] = idle0["pel_pitch"] + 0.25 * ln * lean
    C["chest_roll"] = idle0["chest_roll"] + (-sg) * 4 * DEG * turn * lean
    C["head_pitch"] = idle0["head_pitch"] + 4 * DEG * lean
    C["pel_pos"][:, 2] = idle0["pel_pos"][2] + P["lunge"] * fwd
    C["pel_pos"][:, 0] = idle0["pel_pos"][0] + sg * 0.012 * turn * (turn < 0)        # weight on the rear leg in the wind-up
    # feet: the foot opposite to the weapon steps into the strike (heavy) / the weapon-side foot lunges (light)
    step_foot = other if heavy else lead
    home = _home(idle0)
    st = {}
    for s in SIDES:
        G, yaw = home[s]
        if s == step_foot:
            G1 = G + np.array([0, 0, P["step"]])
            st[s] = [Stance(G, yaw, th=(step_on - 0.05) * D, tl=step_on * D, lift=14 * DEG),
                     Stance(G1, yaw, tc=step_off * D, tf=(step_off + 0.04) * D, th=(back_on - 0.05) * D, tl=back_on * D,
                            land=-10 * DEG, lift=10 * DEG, clear=0.022),
                     Stance(G, yaw, tc=back_off * D, tf=(back_off + 0.04) * D, land=8 * DEG, clear=0.016)]
        else:
            st[s] = [Stance(G, yaw)]
    _apply_feet(rig, C, st, t)
    # free arm (single weapon): counterbalance
    if not dual:
        k = other[0]
        bal = kf(p, [(0, 0), (0.29, 0.6), (0.42, 1.0), (0.62, 0.8), (1, 0)])
        C[k + "_abd"] = idle0[k + "_abd"] + 22 * DEG * bal
        C[k + "_flex"] = idle0[k + "_flex"] - 24 * DEG * bal + 16 * DEG * kf(p, [(0, 0), (0.29, 1.0), (0.42, 0), (1, 0)])
        C[k + "_elbow"] = idle0[k + "_elbow"] + 38 * DEG * bal
    y_des = idle0["pel_pos"][1] - P["crouch"] * low
    pelvis_height(rig, C, y_des, {s: 0.990 for s in SIDES}, periodic=False, sigma=0.6 * A.FPS / 30)
    wgt = kf(p, [(0, 0), (0.07 if not heavy else 0.10, 1), (0.74 if not heavy else 0.86, 1), (1, 0)])
    arms = []
    if heavy or not dual:
        keys = _scaled(SWORD_CUT if heavy else DAGGER_LEAD, D)
        arms.append(lambda sol: weapon_arm(rig, sol, lead, sockets.get(lead), t, keys, wgt))
    else:
        k1, k2 = _scaled(DAGGER_LEAD, D), _scaled(DAGGER_FOLLOW, D)
        arms.append(lambda sol: weapon_arm(rig, sol, lead, sockets.get(lead), t, k1, wgt))
        arms.append(lambda sol: weapon_arm(rig, sol, other, sockets.get(other), t, k2, wgt))
    ex = dict(hits=[round(float(h), 3) for h in hits], weightClass="heavy" if heavy else "light",
              footsteps=[dict(foot=step_foot[0], t=round(step_off * D, 3)), dict(foot=step_foot[0], t=round(back_off * D, 3))])
    return make_clip(rig, "Attack", C, ex, root_motion=False, idle=idle, arms=arms)


# ------------------------------------------------------------------------------------------------ dodge / dash
def build_dash(rig, idle, cfg, name, direction, key):
    """direction = -1 back (Dodge_Back) or +1 forward (Dash_Forward).  Root travels `dist` along Z."""
    heavy = cfg.get("weightClass", "light") == "heavy"
    P = PRESETS["heavy" if heavy else "light"][key]
    D, dist = P["dur"], P["dist"] * direction
    n, t = frames(D)
    p = t / D
    idle0 = controls_at(idle["_C"], 0)
    C = controls_from(rig, idle0, n)
    a = P["antic"]
    if P["flight"]:      # light: push off, both feet in the air, land one after the other
        f1 = dict(off=a - 0.02, lift=a + 0.04, on=0.62, flat=0.68)
        f2 = dict(off=0.26, lift=0.34, on=0.80, flat=0.86)
        go = kf(p, [(0, 0), (a, 0.0), (0.30, 0.22), (0.62, 0.86), (0.84, 1.0), (1, 1.0)])
    else:                # heavy: two quick steps, never airborne
        f1 = dict(off=a - 0.05, lift=a + 0.02, on=0.47, flat=0.53)
        f2 = dict(off=0.44, lift=0.50, on=0.76, flat=0.82)
        go = kf(p, [(0, 0), (a, 0.0), (0.34, 0.30), (0.62, 0.80), (0.82, 1.0), (1, 1.0)])
    z = dist * go
    C["root_pos"][:, 2] = z
    C["pel_pos"][:, 2] = idle0["pel_pos"][2] + z
    # the foot on the far side of the travel leaves last: first foot = right when going back, left when going forward
    first, second = ("Right", "Left") if direction < 0 else ("Left", "Right")
    home = _home(idle0)
    shift = np.array([0, 0, dist])
    land = (12 if direction < 0 else -9) * DEG                 # backwards: toes first; forwards: heel first
    lift = 16 * DEG if direction > 0 else 9 * DEG
    st = {}
    for s, f in ((first, f1), (second, f2)):
        G, yaw = home[s]
        st[s] = [Stance(G, yaw, th=f["off"] * D, tl=f["lift"] * D, lift=lift),
                 Stance(G + shift, yaw, tc=f["on"] * D, tf=f["flat"] * D, land=land, clear=P["clear"])]
    _apply_feet(rig, C, st, t)
    # body: sink before leaving, lean into the travel, absorb the landing
    low = kf(p, [(0, 0), (a, 1.0), (0.40, 0.45), (f2["flat"], 0.0), (1, 0)]) * P["crouch"] \
        + kf(p, [(0, 0), (f1["on"], 0.0), (f2["flat"], 1.0), (min(0.97, f2["flat"] + 0.10), 0.55), (1, 0)]) * P["land"]
    lean = kf(p, [(0, 0), (a, 0.55), (0.42, 1.0), (f1["on"], 0.5), (f2["flat"], 0.25 if heavy else 0.1), (1, 0)])
    pitch = P["lean"] * DEG * lean * (1.0 if direction > 0 else 0.75)       # crouched forward also when backing off
    C["chest_pitch"] = idle0["chest_pitch"] + pitch
    C["pel_pitch"] = idle0["pel_pitch"] + 0.35 * pitch
    C["head_pitch"] = idle0["head_pitch"] - 0.35 * pitch                     # eyes stay on the opponent
    C["pel_pos"][:, 2] += direction * 0.018 * kf(p, [(0, 0), (a, -0.6), (0.40, 1.0), (f2["flat"], 0.2), (1, 0)])
    sway = kf(p, [(0, 0), (f1["lift"], 0.0), (0.5 * (f1["lift"] + f1["on"]), 1.0), (f2["on"], -0.5), (1, 0)])
    C["pel_pos"][:, 0] = idle0["pel_pos"][0] + SG[second] * 0.012 * sway
    C["pel_yaw"] = idle0["pel_yaw"] + SG[first] * direction * 5 * DEG * sway
    C["chest_yaw"] = idle0["chest_yaw"] - SG[first] * direction * 4 * DEG * sway
    for s in SIDES:                                          # arms open for balance, more for the light leap
        k = s[0]
        bal = kf(p, [(0, 0), (a, 0.3), (0.45, 1.0), (f2["flat"], 0.5), (1, 0)])
        amp = 0.6 if heavy else 1.0
        C[k + "_abd"] = idle0[k + "_abd"] + amp * 16 * DEG * bal
        C[k + "_elbow"] = idle0[k + "_elbow"] + amp * 34 * DEG * bal
        C[k + "_flex"] = idle0[k + "_flex"] + amp * direction * -12 * DEG * bal
    y_des = idle0["pel_pos"][1] - low
    pelvis_height(rig, C, y_des, {s: 0.988 for s in SIDES}, periodic=False, sigma=0.5 * A.FPS / 30)
    ex = dict(distance=round(abs(P["dist"]), 4), direction="back" if direction < 0 else "forward",
              weightClass="heavy" if heavy else "light", airborne=bool(P["flight"]),
              footsteps=[dict(foot=first[0], t=round(f1["on"] * D, 3)), dict(foot=second[0], t=round(f2["on"] * D, 3))])
    return make_clip(rig, name, C, ex, root_motion=True, idle=idle)


def build_combat_clips(skeleton_path, cfg, sockets=None):
    rig, base = A.build_clips(skeleton_path, _raw=True)
    idle = base[0]
    sockets = sockets or {}
    base_fps = A.FPS
    A.FPS = COMBAT_FPS
    try:
        return [build_attack(rig, idle, cfg, sockets),
                build_dash(rig, idle, cfg, "Dodge_Back", -1, "dodge"),
                build_dash(rig, idle, cfg, "Dash_Forward", +1, "dash")]
    finally:
        A.FPS = base_fps
