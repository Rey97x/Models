"""
Cloth post-processes applied to finished clips (identity-rest skeleton, tracks as produced by animate.build_clips).

sash_follow_thigh : cloth hanging from the belt in front of a thigh (the Sash chain) is pushed forward by that thigh
                    when it swings forward, instead of being crossed by it.  A sash on the centre line (a tabard
                    between the legs) is pushed by whichever thigh is further forward.
cape_follow_legs  : a cape hanging down the back (the Cape chain) hangs by gravity, turns with the chest and is
                    pushed backwards by knees, calves and heels so the legs never go through it.
"""
import numpy as np
from scipy import ndimage as ndi
from glb import q_mul, q_conj, q_rotate, q_axis_angle, fk

_DONE = set()      # ids of rotation arrays already processed (clips may share track arrays)


def _fwd_angle(d):
    """angle (rad) of a direction in the pelvis YZ plane: 0 = straight down, positive = forward (+Z)."""
    return np.arctan2(d[2], -d[1])


def sash_follow_thigh(clip, joints, gain=0.9, soft=0.06):
    heads = {j["name"]: np.asarray(j["head"], float) for j in joints}
    tr = clip["tracks"]
    if "LeftUpLeg" not in tr or tr["LeftUpLeg"].get("rotation") is None:
        return clip
    n = len(tr["LeftUpLeg"]["t"])
    central = abs(heads["Sash1"][0]) < 0.03
    sides = ("Left", "Right") if central else (("Left",) if heads["Sash1"][0] > 0 else ("Right",))
    thigh0 = {s: heads[s + "Leg"] - heads[s + "UpLeg"] for s in sides}
    sash0 = heads["Sash2"] - heads["Sash1"]
    a_sash0 = _fwd_angle(sash0)
    q_thigh = {s: np.asarray(tr[s + "UpLeg"]["rotation"], float) for s in sides}
    if "Sash1" not in tr or tr["Sash1"].get("rotation") is None:
        tr["Sash1"] = {"t": tr["LeftUpLeg"]["t"], "rotation": np.tile([0, 0, 0, 1.0], (n, 1))}
    if id(tr["Sash1"]["rotation"]) in _DONE:
        return clip
    q_sash = np.asarray(tr["Sash1"]["rotation"], float).copy()
    if len(q_sash) != n:
        return clip
    for i in range(n):
        # the joints are children of Hips, so their local rotations are already in the pelvis frame
        flex = max(_fwd_angle(q_rotate(q_thigh[s][i], thigh0[s])) - _fwd_angle(thigh0[s]) for s in sides)
        cur = _fwd_angle(q_rotate(q_sash[i], sash0)) - a_sash0
        want = gain * flex
        # smooth maximum of (cur, want): no visible snap when the thigh reaches the cloth
        target = soft * np.log(np.exp(cur / soft) + np.exp(want / soft))
        delta = target - cur
        if delta > 1e-5:
            q_sash[i] = q_mul(q_axis_angle([1, 0, 0], -delta), q_sash[i])
    tr["Sash1"] = dict(tr["Sash1"], rotation=q_sash)
    _DONE.add(id(q_sash))
    return clip


def _yaw(q):
    f = q_rotate(q, np.array([0, 0, 1.0]))
    return np.arctan2(f[0], f[2])


def _smooth_up(theta, loop):
    """the cloth starts to lift a little before the leg arrives and settles after it has gone; never below theta."""
    mode = "wrap" if loop else "nearest"
    core = theta[:-1] if loop else theta
    s = ndi.gaussian_filter1d(ndi.maximum_filter1d(core, 5, mode=mode), 1.6, mode=mode)
    out = np.maximum(core, s)
    return np.append(out, out[0]) if loop else out


def cape_follow_legs(clip, joints, ends, lms, margin=0.014, sway=0.35):
    """Cape1 / Cape2 (children of Spine2): gravity hang in the chest's heading + push-back by the legs."""
    tr = clip["tracks"]
    if "Cape1" not in tr or id(tr["Cape1"]["rotation"]) in _DONE:
        return clip
    heads = {j["name"]: np.asarray(j["head"], float) for j in joints}
    n = len(tr["Cape1"]["t"])
    loop = bool(clip.get("extras", {}).get("loop", True))
    d1 = heads["Cape2"] - heads["Cape1"]
    d2 = np.asarray(ends["Cape_End"], float) - heads["Cape2"]
    L1 = np.linalg.norm(d1)
    back = lambda v: np.arctan2(-v[2], -v[1])              # angle from straight down, positive = backwards
    a1, a2 = back(d1), back(d2)
    # leg points the cape has to stay behind: (joint, offset in the joint's rest frame, radius)
    probes = []
    for S in ("Left", "Right"):
        knee, ank = heads[S + "Leg"], heads[S + "Foot"]
        probes += [(S + "Leg", np.zeros(3), 0.050), (S + "Leg", (ank - knee) * 0.5, 0.048),
                   (S + "Foot", np.zeros(3), 0.038), (S + "Foot", np.asarray(lms[S + "Heel"], float) - ank + [0, 0.02, 0], 0.022)]
    th1, th2, yaws, frames = np.zeros(n), np.zeros(n), np.zeros(n), []
    for i in range(n):
        lr = {k: v["rotation"][i] for k, v in tr.items() if v.get("rotation") is not None}
        lp = {k: v["translation"][i] for k, v in tr.items() if v.get("translation") is not None}
        W = fk(joints, lr, lp)
        frames.append(W)
        H, qs = W["Cape1"][0], W["Spine2"][1]
        yaws[i] = _yaw(qs)
        inv = q_axis_angle([0, 1, 0], -yaws[i])
        pts = []
        for jn, off, r in probes:
            p = q_rotate(inv, W[jn][0] + q_rotate(W[jn][1], off) - H)
            pts.append((np.array([0.0, p[1], p[2] - r - margin]), np.hypot(p[1], p[2])))
        need1 = [back(p) - a1 for p, dist in pts if dist < L1 * 1.05 and p[1] < -0.05]
        th1[i] = max(0.0, max(need1, default=0.0))
        c2 = np.array([0.0, -np.cos(a1 + th1[i]) * L1, -np.sin(a1 + th1[i]) * L1])       # Cape2 head in this frame
        need2 = [back(p - c2) - a2 - th1[i] for p, dist in pts if dist >= L1 * 0.9 and (p - c2)[1] < -0.03]
        th2[i] = max(0.0, max(need2, default=0.0))
    # dynamic extras (anticipation / settle of the push, yaw lag) must vanish at both ends of a one-shot so that it
    # starts and ends exactly in the static pose of the neighbouring clip (Turn360 <-> Idle)
    k = np.minimum(np.arange(n), n - 1 - np.arange(n)) / 6.0
    win = np.ones(n) if loop else np.clip(k, 0, 1) ** 2 * (3 - 2 * np.clip(k, 0, 1))
    th1 = th1 + (_smooth_up(th1, loop) - th1) * win
    th2 = th2 + (_smooth_up(th2, loop) - th2) * win
    # a little lag: the cape trails the chest's yaw
    uy = np.unwrap(yaws)
    core = uy[:-1] if loop else uy
    lagged = ndi.gaussian_filter1d(core - np.linspace(0, uy[-1] - uy[0], len(core), endpoint=not loop), 2.5, mode="wrap" if loop else "nearest")
    lag = (lagged - (core - np.linspace(0, uy[-1] - uy[0], len(core), endpoint=not loop))) * sway
    lag = np.append(lag, lag[0]) if loop else lag
    lag = lag * win if np.ptp(uy) > np.radians(3) else np.zeros(n)        # nothing to trail in a standing clip
    q1, q2 = np.zeros((n, 4)), np.zeros((n, 4))
    for i in range(n):
        Q1 = q_mul(q_axis_angle([0, 1, 0], yaws[i] + lag[i]), q_axis_angle([1, 0, 0], th1[i]))
        q1[i] = q_mul(q_conj(frames[i]["Spine2"][1]), Q1)
        q2[i] = q_axis_angle([1, 0, 0], th2[i])
    tr["Cape1"] = dict(tr["Cape1"], rotation=q1)
    tr["Cape2"] = dict(tr["Cape2"], rotation=q2)
    _DONE.add(id(q1))
    return clip
