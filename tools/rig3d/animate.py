#!/usr/bin/env python3
"""
Procedural, IK-driven animation clips for the rigged nomad (see CONTRACT.md).

    build_clips(skeleton_path) -> [ {"name", "duration", "tracks": {joint: {"t", "rotation"?, "translation"?}}, "extras"} ]

Clips: Idle, Walk, Walk_RootMotion, Turn360_Left, Turn360_Right, TurnLoop_Left, TurnLoop_Right.  30 fps, numpy only,
deterministic.  All geometry (bone lengths, foot sole points, rest directions) is read from the skeleton file, the
left and the right side are solved independently, so the skeleton may be asymmetric.

How a clip is made
  1. a small "control rig" is animated in WORLD space: Root (ground position + yaw), pelvis position / orientation,
     chest and head orientation, arm directions, and for every foot a list of footprints with contact timing
     (FootTrack: land roll -> flat -> heel-off roll about the ball -> swing arc to the next footprint);
  2. the pelvis height is the highest one that keeps both legs inside a prescribed reach profile (so the knee never
     locks and the knee angle of the supporting leg follows a gait-like curve);
  3. the legs are solved analytically (two-bone IK, knee pole from foot / pelvis heading), the feet get their world
     orientation, the toes stay flat on the ground while the heel rolls off;
  4. world orientations are converted to local quaternions (identity-rest skeleton: local = conj(parent) * world);
  5. the cloth helper joints (wrist ribbons, sash) are baked from a small damped pendulum simulation that is run in
     world space (several cycles for loops, so the seam is periodic).
Planted feet are therefore world-fixed by construction; tools/rig3d/check_anim.py verifies it from the GLB alone.

    python3 tools/rig3d/animate.py [skeleton.json]        prints a design / diagnostics summary
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from skeleton_def import HIERARCHY, load_skeleton  # noqa: E402

FPS = 30
DEG = math.pi / 180.0
SIDES = ("Left", "Right")
SG = {"Left": 1.0, "Right": -1.0}          # sign of the side's +X offset
GRAVITY = 9.81 / 1.8                       # model units / s^2  (1 unit ~ 1.8 m)

# ------------------------------------------------------------------------------------------------ tuning constants
REACH_HARD = 0.9945        # absolute IK reach limit (fraction of thigh + shin length); the contract allows 0.995
KNEE_POLE = 0.3            # knee heading = foot yaw + KNEE_POLE * (pelvis yaw - foot yaw)

# relaxed arm hang (angles in degrees; absolute directions, so independent of the A-pose angles of the skeleton)
ARM = dict(abd=11.0, flex=3.0, elbow=17.0, elbow_in=18.0, clav_up=-2.5, clav_fwd=0.0)

IDLE = dict(frames=150, knee_reach=0.9915, sway=0.0085, breath=1.1)

WALK = dict(
    frames=34,             # one cycle = 2 steps
    f_flat=3, f_heel_off=12, f_lift=21,     # frames after heel strike: foot flat, heel-off, toe-off (stance 61.8 %)
    step_frac=0.39,        # step length / body height (reduced automatically if the legs are too short for it)
    bob_max=0.0245,        # largest accepted pelvis bob (peak to peak)
    width=0.70,            # lateral foot placement as a fraction of the rest stance width
    toe_out=4.0,           # deg
    land_pitch=-13.0, lift_pitch=62.0,      # foot pitch at heel strike / toe-off (deg, + = heel up)
    toe_pitch=22.0, f_toe=4,                # the toes peel off during the last f_toe frames of the stance
    swing_mid=(0.5, 6.0, -110.0),           # foot pitch knot inside the swing: (fraction, deg, deg per unit fraction)
    knee_toe_off=38.0, knee_swing=61.0,     # wanted knee flexion at toe-off / peak in swing (deg): the foot roll and
                                            # the swing lift are solved for them
    ease_end=0.94,                          # < 1: the swing foot reaches its footprint early and waits (leg retraction)
    clear=0.016,           # swing clearance of the lowest sole point
    reach=0.9938, reach_load=0.0050,        # stance reach profile: almost straight, small give after heel strike
    pel_yaw=5.0, pel_list=3.3, pel_lat=0.013, pel_pitch=2.0, pel_fore=0.004,
    chest_yaw=3.6, chest_pitch=4.5, chest_roll=0.9,
    arm_swing=17.0, arm_fwd=3.0, elbow_swing=15.0,
)

TURN = dict(
    step=11, swing=8,      # frames between successive lifts / swing duration (TurnLoop and the body of Turn360)
    frames=100,            # Turn360 duration
    lifts=(3, 17, 29, 40, 52, 63, 75, 86),  # Turn360: frames at which the feet leave the ground (inner foot first)
    ramp_in=0.36, ramp_out=0.56,            # s, ease in / out of the yaw rate
    bias=10.0,             # deg, the inner foot is planted this much more "open" than the centre of its yaw window
    heel_off=3, flat=2,    # frames of heel-off roll before a lift / of toe-first landing roll after a plant
    lift_pitch=15.0, land_pitch=9.0,
    clear=0.017, widen=0.006, bulge=0.008,
    reach=0.9905, crouch=0.006, shift=0.30, follow=0.45,
    pel_follow=0.5, chest_lead=6.0, head_lead=13.0, lean=1.4,
)


# ================================================================================================ small math kit
def _norm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def qmul(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], -1)


def qconj(q):
    return np.asarray(q, float) * np.array([-1.0, -1.0, -1.0, 1.0])


def qrot(q, v):
    q, v = np.asarray(q, float), np.asarray(v, float)
    u, w = q[..., :3], q[..., 3:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def qaa(axis, ang):
    """axis (3,) or (...,3) unit, ang (...) radians -> quaternion xyzw"""
    ang = np.asarray(ang, float)
    axis = np.asarray(axis, float)
    s = np.sin(ang / 2)[..., None]
    return np.concatenate([axis * s, np.cos(ang / 2)[..., None]], -1)


def qx(a):
    return qaa(np.array([1.0, 0, 0]), a)


def qy(a):
    return qaa(np.array([0, 1.0, 0]), a)


def qz(a):
    return qaa(np.array([0, 0, 1.0]), a)


def q_arc(a, b):
    """shortest rotation taking unit vector a onto unit vector b (broadcasting)"""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    c = np.cross(a, b)
    d = np.sum(a * b, -1, keepdims=True)
    q = np.concatenate([c, 1.0 + d], -1)
    bad = q[..., 3] < 1e-9                       # antiparallel: any perpendicular axis
    if np.any(bad):
        p = np.cross(a, np.array([1.0, 0, 0]))
        p = np.where(np.linalg.norm(p, axis=-1, keepdims=True) < 1e-6, np.cross(a, np.array([0, 1.0, 0])), p)
        q = np.where(bad[..., None], np.concatenate([_norm(p), np.zeros_like(d)], -1), q)
    return _norm(q)


def q_from_m(m):
    """rotation matrices (...,3,3) -> quaternions (...,4)"""
    m = np.asarray(m, float)
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    c = np.stack([
        np.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1], 1 + m00 + m11 + m22], -1),
        np.stack([1 + m00 - m11 - m22, m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0], m[..., 2, 1] - m[..., 1, 2]], -1),
        np.stack([m[..., 0, 1] + m[..., 1, 0], 1 - m00 + m11 - m22, m[..., 1, 2] + m[..., 2, 1], m[..., 0, 2] - m[..., 2, 0]], -1),
        np.stack([m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1 - m00 - m11 + m22, m[..., 1, 0] - m[..., 0, 1]], -1),
    ], -2)                                                          # (...,4 candidates,4)
    key = np.stack([c[..., 0, 3], c[..., 1, 0], c[..., 2, 1], c[..., 3, 2]], -1)
    pick = np.argmax(key, -1)
    q = np.take_along_axis(c, pick[..., None, None], -2)[..., 0, :]
    return _norm(q)


def qpow(q, f):
    """fraction f of rotation q (shortest way)"""
    q = np.asarray(q, float)
    q = np.where(q[..., 3:4] < 0, -q, q)
    vn = np.linalg.norm(q[..., :3], axis=-1)
    ang = 2.0 * np.arctan2(vn, q[..., 3])
    axis = q[..., :3] / np.maximum(vn, 1e-12)[..., None]
    return qaa(axis, ang * f)


def yaw_of(q):
    f = qrot(q, np.array([0, 0, 1.0]))
    return np.unwrap(np.arctan2(f[..., 0], f[..., 2]))


def continuous(q):
    """flip signs so that neighbouring keys are in the same hemisphere"""
    q = np.array(q, float)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def smootherstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)


def hermite(u, p0, m0, p1, m1):
    u2, u3 = u * u, u * u * u
    return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1


def pspline(tk, vk, t, period=1.0, flat=False):
    """periodic C1 cubic through the knots (tk increasing inside one period); flat=True -> zero tangents (holds)"""
    tk, vk = np.asarray(tk, float), np.asarray(vk, float)
    n = len(tk)
    te = np.concatenate([[tk[-1] - period], tk, [tk[0] + period]])
    ve = np.concatenate([[vk[-1]], vk, [vk[0]]])
    m = np.zeros(n) if flat else (ve[2:] - ve[:-2]) / (te[2:] - te[:-2])
    tk2, vk2, m2 = np.append(tk, tk[0] + period), np.append(vk, vk[0]), np.append(m, m[0])
    t = np.mod(np.asarray(t, float) - tk[0], period) + tk[0]
    i = np.clip(np.searchsorted(tk2, t, "right") - 1, 0, n - 1)
    h = tk2[i + 1] - tk2[i]
    u = (t - tk2[i]) / h
    return hermite(u, vk2[i], m2[i] * h, vk2[i + 1], m2[i + 1] * h)


def smin(a, b, k):
    """smooth minimum with compact support: exact min when |a - b| >= k, C1 blend otherwise (never above min)"""
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def gauss_smooth(y, sigma, periodic):
    """gaussian smoothing along axis 0; periodic: y holds ONE period (no repeated end sample);
    otherwise the ends are padded with the end values and the result is corrected to keep them exactly"""
    y = np.asarray(y, float)
    if sigma <= 0:
        return y.copy()
    hw = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-hw, hw + 1) / sigma) ** 2)
    k /= k.sum()
    n = len(y)
    out = np.zeros_like(y)
    for j, w in zip(range(-hw, hw + 1), k):
        idx = np.arange(n) + j
        idx = np.mod(idx, n) if periodic else np.clip(idx, 0, n - 1)
        out += w * y[idx]
    if not periodic:
        f = smoothstep(np.arange(n) / max(2.0 * hw + 2, 1.0))
        f = f.reshape((n,) + (1,) * (y.ndim - 1))
        out = out + (1 - f) * (y[0] - out[0]) + (1 - f[::-1]) * (y[-1] - out[-1])
    return out


# ================================================================================================ skeleton geometry
def _rest_frame(v):
    """orthonormal frame [x, a, x^a] of a bone with rest vector v (hinge axis = world X made perpendicular)"""
    a = v / np.linalg.norm(v)
    x = np.array([1.0, 0, 0]) - a[0] * a
    x /= np.linalg.norm(x)
    return np.stack([x, a, np.cross(x, a)], 1)


class LegGeom:
    def __init__(self, rig, side):
        P = rig.P
        up, lo, ft, toe = side + "UpLeg", side + "Leg", side + "Foot", side + "ToeBase"
        self.names = (up, lo, ft, toe)
        self.off_up = rig.off[up]
        self.L1, self.L2 = np.linalg.norm(rig.off[lo]), np.linalg.norm(rig.off[ft])
        self.L = self.L1 + self.L2
        self.E0t, self.E0s = _rest_frame(rig.off[lo]), _rest_frame(rig.off[ft])
        A = P[ft]
        self.ank = A.copy()                                     # rest ankle (ground is y = 0)
        heel = P.get(side + "Heel", A + np.array([0, -A[1], -0.035]))
        tend = P.get(side + "Toe_End", P[toe] + np.array([0, 0, 0.045]))
        # sole points: foot-local (relative to the ankle) / toe-local (relative to the ball joint), on the ground
        self.heel = np.array([heel[0] - A[0], -A[1], heel[2] - A[2]])
        self.ball = P[toe] - A                                  # ball (ToeBase) joint
        self.ballsole = np.array([self.ball[0], -A[1], self.ball[2]])
        self.toe_h = float(P[toe][1])                           # height of the ball joint above the ground
        self.tip = np.array([tend[0] - P[toe][0], -P[toe][1], tend[2] - P[toe][2]])
        self.mid = 0.5 * (self.heel + self.tip + self.ball)     # middle of the sole (support centre), foot-local
        self.mid[1] = -A[1]


class ArmGeom:
    def __init__(self, rig, side):
        arm, fore, hand = side + "Arm", side + "ForeArm", side + "Hand"
        self.a_u = rig.off[fore] / np.linalg.norm(rig.off[fore])       # rest direction of the upper arm
        self.a_f = rig.off[hand] / np.linalg.norm(rig.off[hand])       # rest direction of the forearm
        e = rig.P.get(side + "Hand_End", rig.P[hand] + 0.09 * self.a_f)
        self.hand_end = e - rig.P[hand]


class Rig:
    def __init__(self, path):
        joints, ends, lms = load_skeleton(path)
        self.P = {j["name"]: np.array(j["head"], float) for j in joints}
        for k, v in list(ends.items()) + list(lms.items()):
            self.P[k] = np.array(v, float)
        self.off = {n: self.P[n] - (self.P[p] if p else 0.0) for n, p in HIERARCHY}
        self.height = float(self.P["Head_End"][1]) if "Head_End" in self.P else float(self.P["Head"][1]) + 0.11
        self.leg = {s: LegGeom(self, s) for s in SIDES}
        self.arm = {s: ArmGeom(self, s) for s in SIDES}


# ================================================================================================ foot tracks
def _rx(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c])


def _ry(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c])


class Stance:
    """one footprint: flat ankle position G (y = rest ankle height), yaw, and its timing
       tc contact, tf foot flat, th heel-off (the foot starts to roll about the ball), tt toe roll (the toes start
       to peel off, pivot = toe tip), tl lift-off.
       land < 0: heel strike (pivot about the heel), land > 0: toe-first landing (pivot about the ball), 0: flat.
       lift >= 0: foot pitch reached at lift-off, toe >= 0: toe pitch reached at lift-off (0 = toes flat until the
       foot leaves the ground).  clear / bulge / ease_end shape the swing that LEADS INTO this stance."""
    def __init__(self, G, yaw, tc=-1e9, tf=None, th=1e9, tl=1e9, land=0.0, lift=0.0, toe=0.0, tt=None, clear=0.02,
                 bulge=0.0, ease_end=1.0, mid=None, clear2=0.0):
        self.G = np.asarray(G, float)
        self.yaw, self.tc, self.tf, self.th, self.tl = yaw, tc, (tc if tf is None else tf), th, tl
        self.tt = tl if tt is None else tt
        self.land, self.lift, self.toe, self.clear, self.bulge, self.ease_end = land, lift, toe, clear, bulge, ease_end
        self.clear2 = clear2    # extra ankle lift in the first part of the swing (heel kicks up behind)
        self.mid = mid          # optional (u, pitch, slope d pitch / d u) knot of the foot pitch inside the swing


def _foot_rise(g, phi, phit, soft):
    """height of the ankle above the lowest sole point for foot pitch phi and toe pitch phit (world pitches)"""
    c, s = math.cos(phi), math.sin(phi)
    yb = g.ball[1] * c - g.ball[2] * s
    ct, st = math.cos(phit), math.sin(phit)
    vals = (g.heel[1] * c - g.heel[2] * s, g.ballsole[1] * c - g.ballsole[2] * s,
            yb + g.tip[1] * ct - g.tip[2] * st, yb - g.toe_h * ct)
    lo = min(vals)
    if soft > 1e-9:
        lo -= soft * math.log(sum(math.exp(-(v - lo) / soft) for v in vals))
    return -lo


def _clear_shape(u):
    """swing clearance profile: leaves the ground with zero velocity, lands with a small downward velocity"""
    return math.sin(math.pi * u) * (1.0 - (1.0 - u) ** 3)


def foot_track(g, stances, times):
    """evaluate a list of Stance (time ordered) at `times` ->
       dict(ank (n,3), yaw, pitch, tpitch (world toe pitch), swing (0 planted .. 1 mid swing), flat (bool))"""
    n = len(times)
    out = dict(ank=np.zeros((n, 3)), yaw=np.zeros(n), pitch=np.zeros(n), tpitch=np.zeros(n), swing=np.zeros(n),
               flat=np.zeros(n, bool))
    for i, t in enumerate(times):
        k = 0
        while k < len(stances) - 1 and t > stances[k].tl + 1e-9:
            k += 1
        s = stances[k]
        if t >= s.tc - 1e-9:                                   # ---------------- in contact
            yaw = s.yaw
            if t < s.tf - 1e-9:                                # landing roll
                u = (t - s.tc) / (s.tf - s.tc)
                phi = s.land * (1 - u) ** 2
                piv, phit = (g.heel, phi) if s.land < 0 else (g.ball, 0.0)
            elif t <= s.th + 1e-9:
                phi, phit, piv = 0.0, 0.0, g.ball
                out["flat"][i] = True
            else:                                              # heel-off roll about the ball, then about the toe tip
                u = (t - s.th) / (s.tl - s.th)
                phi, phit, piv = s.lift * u * u, 0.0, g.ball
                if s.toe and t > s.tt:
                    w = (t - s.tt) / (s.tl - s.tt)
                    phit = s.toe * w * w
            if phit > 0 and phi > 0:                           # pivot = toe tip (fixed), the ball joint lifts
                B = s.G + _ry(g.ball + g.tip, yaw) - _ry(_rx(g.tip, phit), yaw)
                A = B - _ry(_rx(g.ball, phi), yaw)
            else:
                A = s.G + _ry(piv, yaw) - _ry(_rx(piv, phi), yaw)
        else:                                                  # ---------------- swing from stances[k-1] to s
            s0 = stances[k - 1]
            T = s.tc - s0.tl
            u = (t - s0.tl) / T
            m0 = (2 * s0.lift / (s0.tl - s0.th) * T) if s0.lift else 0.0
            m1 = (-2 * s.land / (s.tf - s.tc) * T) if s.land else 0.0
            if s.mid is None:
                phi = hermite(u, s0.lift, m0, s.land, m1)
            else:
                um, pm, sm = s.mid
                phi = hermite(u / um, s0.lift, m0 * um, pm, sm * um) if u < um else \
                    hermite((u - um) / (1 - um), pm, sm * (1 - um), s.land, m1 * (1 - um))
            # toes: bend relative to the foot goes from its lift-off value to relaxed (0) in the first part of the swing
            ub = 0.45
            b0 = s0.toe - s0.lift
            db0 = ((2 * s0.toe / (s0.tl - s0.tt) * T) if s0.toe else 0.0) - m0
            bend = hermite(u / ub, b0, db0 * ub, 0.0, 0.0) if u < ub else 0.0
            phit = phi + bend
            if s.land > 0:
                phit *= 1.0 - float(smoothstep((u - 0.55) / 0.45))   # ... and flat again for a toe-first landing
            e = float(smoothstep(u / s.ease_end))
            yaw = s0.yaw + (s.yaw - s0.yaw) * e
            pivc = g.heel if s.land < 0 else g.ball
            w = float(smoothstep(u))
            # reference point slides from the toe tip (lift pivot) to the landing pivot
            ref = (1 - w) * (g.ball + _rx(g.tip, phit - phi)) + w * pivc
            P0 = s0.G + _ry(g.ball + g.tip, s0.yaw)
            P1 = s.G + _ry(pivc, s.yaw)
            Gr = P0 + (P1 - P0) * e
            if s.bulge:
                mid = 0.5 * (P0 + P1)
                r = math.hypot(mid[0], mid[2])
                if r > 1e-6:
                    Gr = Gr + s.bulge * math.sin(math.pi * u) ** 2 * np.array([mid[0] / r, 0, mid[2] / r])
            A = Gr - _ry(_rx(ref, phi), yaw)
            A[1] = s.clear * _clear_shape(u) + _foot_rise(g, phi, phit, 0.004 * math.sin(math.pi * u))
            if s.clear2:
                A[1] += s.clear2 * math.sin(math.pi * min(u / 0.62, 1.0)) ** 2
            out["swing"][i] = math.sin(math.pi * u)
        out["ank"][i], out["yaw"][i], out["pitch"][i], out["tpitch"][i] = A, yaw, phi, phit
    return out


# ================================================================================================ control rig -> pose
def new_controls(rig, n):
    """neutral control values: bind stance, relaxed arms"""
    z = lambda v=0.0: np.full(n, float(v))                       # noqa: E731
    C = dict(n=n, root_pos=np.zeros((n, 3)), root_yaw=z(), pel_pos=np.tile(rig.P["Hips"], (n, 1)),
             pel_yaw=z(), pel_pitch=z(), pel_roll=z(), chest_yaw=z(), chest_pitch=z(), chest_roll=z(), breath=z(),
             head_yaw=z(), head_pitch=z(), head_roll=z())
    for s in SIDES:
        k = s[0]
        C[k + "_abd"], C[k + "_flex"], C[k + "_elbow"] = z(ARM["abd"] * DEG), z(ARM["flex"] * DEG), z(ARM["elbow"] * DEG)
        C[k + "_clav_up"], C[k + "_clav_fwd"] = z(ARM["clav_up"] * DEG), z(ARM["clav_fwd"] * DEG)
        C[k + "_hand_pitch"], C[k + "_hand_roll"] = z(), z()
        C[k + "_ank"] = np.tile(rig.leg[s].ank, (n, 1))
        C[k + "_fyaw"], C[k + "_fpitch"], C[k + "_tpitch"] = z(), z(), z()
    return C


CONTROL_KEYS_ANGLE = ("pel_yaw", "pel_pitch", "pel_roll", "chest_yaw", "chest_pitch", "chest_roll", "breath",
                      "head_yaw", "head_pitch", "head_roll")


def controls_from(rig, base, n):
    """n-frame controls holding the single-frame control set `base`"""
    C = dict(n=n)
    for k, v in base.items():
        if k == "n":
            continue
        C[k] = np.tile(v, (n, 1)) if np.ndim(v) == 1 else np.full(n, float(v))
    return C


def controls_at(C, i):
    return {k: (v[i].copy() if isinstance(v, np.ndarray) else v) for k, v in C.items() if k != "n"}


def _hip_frames(rig, C):
    Qh = qmul(qmul(qy(C["pel_yaw"]), qx(C["pel_pitch"])), qz(C["pel_roll"]))
    return Qh


def leg_limits(rig, C, rho):
    """pelvis (Hips joint) height at which each leg reaches rho * leg length for the current feet / pelvis xz
       rho: {side: (n,) or float}.  Returns {side: (n,)} (very low where the foot is out of horizontal reach)"""
    Qh = _hip_frames(rig, C)
    out = {}
    for s in SIDES:
        g = rig.leg[s]
        ho = qrot(Qh, g.off_up)
        A = C[s[0] + "_ank"]
        dx = C["pel_pos"][:, 0] + ho[:, 0] - A[:, 0]
        dz = C["pel_pos"][:, 2] + ho[:, 2] - A[:, 2]
        r2 = (np.asarray(rho[s], float) * g.L) ** 2 - dx * dx - dz * dz
        out[s] = np.where(r2 > 0, A[:, 1] + np.sqrt(np.maximum(r2, 0)), A[:, 1] - 1.0) - ho[:, 1]
    return out


def solve(rig, C):
    """controls -> local rotations of all body joints (no cloth), Hips local translation, world transforms"""
    n = C["n"]
    lq = {}
    q_root = qy(C["root_yaw"])
    Qh = _hip_frames(rig, C)
    Ph = C["pel_pos"]
    lq["Root"] = q_root
    lq["Hips"] = qmul(qconj(q_root), Qh)
    hips_t = qrot(qconj(q_root), Ph - C["root_pos"])

    # ---- spine: spread the pelvis -> chest rotation over three joints, breathing on the upper two
    Qc = qmul(qmul(qy(C["chest_yaw"]), qx(C["chest_pitch"])), qz(C["chest_roll"]))
    rel = qmul(qconj(Qh), Qc)
    lq["Spine"] = qpow(rel, 0.30)
    lq["Spine1"] = qmul(qpow(rel, 0.35), qx(0.45 * C["breath"]))
    lq["Spine2"] = qmul(qpow(rel, 0.35), qx(0.55 * C["breath"]))
    Qs2 = qmul(qmul(qmul(Qh, lq["Spine"]), lq["Spine1"]), lq["Spine2"])
    # ---- head: absolute orientation, half of the correction on the neck
    Qhd = qmul(qmul(qy(C["head_yaw"]), qx(C["head_pitch"])), qz(C["head_roll"]))
    lq["Neck"] = qpow(qmul(qconj(Qs2), Qhd), 0.5)
    lq["Head"] = qmul(qconj(qmul(Qs2, lq["Neck"])), Qhd)

    # ---- arms: absolute bone directions in the heading frame of the chest (the arms hang by gravity)
    Qref = qy(C["chest_yaw"])
    for s in SIDES:
        g, sg, k = rig.arm[s], SG[s], s[0]
        lcl = qmul(qz(sg * C[k + "_clav_up"]), qy(-sg * C[k + "_clav_fwd"]))
        lq[s + "Shoulder"] = lcl
        Qcl = qmul(Qs2, lcl)
        abd, flex, elb = C[k + "_abd"], C[k + "_flex"], C[k + "_elbow"]
        du = np.stack([sg * np.sin(abd), -np.cos(abd) * np.cos(flex), np.cos(abd) * np.sin(flex)], -1)
        gam = ARM["elbow_in"] * DEG
        F = np.array([-sg * math.sin(gam), 0.0, math.cos(gam)])            # the elbow folds forward / slightly inward
        perp = _norm(F - np.sum(du * F, -1, keepdims=True) * du)
        df = np.cos(elb)[:, None] * du + np.sin(elb)[:, None] * perp
        Su = q_arc(g.a_u, du)
        Sf = qmul(q_arc(qrot(Su, g.a_f), df), Su)
        Qu, Qf = qmul(Qref, Su), qmul(Qref, Sf)
        lq[s + "Arm"] = qmul(qconj(Qcl), Qu)
        lq[s + "ForeArm"] = qmul(qconj(Qu), Qf)
        lq[s + "Hand"] = qmul(qx(C[k + "_hand_pitch"]), qz(sg * C[k + "_hand_roll"]))

    # ---- legs: analytic two-bone IK
    diag = {}
    for s in SIDES:
        g, k = rig.leg[s], s[0]
        up, lo, ft, toe = g.names
        A, fy = C[k + "_ank"], C[k + "_fyaw"]
        Qf = qmul(qy(fy), qx(C[k + "_fpitch"]))
        Qt = qmul(qy(fy), qx(C[k + "_tpitch"]))
        ky = C.get(k + "_kyaw")
        if ky is None:
            ky = fy + KNEE_POLE * (C["pel_yaw"] - fy)
        H = Ph + qrot(Qh, g.off_up)
        v = A - H
        d = np.linalg.norm(v, axis=-1)
        reach = d / g.L
        dc = np.minimum(d, g.L * 0.99995)
        u = v / d[:, None]
        cosA = np.clip((g.L1 ** 2 + dc ** 2 - g.L2 ** 2) / (2 * g.L1 * dc), -1, 1)
        sinA = np.sqrt(1 - cosA ** 2)
        fwd = np.stack([np.sin(ky), np.zeros(n), np.cos(ky)], -1)
        w = _norm(fwd - np.sum(fwd * u, -1, keepdims=True) * u)
        K = H + g.L1 * (cosA[:, None] * u + sinA[:, None] * w)
        Aeff = H + u * dc[:, None]
        hinge = np.cross(w, u)
        tdir, sdir = (K - H) / g.L1, _norm(Aeff - K)
        Et = np.stack([hinge, tdir, np.cross(hinge, tdir)], -1)
        Es = np.stack([hinge, sdir, np.cross(hinge, sdir)], -1)
        Qth = q_from_m(Et @ g.E0t.T)
        Qsh = q_from_m(Es @ g.E0s.T)
        lq[up] = qmul(qconj(Qh), Qth)
        lq[lo] = qmul(qconj(Qth), Qsh)
        lq[ft] = qmul(qconj(Qsh), Qf)
        lq[toe] = qmul(qconj(Qf), Qt)
        diag[s] = dict(reach=reach, knee=np.degrees(np.arccos(np.clip(np.sum(tdir * sdir, -1), -1, 1))))

    # ---- world transforms of every body joint (cloth joints excluded)
    Wp, Wq = {}, {}
    for name, parent in HIERARCHY:
        if name not in lq:
            continue
        if parent is None:
            Wp[name], Wq[name] = C["root_pos"].copy(), lq[name]
        elif name == "Hips":
            Wp[name], Wq[name] = Ph.copy(), Qh
        else:
            Wp[name] = Wp[parent] + qrot(Wq[parent], rig.off[name])
            Wq[name] = qmul(Wq[parent], lq[name])
    return dict(lq=lq, hips_t=hips_t, Wp=Wp, Wq=Wq, diag=diag)


# ================================================================================================ cloth helpers
CHAINS = (("LeftRibbon1", "LeftRibbon2", "LeftRibbon_End", "LeftForeArm"),
          ("RightRibbon1", "RightRibbon2", "RightRibbon_End", "RightForeArm"),
          ("Sash1", "Sash2", "Sash_End", "Hips"))
SKIRT = (0.158, 0.128)        # half axes (x, z) of the tunic skirt around the pelvis (mesh measurement, model units)


def _seg_dist(p, a, b):
    ab = b - a
    t = min(max(float(np.dot(p - a, ab)) / max(float(np.dot(ab, ab)), 1e-12), 0.0), 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def bake_cloth(rig, sol, C, loop=False, cycle=None, init=None, settle=0, n_sub=6, n_cycles=4):
    """Damped pendulum chains for the ribbons and the sash, simulated in world space.
       loop: the frames 0..n-1 are one period (frame n-1 == frame 0 moved by `cycle` = (yaw, translation));
       init: particle state to start from (one-shots start from the Idle state);  settle: number of frames at the
       end over which a one-shot is blended back to its initial state.
       The equilibrium of every segment is its rest direction (gravity is aligned with it), so the modelled shape of
       the cloth is kept; the segments lag behind the motion of their attachment and are pushed out of simple
       colliders (tunic skirt, hands, left thigh).  Returns (local rotations, particle state at frame 0)."""
    n = C["n"]
    Wp, Wq = sol["Wp"], sol["Wq"]
    h = 1.0 / (FPS * n_sub)
    chest_yaw, pel_yaw = yaw_of(Wq["Spine2"]), yaw_of(Wq["Hips"])
    P = rig.P
    hip_seg = (P["LeftUpLeg"] + np.array([0, 0.07, 0]), P["LeftLeg"])
    chains = []
    for j1, j2, je, par in CHAINS:
        if je not in P:
            P[je] = P[j2] + (P[j2] - P[j1])
        r1, r2 = P[j2] - P[j1], P[je] - P[j2]
        ch = dict(names=(j1, j2), parent=par, len=(np.linalg.norm(r1), np.linalg.norm(r2)), rest=(_norm(r1), _norm(r2)),
                  anchor=Wp[par] + qrot(Wq[par], rig.off[j1]), yaw=pel_yaw if par == "Hips" else chest_yaw)
        if par == "Hips":
            # No capsule collider for the sash: a radial push-out lets the pendulum tunnel to the wrong side of the
            # thigh during turn steps and stay stuck there.  The thigh / sash contact is resolved afterwards by
            # cloth_fix.sash_follow_thigh (the sash is pushed forward by the left thigh, never sideways or back).
            ch["damp"], ch["air"], ch["caps"] = 30.0, 1.0, []
            ch["skirt"] = False
        else:
            side = j1[:-7]
            hand = Wp[side + "Hand"]
            ch["damp"], ch["air"], ch["caps"] = 24.0, 1.1, [
                (hand, hand + qrot(Wq[side + "Hand"], rig.arm[side].hand_end), 0.024)]
            ch["skirt"] = True
        chains.append(ch)

    def collide(ch, x, f0, f1, a):
        for ca, cb, r in ch["caps"]:
            pa = ca[f0] + (ca[f1] - ca[f0]) * a
            pb = cb[f0] + (cb[f1] - cb[f0]) * a
            ab = pb - pa
            t = min(max(float(np.dot(x - pa, ab)) / max(float(np.dot(ab, ab)), 1e-12), 0.0), 1.0)
            c = pa + t * ab
            dv = x - c
            dist = float(np.linalg.norm(dv))
            if 1e-9 < dist < r:
                x = c + dv * (r / dist)
        if ch["skirt"]:
            pc = Wp["Hips"][f0] + (Wp["Hips"][f1] - Wp["Hips"][f0]) * a
            if x[1] < pc[1] + 0.03:
                ya = pel_yaw[f0] + (pel_yaw[f1] - pel_yaw[f0]) * a
                rel = _ry(x - pc, -ya)
                f = (rel[0] / SKIRT[0]) ** 2 + (rel[2] / SKIRT[1]) ** 2
                if f < 1.0:
                    sc = 1.0 / math.sqrt(max(f, 1e-6))
                    rel = np.array([rel[0] * sc, rel[1], rel[2] * sc])
                    x = pc + _ry(rel, ya)
        return x

    def rest_state(ch, f):
        qb = qy(ch["yaw"][f])
        x1 = ch["anchor"][f] + ch["len"][0] * qrot(qb, ch["rest"][0])
        x2 = x1 + ch["len"][1] * qrot(qb, ch["rest"][1])
        return [x1, x2, x1.copy(), x2.copy()]

    if cycle is None:
        cyc_yaw, cyc_t = 0.0, np.zeros(3)
    else:
        cyc_yaw, cyc_t = cycle
    rec = [np.zeros((n, 2, 3)) for _ in chains]
    state0 = []
    for ci, ch in enumerate(chains):
        st = [np.array(v, float) for v in init[ci]] if init is not None else rest_state(ch, 0)
        passes = n_cycles if loop else 1
        for cyc in range(passes):
            for f in range(n - 1):
                if f == 0 and cyc == passes - 1:
                    state0.append([v.copy() for v in st])
                rec[ci][f, 0], rec[ci][f, 1] = st[0], st[1]
                for sub in range(n_sub):
                    a0, a1 = sub / n_sub, (sub + 1) / n_sub
                    anc0 = ch["anchor"][f] + (ch["anchor"][f + 1] - ch["anchor"][f]) * a0
                    anc1 = ch["anchor"][f] + (ch["anchor"][f + 1] - ch["anchor"][f]) * a1
                    ya = ch["yaw"][f] + (ch["yaw"][f + 1] - ch["yaw"][f]) * a1
                    par0, par1 = anc0, anc1
                    new = []
                    for p in range(2):
                        x, xp = st[p], st[p + 2]
                        v = (x - xp) / h
                        vpar = (par1 - par0) / h
                        gdir = _ry(ch["rest"][p], ya)
                        v = (v + h * (GRAVITY * gdir + ch["damp"] * vpar)) / (1.0 + h * (ch["damp"] + ch["air"]))
                        xn = x + v * h
                        for _ in range(2):
                            dv = xn - par1
                            xn = par1 + dv * (ch["len"][p] / max(float(np.linalg.norm(dv)), 1e-9))
                            xn = collide(ch, xn, f, f + 1, a1)
                        dv = xn - par1
                        xn = par1 + dv * (ch["len"][p] / max(float(np.linalg.norm(dv)), 1e-9))
                        new.append(xn)
                        par0, par1 = x, xn
                    st = [new[0], new[1], st[0], st[1]]
            rec[ci][n - 1, 0], rec[ci][n - 1, 1] = st[0], st[1]
            if loop and cyc < passes - 1:                      # map the state back into the first period
                st = [_ry(v - cyc_t, -cyc_yaw) for v in st]
        if loop:                                               # close the seam exactly (residual is tiny)
            end = np.stack([_ry(rec[ci][n - 1, p] - cyc_t, -cyc_yaw) for p in range(2)])
            res = end - rec[ci][0]
            wgt = (np.arange(n) / (n - 1.0))[:, None, None]
            corr = wgt * res[None]
            for f in range(n):
                a = f / (n - 1.0)
                for p in range(2):
                    rec[ci][f, p] -= _ry(corr[f, p], cyc_yaw * a) if cyc_yaw else corr[f, p]
        elif settle and init is not None:
            for f in range(max(n - settle, 0), n):
                wgt = float(smoothstep((f - (n - settle)) / max(settle - 1.0, 1.0)))
                tgt0 = np.array(init[ci][0]) - chains[ci]["anchor"][0] + chains[ci]["anchor"][f]
                tgt1 = np.array(init[ci][1]) - chains[ci]["anchor"][0] + chains[ci]["anchor"][f]
                rec[ci][f, 0] += wgt * (tgt0 - rec[ci][f, 0])
                rec[ci][f, 1] += wgt * (tgt1 - rec[ci][f, 1])

    lq = {}
    for ci, ch in enumerate(chains):
        qb = qy(ch["yaw"])
        d1 = _norm(rec[ci][:, 0] - ch["anchor"])
        d2 = _norm(rec[ci][:, 1] - rec[ci][:, 0])
        Q1 = qmul(q_arc(qrot(qb, ch["rest"][0]), d1), qb)
        Q2 = qmul(q_arc(qrot(Q1, ch["rest"][1]), d2), Q1)
        lq[ch["names"][0]] = qmul(qconj(Wq[ch["parent"]]), Q1)
        lq[ch["names"][1]] = qmul(qconj(Q1), Q2)
        sol["Wp"][ch["names"][0]], sol["Wq"][ch["names"][0]] = ch["anchor"], Q1
        sol["Wp"][ch["names"][1]], sol["Wq"][ch["names"][1]] = rec[ci][:, 0], Q2
    return lq, state0


# ================================================================================================ pelvis height
def pelvis_height(rig, C, y_des, rho, periodic, sigma=0.9, k=0.004):
    """highest pelvis not above y_des that keeps both legs within the reach profile rho (smooth), plus a hard
       safety clamp at REACH_HARD.  Writes C['pel_pos'][:, 1]."""
    lim = leg_limits(rig, C, rho)
    y = smin(smin(lim["Left"], lim["Right"], k), np.asarray(y_des, float), k)
    if periodic:
        y[:-1] = gauss_smooth(y[:-1], sigma, True)
        y[-1] = y[0]
    else:
        y = gauss_smooth(y, sigma, False)
    hard = leg_limits(rig, C, {s: REACH_HARD for s in SIDES})
    y = np.minimum(y, np.minimum(hard["Left"], hard["Right"]))
    C["pel_pos"][:, 1] = y
    return y


# ================================================================================================ clip assembly
def make_clip(rig, name, C, extras, loop, root_motion, cloth_cycle=None, cloth_init=None, settle=0, keep_root_yaw=True):
    sol = solve(rig, C)
    cl, cloth0 = bake_cloth(rig, sol, C, loop=loop, cycle=cloth_cycle, init=cloth_init, settle=settle)
    lq = dict(sol["lq"])
    lq.update(cl)
    n = C["n"]
    t = np.arange(n) / FPS
    tracks = {}
    for jn, _ in HIERARCHY:
        q = continuous(lq[jn])
        if jn == "Root" and not keep_root_yaw:
            q = np.tile([0, 0, 0, 1.0], (n, 1))
        elif loop and jn != "Root":
            q[-1] = q[0]
        tr = {"t": t, "rotation": q}
        if jn == "Root":
            tr["translation"] = C["root_pos"].copy() if root_motion else np.zeros((n, 3))
        elif jn == "Hips":
            ht = sol["hips_t"].copy()
            if loop:
                ht[-1] = ht[0]
            tr["translation"] = ht
        tracks[jn] = tr
    ex = dict(loop=bool(loop), rootMotion=bool(root_motion), fps=FPS)
    ex.update(extras)
    return dict(name=name, duration=float((n - 1) / FPS), tracks=tracks, extras=ex, _sol=sol, _C=C, _cloth0=cloth0)


# ================================================================================================ Idle
def build_idle(rig):
    n = IDLE["frames"] + 1
    D = IDLE["frames"] / FPS
    t = np.arange(n) / FPS
    p = t / D
    C = new_controls(rig, n)
    # breathing: inhale ~40 % of the cycle, slower exhale, short pause
    br = pspline([0.0, 0.40, 0.90], [0.0, 1.0, 0.03], p, flat=True)
    brm = br - 0.45
    # weight shift: drift to the left leg, settle, drift to the right, come back (zero at t = 0, not a sine)
    sw = pspline([0.0, 0.27, 0.50, 0.78], [0.0, 1.0, 0.82, -0.55], p)
    sw = sw - sw[0]
    sw2 = np.sin(2 * np.pi * (2 * p)) * 0.12                         # small faster component
    lat = IDLE["sway"] * (sw + sw2)
    hips = rig.P["Hips"]
    C["pel_pos"][:, 0] = hips[0] + lat
    C["pel_pos"][:, 2] = hips[2] + 0.0025 * np.sin(2 * np.pi * (p - 0.18)) - 0.0025 * np.sin(2 * np.pi * (-0.18)) + 0.0008 * brm
    C["pel_roll"] = 1.3 * DEG * (sw + sw2)                         # hip of the loaded leg rises a little
    C["pel_yaw"] = -0.7 * DEG * sw
    C["pel_pitch"] = 0.6 * DEG + 0.0 * p
    C["chest_roll"] = -0.9 * DEG * pspline([0.0, 0.27, 0.50, 0.78], [0.0, 1.0, 0.82, -0.55], p - 0.035)
    C["chest_roll"] -= C["chest_roll"][0]
    C["chest_yaw"] = 0.9 * DEG * np.sin(2 * np.pi * (p - 0.08)) - 0.9 * DEG * math.sin(2 * math.pi * (-0.08))
    C["chest_pitch"] = 1.2 * DEG - 0.25 * DEG * brm
    C["breath"] = -IDLE["breath"] * DEG * brm                      # chest lifts (extends) on the inhale
    hy = 2.6 * np.sin(2 * np.pi * (p - 0.31)) + 1.1 * np.sin(2 * np.pi * (2 * p + 0.12))
    C["head_yaw"] = DEG * (hy - hy[0])
    C["head_pitch"] = DEG * (1.5 + 0.9 * np.sin(2 * np.pi * (p - 0.55)) + 0.35 * brm)
    hr = 0.8 * np.sin(2 * np.pi * (p - 0.2))
    C["head_roll"] = DEG * (hr - hr[0])
    for s in SIDES:
        k, sg = s[0], SG[s]
        C[k + "_clav_up"] = C[k + "_clav_up"] + 0.9 * DEG * brm
        C[k + "_abd"] = C[k + "_abd"] + 0.35 * DEG * brm + sg * 0.5 * DEG * np.roll(sw, 0) * 0.6
        lag = pspline([0.0, 0.27, 0.50, 0.78], [0.0, 1.0, 0.82, -0.55], p - 0.06)
        C[k + "_flex"] = C[k + "_flex"] + 0.5 * DEG * np.sin(2 * np.pi * (p - 0.4 - 0.07 * sg)) - 0.3 * DEG * brm
        C[k + "_abd"] = C[k + "_abd"] - sg * 0.0 * lag
        C[k + "_elbow"] = C[k + "_elbow"] + 0.7 * DEG * brm + 0.5 * DEG * np.sin(2 * np.pi * (p - 0.6 + 0.05 * sg))
        C[k + "_hand_pitch"] = 0.8 * DEG * np.sin(2 * np.pi * (p - 0.7))
    y_des = hips[1] - 0.0015 * (sw / np.max(np.abs(sw))) ** 2 + 0.0006 * brm
    pelvis_height(rig, C, y_des, {s: IDLE["knee_reach"] for s in SIDES}, periodic=True, sigma=0.0)
    clip = make_clip(rig, "Idle", C, dict(footsteps=[]), loop=True, root_motion=False)
    return clip


# ================================================================================================ Walk
def _walk_controls(rig, S, lead, full=True, tune=None):
    W = WALK
    tune = tune or dict(lift=1.0, clear2=0.0)
    N = W["frames"]
    n = N + 1
    T = N / FPS
    v = 2 * S / T
    t = np.arange(n) / FPS
    p = t / T
    C = new_controls(rig, n)
    hips = rig.P["Hips"]
    hipz = hips[2] + 0.5 * (rig.leg["Left"].off_up[2] + rig.leg["Right"].off_up[2])
    # ---- root and pelvis (world, with root motion)
    C["root_pos"][:, 2] = v * t
    C["pel_yaw"] = -W["pel_yaw"] * DEG * np.cos(2 * np.pi * p)                # hip of the leading leg forward
    C["pel_roll"] = W["pel_list"] * DEG * np.sin(2 * np.pi * (p + 0.05))      # swing side drops
    C["pel_pitch"] = np.full(n, W["pel_pitch"] * DEG)
    C["pel_pos"][:, 0] = hips[0] + W["pel_lat"] * np.sin(2 * np.pi * (p - 0.05))
    C["pel_pos"][:, 2] = hips[2] + v * t - W["pel_fore"] * np.sin(4 * np.pi * (p - 0.24))
    # ---- feet
    ft = {}
    for s in SIDES:
        g, sg = rig.leg[s], SG[s]
        ph = 0.0 if s == "Left" else 0.5
        st = []
        for c in (-1, 0, 1):
            tc = (c + ph) * T
            G = np.array([W["width"] * g.ank[0], g.ank[1], hipz + lead + (c + ph) * 2 * S])
            st.append(Stance(G, sg * W["toe_out"] * DEG, tc=tc, tf=tc + W["f_flat"] / FPS, th=tc + W["f_heel_off"] / FPS,
                             tl=tc + W["f_lift"] / FPS, land=W["land_pitch"] * DEG,
                             lift=W["lift_pitch"] * DEG * tune["lift"], toe=W["toe_pitch"] * DEG * tune["lift"],
                             tt=tc + (W["f_lift"] - W["f_toe"]) / FPS,
                             clear=W["clear"], clear2=tune["clear2"], ease_end=W["ease_end"],
                             mid=(W["swing_mid"][0], W["swing_mid"][1] * DEG, W["swing_mid"][2] * DEG)))
        st[0].tc = -1e9
        ft[s] = foot_track(g, st, t)
        k = s[0]
        C[k + "_ank"], C[k + "_fyaw"], C[k + "_fpitch"], C[k + "_tpitch"] = ft[s]["ank"], ft[s]["yaw"], ft[s]["pitch"], ft[s]["tpitch"]
    # ---- pelvis height from the reach profile of the legs
    rho = {}
    for s in SIDES:
        ps = np.mod(p - (0.0 if s == "Left" else 0.5), 1.0)
        bump = np.where(ps < 0.24, 0.5 - 0.5 * np.cos(2 * np.pi * ps / 0.24), 0.0)
        rho[s] = W["reach"] - W["reach_load"] * bump
    y = pelvis_height(rig, C, hips[1] + 0.05, rho, periodic=True)
    if not full:
        return C, y
    # ---- upper body
    C["chest_yaw"] = W["chest_yaw"] * DEG * np.cos(2 * np.pi * (p - 0.02))
    C["chest_pitch"] = W["chest_pitch"] * DEG + 0.5 * DEG * np.cos(4 * np.pi * (p - 0.08))
    C["chest_roll"] = -W["chest_roll"] * DEG * np.sin(2 * np.pi * (p + 0.03))
    C["breath"] = np.zeros(n)
    C["head_yaw"] = 0.18 * C["chest_yaw"]
    C["head_roll"] = 0.25 * C["chest_roll"]
    C["head_pitch"] = 1.0 * DEG + 0.4 * DEG * np.cos(4 * np.pi * (p - 0.12))
    for s in SIDES:
        k, sg = s[0], SG[s]
        pa = p - 0.035 - (0.0 if s == "Left" else 0.5)          # this arm is furthest back when its own leg is in front
        swing = -np.cos(2 * np.pi * pa)                            # -1 back .. +1 forward
        C[k + "_flex"] = (W["arm_fwd"] + W["arm_swing"] * swing) * DEG
        fwd = 0.5 + 0.5 * np.cos(2 * np.pi * (pa - 0.5 - 0.045))
        C[k + "_elbow"] = (ARM["elbow"] - 2.0 + W["elbow_swing"] * fwd ** 1.3) * DEG
        C[k + "_abd"] = (ARM["abd"] + 1.0 - 1.2 * swing) * DEG
        C[k + "_clav_fwd"] = 1.6 * DEG * swing
        C[k + "_clav_up"] = np.full(n, (ARM["clav_up"]) * DEG)
        C[k + "_hand_pitch"] = -4.0 * DEG * np.cos(2 * np.pi * (pa - 0.12))   # wrist lags behind the swing
    return C, y


def _bisect(fn, lo, hi, target, it=12):
    """monotonic increasing fn: argument in [lo, hi] where fn = target (clamped to the interval)"""
    if fn(lo) >= target:
        return lo
    if fn(hi) <= target:
        return hi
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _walk_layout(rig):
    """choose the step length, the fore-aft placement of the feet under the pelvis and the foot roll / swing lift
       that give the wanted knee angles at toe-off and in mid swing (adapts to the proportions of the skeleton)"""
    W = WALK
    S = W["step_frac"] * rig.height
    Lmin = min(rig.leg[s].L for s in SIDES)
    best = None
    for _ in range(14):
        cands = []
        for lead in np.linspace(0.30 * S, 0.62 * S, 17):
            C, y = _walk_controls(rig, S, lead, full=False)
            cands.append((y.min(), lead, y.max() - y.min()))
        # the best fore-aft offset is the one that keeps the pelvis highest in double support
        ymin, lead, bob = max(cands)
        lo = max(lead - 0.02 * S, 0.0)
        for lead2 in np.linspace(lo, lead + 0.02 * S, 9):
            C, y = _walk_controls(rig, S, lead2, full=False)
            if y.min() > ymin:
                ymin, lead, bob = y.min(), lead2, y.max() - y.min()
        best = (S, lead, bob)
        if bob <= W["bob_max"] and ymin > 0.2 * Lmin:
            break
        S *= 0.975
    S, lead, bob = best
    N, fl = W["frames"], W["f_lift"]

    def knees(tune):
        C, _ = _walk_controls(rig, S, lead, full=False, tune=tune)
        d = solve(rig, C)["diag"]
        kl, kr = d["Left"]["knee"][:N], np.roll(d["Right"]["knee"][:N], -N // 2)
        return 0.5 * (kl + kr)

    lift = _bisect(lambda f: knees(dict(lift=f, clear2=0.0))[fl], 0.8, 1.2, W["knee_toe_off"])
    clear2 = _bisect(lambda c: knees(dict(lift=lift, clear2=c))[fl:].max(), 0.0, 0.05, W["knee_swing"])
    return S, lead, bob, dict(lift=lift, clear2=clear2)


def build_walk(rig):
    S, lead, bob, tune = _walk_layout(rig)
    C, _ = _walk_controls(rig, S, lead, tune=tune)
    N = WALK["frames"]
    T = N / FPS
    v = 2 * S / T
    ex = dict(speed=float(v), strideLength=float(2 * S), stepLength=float(S),
              footsteps=[dict(foot="L", t=0.0), dict(foot="R", t=float(T / 2))])
    rm = make_clip(rig, "Walk_RootMotion", C, ex, loop=True, root_motion=True, cloth_cycle=(0.0, np.array([0, 0, 2 * S])))
    ip = dict(rm)
    ip["name"] = "Walk"
    ip["tracks"] = dict(rm["tracks"])
    ip["tracks"]["Root"] = dict(rm["tracks"]["Root"], translation=np.zeros((N + 1, 3)))
    ip["extras"] = dict(rm["extras"], rootMotion=False)
    ip["_info"] = rm["_info"] = dict(step=S, lead=lead, bob=bob, speed=v, tune=tune)
    return ip, rm


# ================================================================================================ turns
def _yaw_profile(t, D, ta, td):
    """0 -> 1 with smooth acceleration over ta and deceleration over td; returns (fraction, normalised rate)"""
    def integ(x):                                                # integral of smoothstep on [0, x]
        x = np.clip(x, 0, 1)
        return x ** 3 - 0.5 * x ** 4
    t = np.asarray(t, float)
    rate = np.minimum(smoothstep(t / ta), smoothstep((D - t) / td))
    area = ta * integ(np.minimum(t / ta, 1)) + np.clip(t - ta, 0, D - ta - td) + td * (0.5 - integ(np.clip((D - t) / td, 0, 1)))
    total = 0.5 * ta + (D - ta - td) + 0.5 * td
    return area / total, rate


def _turn_controls(rig, idle0, sgn, mode):
    """world-space controls of a stepping turn.  sgn +1 = left (counter-clockwise), -1 = right (mirrored).
       mode 'one': 0 -> 360 deg one-shot starting / ending in the Idle start pose
       mode 'loop': steady turning, returns the controls of a few periods (the caller cuts one period out)"""
    Tn = TURN
    inner, outer = ("Left", "Right") if sgn > 0 else ("Right", "Left")
    stp, swf = Tn["step"], Tn["swing"]
    if mode == "one":
        N = Tn["frames"]
        n = N + 1
        t = np.arange(n) / FPS
        D = N / FPS
        frac, rate = _yaw_profile(t, D, Tn["ramp_in"], Tn["ramp_out"])
        psi = sgn * 2 * np.pi * frac
        def psi_at(tt):
            return sgn * 2 * np.pi * _yaw_profile(np.clip(tt, 0, D), D, Tn["ramp_in"], Tn["ramp_out"])[0]
        def rate_at(tt):
            return _yaw_profile(np.clip(tt, 0, D), D, Tn["ramp_in"], Tn["ramp_out"])[1]
        # 8 steps: inner foot first.  lift frames; the last two steps land in the final (home) footprints
        lifts = Tn["lifts"]
        swings = [swf + 1] + [swf] * (len(lifts) - 2) + [swf + 1]
        final_psi = sgn * 2 * np.pi
        env = smoothstep(t / 0.3) * smoothstep((D - t) / 0.4)
    else:
        cycles = 5
        N = 2 * stp * cycles
        n = N + 1
        t = np.arange(n) / FPS
        omega = sgn * (0.5 * np.pi) / (2 * stp / FPS)             # 90 deg per period (two steps)
        psi = omega * t
        rate = np.ones(n)
        def psi_at(tt):
            return omega * np.asarray(tt, float)
        def rate_at(tt):
            return np.ones_like(np.asarray(tt, float))
        lifts = [stp * i for i in range(-2, 2 * cycles + 2)]
        swings = [swf] * len(lifts)
        final_psi = None
        env = np.ones(n)
    C = controls_from(rig, idle0, n)
    C["root_yaw"] = psi.copy()
    # ---- footprints
    home = {s: rig.leg[s].ank.copy() for s in SIDES}
    steps = {s: [] for s in SIDES}                                # (lift frame, plant frame) per foot
    for i, (lf, sw) in enumerate(zip(lifts, swings)):
        steps[inner if i % 2 == 0 else outer].append((lf, lf + sw))
    stances, plants = {}, []
    for s in SIDES:
        g, sg = rig.leg[s], SG[s]
        bias = sgn * Tn["bias"] * DEG * (1.0 if s == inner else -1.0)
        seq = steps[s]
        if mode == "one":
            cur = Stance(home[s], 0.0, th=(seq[0][0] - Tn["heel_off"]) / FPS, tl=seq[0][0] / FPS, lift=Tn["lift_pitch"] * DEG)
        else:                                                    # a footprint consistent with the steady state
            lf0 = seq[0][0]
            th0 = 0.5 * (psi_at((lf0 - 2 * stp + swf) / FPS) + psi_at(lf0 / FPS)) + bias
            cur = Stance(_ry(home[s] + np.array([sg * Tn["widen"], 0, 0]), th0), th0, th=(lf0 - Tn["heel_off"]) / FPS,
                         tl=lf0 / FPS, lift=Tn["lift_pitch"] * DEG)
        lst = [cur]
        for j, (lf, pf) in enumerate(seq):
            last = j == len(seq) - 1
            if last and final_psi is not None:
                th, G = final_psi, _ry(home[s], final_psi)
                nxt_lift = 1e9
            else:
                nxt_lift = seq[j + 1][0] if not last else pf + 2 * stp - swf
                th = 0.5 * (psi_at(pf / FPS) + psi_at(nxt_lift / FPS)) + bias
                G = _ry(home[s] + np.array([sg * Tn["widen"], 0, 0]), th)
            lst.append(Stance(G, th, tc=pf / FPS, tf=(pf + Tn["flat"]) / FPS,
                              th=(nxt_lift - Tn["heel_off"]) / FPS if nxt_lift < 1e8 else 1e9,
                              tl=nxt_lift / FPS if nxt_lift < 1e8 else 1e9,
                              land=Tn["land_pitch"] * DEG, lift=Tn["lift_pitch"] * DEG, clear=Tn["clear"], bulge=Tn["bulge"]))
            plants.append((s, pf))
        stances[s] = lst
    ft = {s: foot_track(rig.leg[s], stances[s], t) for s in SIDES}
    for s in SIDES:
        k = s[0]
        C[k + "_ank"], C[k + "_fyaw"], C[k + "_fpitch"], C[k + "_tpitch"] = ft[s]["ank"], ft[s]["yaw"], ft[s]["pitch"], ft[s]["tpitch"]
    # ---- pelvis: over the support, shifted towards the stance foot, smoothed
    per = mode == "loop"
    swL, swR = ft["Left"]["swing"], ft["Right"]["swing"]
    load = gauss_smooth(np.clip(0.5 + 0.5 * (np.sign(swR) - np.sign(swL)), 0, 1), 2.0, False)   # 1 = all weight on the left foot
    cen = {}
    for s in SIDES:
        g = rig.leg[s]
        mid = np.array([_ry(g.mid, a) for a in ft[s]["yaw"]])
        hm = np.array([_ry(home[s] + g.mid, a) for a in psi])
        cen[s] = (ft[s]["ank"] + mid, hm)
    dL, dR = cen["Left"][0] - cen["Left"][1], cen["Right"][0] - cen["Right"][1]
    off = Tn["follow"] * (load[:, None] * dL + (1 - load)[:, None] * dR)
    off += Tn["shift"] * (load - 0.5)[:, None] * (cen["Left"][0] - cen["Right"][0]) * env[:, None]
    off[:, 1] = 0.0
    off = gauss_smooth(off, 2.2, False)
    base_xz = np.array([_ry(np.array([idle0["pel_pos"][0], 0, idle0["pel_pos"][2]]), a) for a in psi])
    C["pel_pos"][:, 0] = base_xz[:, 0] + off[:, 0]
    C["pel_pos"][:, 2] = base_xz[:, 2] + off[:, 2]
    xi = 0.5 * ((ft["Left"]["yaw"] - psi) + (ft["Right"]["yaw"] - psi))     # the pelvis follows the stepping feet a little
    C["pel_yaw"] = idle0["pel_yaw"] + psi + Tn["pel_follow"] * gauss_smooth(xi, 2.0, False)
    C["pel_roll"] = idle0["pel_roll"] + gauss_smooth(1.2 * DEG * (load - 0.5) * 2 * env, 1.5, False) * 1.0
    C["pel_pitch"] = idle0["pel_pitch"] + 1.0 * DEG * env
    # ---- upper body: chest and head lead into the turn, slight lean to the inside
    lead_c = rate_at(t + 0.07)
    lead_h = rate_at(t + 0.16)
    C["chest_yaw"] = idle0["chest_yaw"] + psi + sgn * Tn["chest_lead"] * DEG * lead_c
    C["chest_roll"] = idle0["chest_roll"] - sgn * Tn["lean"] * DEG * lead_c
    C["chest_pitch"] = idle0["chest_pitch"] + 1.6 * DEG * env
    C["head_yaw"] = idle0["head_yaw"] + psi + sgn * Tn["head_lead"] * DEG * lead_h
    C["head_roll"] = idle0["head_roll"] - sgn * 1.0 * DEG * lead_h
    C["head_pitch"] = idle0["head_pitch"] + 1.0 * DEG * env
    stepw = gauss_smooth(swL - swR, 1.5, False)                    # > 0 while the left foot is in the air
    for s in SIDES:
        k, sg = s[0], SG[s]
        C[k + "_abd"] = idle0[k + "_abd"] + (1.6 * env + 0.6 * rate) * DEG
        C[k + "_elbow"] = idle0[k + "_elbow"] + 5.0 * DEG * env
        C[k + "_flex"] = idle0[k + "_flex"] + (2.0 * env - sg * sgn * 2.5 * rate - sg * 2.2 * stepw) * DEG
    # ---- pelvis height: a little lower than Idle while stepping, limited by the reach of both legs
    dip = gauss_smooth(np.abs(np.sign(swL) + np.sign(swR)) < 0.5, 1.5, False)        # 1 in double support
    y_des = idle0["pel_pos"][1] - Tn["crouch"] * env - 0.0035 * dip * env
    y0 = y_des[0]
    y = pelvis_height(rig, C, y_des, {s: Tn["reach"] + (REACH_HARD - Tn["reach"]) * (1 - env) for s in SIDES},
                      periodic=False, sigma=1.0)
    if mode == "one":                                            # exact Idle pose at both ends
        f = smoothstep(np.arange(n) / 8.0)
        y = y + (1 - f) * (y0 - y[0]) + (1 - f[::-1]) * (y0 - y[-1])
        hard = leg_limits(rig, C, {s: REACH_HARD for s in SIDES})
        C["pel_pos"][:, 1] = np.minimum(y, np.minimum(hard["Left"], hard["Right"]))
    info = dict(psi=psi, ft=ft, plants=sorted(plants, key=lambda a: a[1]), stances=stances, inner=inner)
    return C, info


def build_turn360(rig, idle, sgn):
    idle0 = controls_at(idle["_C"], 0)
    C, info = _turn_controls(rig, idle0, sgn, "one")
    name = "Turn360_Left" if sgn > 0 else "Turn360_Right"
    ex = dict(footsteps=[dict(foot=s[0], t=float(pf / FPS)) for s, pf in info["plants"]], yaw=float(sgn * 360.0))
    clip = make_clip(rig, name, C, ex, loop=False, root_motion=True, cloth_init=idle["_cloth0"], settle=12)
    # exact Idle start pose at both ends (cloth joints included)
    for jn, tr in clip["tracks"].items():
        if jn == "Root":
            continue
        q0 = idle["tracks"][jn]["rotation"][0]
        for i in (0, -1):
            tr["rotation"][i] = q0 if np.dot(q0, tr["rotation"][i]) >= 0 else -q0
        if "translation" in tr:
            tr["translation"][0] = tr["translation"][-1] = idle["tracks"][jn]["translation"][0]
    clip["_info"] = info
    return clip


def build_turnloop(rig, idle, sgn):
    idle0 = controls_at(idle["_C"], 0)
    C, info = _turn_controls(rig, idle0, sgn, "loop")
    stp = TURN["step"]
    a, b = 2 * stp * 2, 2 * stp * 3                               # one period from the middle
    Cc = dict(n=b - a + 1)
    for k, v in C.items():
        if k != "n":
            Cc[k] = v[a:b + 1].copy()
    psi0 = Cc["root_yaw"][0]
    dpsi = Cc["root_yaw"][-1] - psi0
    T = (b - a) / FPS
    name = "TurnLoop_Left" if sgn > 0 else "TurnLoop_Right"
    ex = dict(turnRate=float(abs(dpsi) / DEG / T), turnDirection="left" if sgn > 0 else "right",
              footsteps=[dict(foot=s[0], t=float((pf - a) / FPS)) for s, pf in info["plants"] if a <= pf < b])
    clip = make_clip(rig, name, Cc, ex, loop=True, root_motion=False, cloth_cycle=(dpsi, np.zeros(3)), keep_root_yaw=False)
    clip["_info"] = dict(psi=Cc["root_yaw"], ft={s: {k: v[a:b + 1] for k, v in info["ft"][s].items()} for s in SIDES},
                         inner=info["inner"])
    return clip


# ================================================================================================ public API
def build_clips(skeleton_path, _raw=False):
    rig = Rig(skeleton_path)
    idle = build_idle(rig)
    walk, walk_rm = build_walk(rig)
    clips = [idle, walk, walk_rm,
             build_turn360(rig, idle, +1), build_turn360(rig, idle, -1),
             build_turnloop(rig, idle, +1), build_turnloop(rig, idle, -1)]
    if _raw:
        return rig, clips
    return [{k: v for k, v in c.items() if not k.startswith("_")} for c in clips]


def _summary(path):
    rig, clips = build_clips(path, _raw=True)
    print(f"skeleton {path}\n  height {rig.height:.3f}  leg L/R {rig.leg['Left'].L:.4f} / {rig.leg['Right'].L:.4f}")
    for c in clips:
        d = c["_sol"]["diag"]
        C = c["_C"]
        line = f"{c['name']:<16} {c['duration']:.3f}s  frames {C['n'] - 1:<4}"
        for s in SIDES:
            line += f" | {s[0]} reach {d[s]['reach'].min():.3f}..{d[s]['reach'].max():.4f} knee {d[s]['knee'].min():4.1f}..{d[s]['knee'].max():4.1f}"
        py = C["pel_pos"][:, 1]
        line += f" | pelvis y {py.min():.4f}..{py.max():.4f}"
        print(line)
        ex = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c["extras"].items() if k != "footsteps"}
        print("   extras", ex, "footsteps", [(f["foot"], round(f["t"], 3)) for f in c["extras"].get("footsteps", [])])
        if "_info" in c and "psi" in c["_info"]:
            inf = c["_info"]
            for s in SIDES:
                xi = np.degrees(inf["ft"][s]["yaw"] - (C["pel_yaw"] - C["pel_yaw"][0] + inf["psi"][0]))
                print(f"   {s} foot yaw - pelvis yaw: {xi.min():+.1f} .. {xi.max():+.1f} deg")
        if "_info" in c and "step" in c["_info"]:
            inf = c["_info"]
            print(f"   step {inf['step']:.4f} ({inf['step'] / rig.height:.3f} H)  lead {inf['lead']:.4f}  bob {inf['bob']:.4f}  speed {inf['speed']:.4f}"
                  f"  foot-roll scale {inf['tune']['lift']:.2f}  swing lift {inf['tune']['clear2']:.4f}")
    return rig, clips


if __name__ == "__main__":
    _summary(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "character3d", "rig", "skeleton.json"))
