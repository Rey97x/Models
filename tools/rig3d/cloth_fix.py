"""
Cloth post-process applied to finished clips: the sash hangs from the belt in front of the LEFT thigh, so when
that thigh swings forward it must push the sash forward instead of passing through it.

For every key the forward swing of the left thigh (relative to the pelvis) is measured and the Sash1 joint is
rotated forward so that it stays at least `gain` x that angle in front of its rest direction.  A backward swing
leaves the sash hanging.  Works on any clip dict produced by animate.build_clips (identity-rest skeleton).
"""
import numpy as np
from glb import q_mul, q_conj, q_rotate, q_axis_angle


def _fwd_angle(d):
    """angle (rad) of a direction in the pelvis YZ plane: 0 = straight down, positive = forward (+Z)."""
    return np.arctan2(d[2], -d[1])


def sash_follow_thigh(clip, joints, gain=0.9, soft=0.06):
    heads = {j["name"]: np.asarray(j["head"], float) for j in joints}
    tr = clip["tracks"]
    if "LeftUpLeg" not in tr or tr["LeftUpLeg"].get("rotation") is None:
        return clip
    n = len(tr["LeftUpLeg"]["t"])
    thigh0 = heads["LeftLeg"] - heads["LeftUpLeg"]
    sash0 = heads["Sash2"] - heads["Sash1"]
    a_thigh0, a_sash0 = _fwd_angle(thigh0), _fwd_angle(sash0)
    q_thigh = np.asarray(tr["LeftUpLeg"]["rotation"], float)
    if "Sash1" not in tr or tr["Sash1"].get("rotation") is None:
        tr["Sash1"] = {"t": tr["LeftUpLeg"]["t"], "rotation": np.tile([0, 0, 0, 1.0], (n, 1))}
    q_sash = np.asarray(tr["Sash1"]["rotation"], float).copy()
    if len(q_sash) != n:
        return clip
    for i in range(n):
        # both joints are children of Hips, so their local rotations are already in the pelvis frame
        flex = _fwd_angle(q_rotate(q_thigh[i], thigh0)) - a_thigh0
        cur = _fwd_angle(q_rotate(q_sash[i], sash0)) - a_sash0
        want = gain * flex
        # smooth maximum of (cur, want): no visible snap when the thigh reaches the cloth
        target = soft * np.log(np.exp(cur / soft) + np.exp(want / soft))
        delta = target - cur
        if delta > 1e-5:
            q_sash[i] = q_mul(q_axis_angle([1, 0, 0], -delta), q_sash[i])
    tr["Sash1"]["rotation"] = q_sash
    return clip
