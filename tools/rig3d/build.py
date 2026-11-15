#!/usr/bin/env python3
"""
Assemble the rigged + animated GLB.

  python3 tools/rig3d/build.py [--out character3d/nomade3d_rigged.glb] [--skeleton character3d/rig/skeleton.json]
                               [--weights character3d/rig/weights.npz] [--no-anim] [--quick-skin] [--clips Idle,Walk]

  source mesh  : character3d/src/desert_assassin_tripo.glb   (never modified)
  skeleton     : character3d/rig/skeleton.json               ({"points": {name: [x,y,z]}}, see skeleton_def.py)
  skin weights : character3d/rig/weights.npz                 (joint_names, joint_idx (N,4), weights (N,4)); if the file is
                 missing or --quick-skin is given, a crude preview skin (quick_skin.py) is used instead
  animations   : tools/rig3d/animate.py  ->  build_clips(skeleton_path) returns a list of
                 {"name", "duration", "tracks": {joint: {"t", "rotation"?, "translation"?}}, "extras": {...}}
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from glb import GLB
from skeleton_def import JOINT_NAMES, load_skeleton


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "character3d/src/desert_assassin_tripo.glb"))
    ap.add_argument("--skeleton", default=os.path.join(ROOT, "character3d/rig/skeleton.json"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "character3d/rig/weights.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "character3d/nomade3d_rigged.glb"))
    ap.add_argument("--no-anim", action="store_true")
    ap.add_argument("--quick-skin", action="store_true")
    ap.add_argument("--clips", help="comma separated subset of clip names")
    a = ap.parse_args()

    g = GLB.load(a.src)
    P = g.mesh_data()["positions"].astype(np.float64)
    joints, ends, lms = load_skeleton(a.skeleton)
    if os.path.exists(a.weights) and not a.quick_skin:
        z = np.load(a.weights, allow_pickle=False)
        assert [str(n) for n in z["joint_names"]] == [j["name"] for j in joints], "weights.npz joints differ from skeleton.json"
        idx, w = z["joint_idx"], z["weights"]
        assert idx.shape == (len(P), 4) and w.shape == (len(P), 4)
        print("skin: weights.npz")
    else:
        from quick_skin import quick_weights
        pts = {j["name"]: j["head"] for j in joints}; pts.update(ends)
        idx, w = quick_weights(P, pts)
        print("skin: QUICK PREVIEW SKIN (not final)")
    g.add_skin(joints, idx, w, name="NomadeArmature")
    g.js["asset"]["generator"] = "Tripo + rig3d"
    if not a.no_anim and os.path.exists(os.path.join(HERE, "animate.py")):
        import animate
        clips = animate.build_clips(a.skeleton)
        want = a.clips.split(",") if a.clips else None
        from cloth_fix import sash_follow_thigh, cape_follow_legs
        for c in clips:
            if want and c["name"] not in want:
                continue
            if any(j["name"] == "Sash1" for j in joints):
                sash_follow_thigh(c, joints)
            if any(j["name"] == "Cape1" for j in joints):
                cape_follow_legs(c, joints, ends, lms)
            g.add_animation(c["name"], c["tracks"], extras=c.get("extras"))
            print(f"clip {c['name']:<18} {c['duration']:.3f}s  tracks={len(c['tracks'])}  extras={c.get('extras')}")
    g.save(a.out)
    print("wrote", a.out, f"{os.path.getsize(a.out)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
