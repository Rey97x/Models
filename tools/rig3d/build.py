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
import argparse, json, os, sys
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
        if "dup_src" in z.files and len(z["dup_src"]):
            # the skinning step cut the mesh (cape fused with legs): duplicated border vertices + new triangles
            md, dup = g.mesh_data(), z["dup_src"].astype(np.int64)
            cat = lambda a: None if a is None else np.concatenate([a, a[dup]])
            g.set_mesh_arrays(cat(md["positions"]), cat(md.get("normals")), cat(md.get("uvs")), z["triangles"])
            P = g.mesh_data()["positions"].astype(np.float64)
            print(f"mesh cut: {len(dup)} vertices added")
        assert idx.shape == (len(P), 4) and w.shape == (len(P), 4)
        print("skin: weights.npz")
    else:
        from quick_skin import quick_weights
        pts = {j["name"]: j["head"] for j in joints}; pts.update(ends)
        idx, w = quick_weights(P, pts)
        print("skin: QUICK PREVIEW SKIN (not final)")
    # optional per-character configuration next to the skeleton: weapons, weight class (see CONTRACT.md)
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(a.skeleton)), "character.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    sockets, grip_pose = {}, {}
    body_joints = joints                                   # the skeleton the animation code knows
    if cfg.get("weapons"):
        from weapons import compute_socket, add_weapon, add_grip_joints
        names = [j["name"] for j in joints]
        idx, w = np.asarray(idx).astype(np.int64), np.asarray(w, np.float64)
        for wp in cfg["weapons"]:
            side, kind = wp["hand"], wp["type"]
            hw = np.where(idx == names.index(side + "Hand"), w, 0).sum(1)
            sockets[side] = dict(compute_socket(P, hw, joints, ends, side, kind, wp), kind=kind)
            if wp.get("closeHand", cfg.get("closeHand", True)):
                joints, idx, w, pose = add_grip_joints(P, joints, idx, w, side, sockets[side])
                grip_pose.update(pose)
    node_of = g.add_skin(joints, idx, w, name="NomadeArmature")
    for side, sk in sockets.items():
        add_weapon(g, node_of[side + "Hand"], side, sk["kind"], sk)
        print(f"weapon: {sk['kind']} in the {side.lower()} hand" + (" (fingers closed around the grip)" if side + "Thumb" in grip_pose else ""))
    g.js["asset"]["generator"] = "Tripo + rig3d"
    if not a.no_anim and os.path.exists(os.path.join(HERE, "animate.py")):
        import animate
        # a hand that holds a weapon hangs a little further from the thigh
        animate.ARM["abd"] = 11.0 + float(cfg.get("armSpread", 8.0 if cfg.get("weapons") else 0.0))
        clips = animate.build_clips(a.skeleton)
        if cfg.get("weightClass") or cfg.get("weapons"):
            import combat
            from audit_pose import capsules_from_arrays
            names = [j["name"] for j in joints]
            pts = {j["name"]: np.asarray(j["head"], float) for j in body_joints}
            pts.update({k: np.asarray(v, float) for k, v in list(ends.items()) + list(lms.items())})
            caps = capsules_from_arrays(P, idx[np.arange(len(P)), np.argmax(w, 1)], names, pts)
            clips += combat.build_combat_clips(a.skeleton, cfg, sockets, caps)
        want = a.clips.split(",") if a.clips else None
        from cloth_fix import sash_follow_thigh, cape_follow_legs
        for c in clips:
            if want and c["name"] not in want:
                continue
            if any(j["name"] == "Sash1" for j in joints):
                sash_follow_thigh(c, body_joints)
            if any(j["name"] == "Cape1" for j in joints):
                cape_follow_legs(c, body_joints, ends, lms)
            for jn, q in grip_pose.items():             # the hand stays closed around the grip in every clip
                tt = c["tracks"]["Hips"]["t"]             # same keys as the rest of the clip
                c["tracks"][jn] = {"t": tt, "rotation": np.tile(q, (len(tt), 1))}
            g.add_animation(c["name"], c["tracks"], extras=c.get("extras"))
            print(f"clip {c['name']:<18} {c['duration']:.3f}s  tracks={len(c['tracks'])}  extras={c.get('extras')}")
    g.save(a.out)
    print("wrote", a.out, f"{os.path.getsize(a.out)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
