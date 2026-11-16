#!/usr/bin/env python3
"""
Rig + animate one character with a single command.

  python3 tools/rig3d/pipeline.py <character dir> [--name NAME] [--title "Display name"] [--no-fbx] [--no-sheets]

Layout of a character directory
  <dir>/src/<model>.glb        static source mesh (A/T-pose, faces +Z, feet on y=0, ~1 unit tall)
  <dir>/rig/character.json     optional: {"weightClass": "heavy"|"light", "weapons": [{"hand": "Right", "type": "sword"|"dagger"}]}
                               adds the weapons and the combat clips (Attack, Dodge_Back, Dash_Forward)
  <dir>/rig/skeleton.json      joint positions.  If it is missing, a first guess is written (auto_skeleton.py) together
                               with an overlay image, and the pipeline STOPS so the skeleton can be checked and fixed.
Outputs
  <dir>/rig/weights.npz, <dir>/<name>_rigged.glb, <dir>/<name>_web.glb, <dir>/web/ (glTF JSON + jpg for the viewer),
  <dir>/<name>_rigged.fbx, <dir>/index.html (viewer), <dir>/checks/*.png (contact sheets to look at)
Every check must pass (glTF validator, audit_feet, check_anim, Blender import); the script exits non-zero otherwise.
"""
import argparse, glob, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))


def run(*cmd, check=True):
    print("$", " ".join(os.path.relpath(c) if os.path.exists(c) else c for c in cmd), flush=True)
    r = subprocess.run(cmd)
    if check and r.returncode:
        sys.exit(f"FAILED: {cmd[1] if len(cmd) > 1 else cmd[0]}")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("char"); ap.add_argument("--name"); ap.add_argument("--title")
    ap.add_argument("--no-fbx", action="store_true"); ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args()
    d = os.path.abspath(a.char)
    srcs = sorted(glob.glob(os.path.join(d, "src", "*.glb")))
    if len(srcs) != 1:
        sys.exit(f"expected exactly one .glb in {d}/src, found {len(srcs)}")
    src = srcs[0]
    name = a.name or os.path.basename(d.rstrip("/"))
    py, T = sys.executable, lambda f: os.path.join(HERE, f)
    os.makedirs(os.path.join(d, "rig"), exist_ok=True)
    os.makedirs(os.path.join(d, "checks"), exist_ok=True)
    skel = os.path.join(d, "rig", "skeleton.json")
    if not os.path.exists(skel):
        run(py, T("auto_skeleton.py"), src, skel)
        run(py, T("preview_skeleton.py"), skel, os.path.join(d, "checks", "skeleton_overlay.png"), "--glb", src, "--views", "front,left,back,top")
        print(f"\nSTOP: first-guess skeleton written to {skel}.\nLook at {d}/checks/skeleton_overlay.png (and close-ups), "
              "fix the joints that are off, add cloth chains if needed, then run the pipeline again.")
        return
    weights = os.path.join(d, "rig", "weights.npz")
    glb, web = os.path.join(d, name + "_rigged.glb"), os.path.join(d, name + "_web.glb")
    run(py, T("skin_weights.py"), "--src", src, "--skeleton", skel, "--out", weights)
    run(py, T("build.py"), "--src", src, "--skeleton", skel, "--weights", weights, "--out", glb)
    run("node", T("validate.cjs"), glb)
    run(py, T("audit_feet.py"), glb, "--skeleton", skel)
    run(py, T("check_anim.py"), glb, "--skeleton", skel)
    run(py, T("make_web.py"), glb, web)
    run(py, T("make_gltf_json.py"), web, os.path.join(d, "web"))
    if not a.no_fbx:
        run(py, T("blender_check.py"), glb, "--fbx", os.path.join(d, name + "_rigged.fbx"))
    index = os.path.join(d, "index.html")
    if not os.path.exists(index):
        html = open(T("viewer_template.html")).read().replace("__TITLE__", a.title or name).replace("__NAME__", name)
        open(index, "w").write(html)
        print("wrote", index)
    if not a.no_sheets:
        c = os.path.join(d, "checks")
        poses = os.path.join(c, "stress_poses.json")
        json.dump({"arms down 40": {"LeftArm": [0, 0, -40], "RightArm": [0, 0, 40]},
                   "arms up 40": {"LeftArm": [0, 0, 40], "RightArm": [0, 0, -40]},
                   "stride": {"LeftUpLeg": [-30, 0, 0], "LeftLeg": [25, 0, 0], "RightUpLeg": [20, 0, 0], "RightLeg": [10, 0, 0]},
                   "knee 65": {"LeftUpLeg": [-25, 0, 0], "LeftLeg": [65, 0, 0]},
                   "thigh twist": {"LeftUpLeg": [0, 35, 8], "RightUpLeg": [0, -35, -8]},
                   "head + spine": {"Spine": [0, 10, 0], "Spine1": [0, 10, 0], "Spine2": [0, 10, 0], "Head": [10, -40, 0]}}, open(poses, "w"))
        R = T("render.py")
        run(py, R, glb, os.path.join(c, "weights.png"), "--mode", "weights", "--views", "front,back,left,q")
        run(py, R, glb, os.path.join(c, "stress_poses.png"), "--poses", poses, "--views", "front,left,back", "--cell", "320x420")
        run(py, R, glb, os.path.join(c, "idle.png"), "--clip", "Idle", "--frames", "4", "--views", "front,left", "--cell", "320x420")
        run(py, R, glb, os.path.join(c, "walk.png"), "--clip", "Walk", "--frames", "8", "--views", "left,front", "--grid", "--cell", "300x400")
        run(py, R, glb, os.path.join(c, "turn360.png"), "--clip", "Turn360_Left", "--frames", "10", "--views", "q,top", "--grid", "--cell", "260x340")
    if not a.no_sheets and os.path.exists(os.path.join(d, "rig", "character.json")):
        c, R = os.path.join(d, "checks"), T("render.py")
        run(py, R, glb, os.path.join(c, "attack.png"), "--clip", "Attack", "--frames", "10", "--views", "q,left", "--cell", "300x400")
        run(py, R, glb, os.path.join(c, "dodge_back.png"), "--clip", "Dodge_Back", "--frames", "8", "--views", "left", "--grid",
            "--zoom", "0.62", "--center=0,0.45,-0.3", "--cell", "330x280")
        run(py, R, glb, os.path.join(c, "dash_forward.png"), "--clip", "Dash_Forward", "--frames", "8", "--views", "left", "--grid",
            "--zoom", "0.62", "--center=0,0.45,0.3", "--cell", "330x280")
    print(f"\nDONE: {glb}\nNow LOOK at {d}/checks/*.png before shipping (skin spikes, cloth through legs, arms through body).")


if __name__ == "__main__":
    main()
