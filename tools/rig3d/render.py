#!/usr/bin/env python3
"""
Render contact sheets of a GLB (static / skinned / animated) with headless three.js.

Examples
  # static model, four views
  python3 tools/rig3d/render.py model.glb out.png --views front,left,back,q

  # 8 evenly spaced frames of a clip, side + front, with ground grid and skeleton overlay
  python3 tools/rig3d/render.py rigged.glb walk.png --clip Walk --frames 8 --views left,front --grid --skeleton

  # explicit times (seconds)
  python3 tools/rig3d/render.py rigged.glb t.png --clip Turn360_L --times 0,0.4,0.8,1.2 --views top,q --grid

  # stress poses: JSON file {"label": {"JointName": [ex,ey,ez degrees], ...}, ...}
  python3 tools/rig3d/render.py rigged.glb poses.png --poses poses.json --views front,left --mode weights

  # list clips / bones
  python3 tools/rig3d/render.py rigged.glb --info

Views: front (camera on +Z, looking at the face), back, left (camera on the character's left, +X), right,
       top, q / ql (3/4 front), qb (3/4 back), or "azimuth:elevation" in degrees.
Modes: pbr (textured), flat (grey clay), wire, weights (one hue per joint; --weight-joint NAME shows one joint in red).
The camera is orthographic and FIXED in world space (use --follow Hips to track a node), so foot sliding
against the --grid (0.1-unit squares) is directly visible.
Rows = views, columns = times/poses.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--clip")
    ap.add_argument("--frames", type=int, help="N evenly spaced samples over the clip (end excluded)")
    ap.add_argument("--times", help="comma separated seconds")
    ap.add_argument("--views", default="front,left")
    ap.add_argument("--poses", help="JSON file: {label: {joint: [ex,ey,ez]}}")
    ap.add_argument("--pose", help="inline JSON for a single pose {joint: [ex,ey,ez]}")
    ap.add_argument("--mode", default="pbr", choices=["pbr", "flat", "wire", "weights"])
    ap.add_argument("--weight-joint")
    ap.add_argument("--skeleton", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--follow", help="node name the camera follows in XZ")
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--center", help="x,y,z look-at point")
    ap.add_argument("--cell", default="420x560", help="cell size WxH")
    a = ap.parse_args()

    w, h = [int(v) for v in a.cell.split("x")]
    tmp = tempfile.mkdtemp(prefix="rig3d_render_")
    job = {"glb": os.path.abspath(a.glb), "outDir": tmp, "size": [w, h], "zoom": a.zoom, "skeleton": a.skeleton,
           "grid": a.grid, "mode": a.mode, "follow": a.follow, "weightJoint": a.weight_joint, "shots": []}
    if a.center:
        job["center"] = [float(v) for v in a.center.split(",")]

    def run():
        jp = os.path.join(tmp, "job.json")
        json.dump(job, open(jp, "w"))
        r = subprocess.run(["node", os.path.join(HERE, "render.cjs"), jp], capture_output=True, text=True)
        if r.stderr.strip():
            print(r.stderr.strip(), file=sys.stderr)
        if r.returncode:
            sys.exit(r.returncode)
        return json.load(open(os.path.join(tmp, "info.json")))

    if a.info or (a.clip and a.frames):
        info = run()
        if a.info:
            print(json.dumps(info, indent=1))
            return
    views = a.views.split(",")
    cols = []   # (label, clip, time, pose)
    if a.poses:
        for label, pose in json.load(open(a.poses)).items():
            cols.append((label, a.clip, 0.0, pose))
    elif a.clip:
        if a.times:
            ts = [float(t) for t in a.times.split(",")]
        else:
            dur = next(c["duration"] for c in info["clips"] if c["name"] == a.clip)
            n = a.frames or 8
            ts = [dur * i / n for i in range(n)]
        cols = [(f"{a.clip} t={t:.3f}s", a.clip, t, None) for t in ts]
    else:
        cols = [("rest" if not a.pose else "pose", None, 0.0, json.loads(a.pose) if a.pose else None)]
    for r_, v in enumerate(views):
        for c_, (label, clip, t, pose) in enumerate(cols):
            job["shots"].append({"file": f"s_{r_}_{c_}.png", "clip": clip, "time": t, "view": v, "pose": pose})
    run()

    from PIL import Image, ImageDraw
    flip = len(cols) == 1          # a single column is laid out horizontally (one cell per view)
    nx, ny = (len(views), 1) if flip else (len(cols), len(views))
    sheet = Image.new("RGB", (w * nx, h * ny), "white")
    d = ImageDraw.Draw(sheet)
    for r_, v in enumerate(views):
        for c_, (label, *_rest) in enumerate(cols):
            im = Image.open(os.path.join(tmp, f"s_{r_}_{c_}.png")).convert("RGB")
            x, y = (r_ * w, 0) if flip else (c_ * w, r_ * h)
            sheet.paste(im, (x, y))
            d.text((x + 6, y + 4), f"{label} | {v}", fill=(0, 0, 0))
    sheet.save(a.out)
    print("wrote", a.out, sheet.size)


if __name__ == "__main__":
    main()
