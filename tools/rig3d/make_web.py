#!/usr/bin/env python3
"""
Make a lighter copy of a GLB for web / mobile: textures are downscaled (default 2048 px) and re-encoded as JPEG,
the binary buffer is repacked so no orphaned bytes remain.  Geometry, skin and animations are untouched.

  python3 tools/rig3d/make_web.py character3d/nomade3d_rigged.glb character3d/nomade3d_web.glb [--size 2048] [--quality 88]
"""
import argparse, io, os, sys
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glb import GLB

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--size", type=int, default=2048); ap.add_argument("--quality", type=int, default=88)
a = ap.parse_args()

g = GLB.load(a.src)
js = g.js
views = js["bufferViews"]
data = [bytes(g.bin[v.get("byteOffset", 0): v.get("byteOffset", 0) + v["byteLength"]]) for v in views]
for im in js.get("images", []):
    if "bufferView" not in im:
        continue
    pic = Image.open(io.BytesIO(data[im["bufferView"]])).convert("RGB")
    if max(pic.size) > a.size:
        pic = pic.resize((a.size, a.size * pic.height // pic.width), Image.LANCZOS)
    buf = io.BytesIO()
    pic.save(buf, "JPEG", quality=a.quality, optimize=True)
    data[im["bufferView"]] = buf.getvalue()
    im["mimeType"] = "image/jpeg"
out = bytearray()
for v, d in zip(views, data):
    while len(out) % 4:
        out.append(0)
    v["byteOffset"] = len(out)
    v["byteLength"] = len(d)
    out.extend(d)
g.bin = out
g.save(a.dst)
print(f"{a.src} {os.path.getsize(a.src)/1e6:.2f} MB -> {a.dst} {os.path.getsize(a.dst)/1e6:.2f} MB")
