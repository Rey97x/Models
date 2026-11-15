#!/usr/bin/env python3
"""
Horizontal cross-sections of a mesh (solid voxels seen from above) with the skeleton joints marked: the quickest way
to centre joints inside limbs and to tell the body from capes / skirts.

  python3 tools/rig3d/slice_view.py <src.glb> <skeleton.json> out.png [--ys 0.05,0.15,0.28,...]
Each panel: x to the right (character's left = right side of the panel), +z (front) DOWN, grid lines every 0.05,
red dots = joints within 0.03 of the slice height, labelled.  Dark = solid interior, light = thin shell / cloth.
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glb import GLB
from skin_weights import voxelize

ap = argparse.ArgumentParser()
ap.add_argument("glb"); ap.add_argument("skeleton"); ap.add_argument("out")
ap.add_argument("--ys", default="0.03,0.10,0.18,0.28,0.38,0.46,0.51,0.58,0.66,0.73,0.78,0.83,0.87,0.91")
a = ap.parse_args()
md = GLB.load(a.glb).mesh_data()
P, T = md["positions"].astype(float), md["triangles"].astype(np.int64)
h = 0.003
solid, lo = voxelize(P, T, h)
edt = ndi.distance_transform_edt(solid) * h
pts = json.load(open(a.skeleton))["points"]
ys = [float(v) for v in a.ys.split(",")]
S = 3                                    # pixels per voxel
nx, nz = solid.shape[0], solid.shape[2]
cols = 5
rows = (len(ys) + cols - 1) // cols
sheet = Image.new("RGB", (cols * nx * S, rows * (nz * S + 18)), "white")
d = ImageDraw.Draw(sheet)
for k, y in enumerate(ys):
    j = int((y - lo[1]) / h)
    sl, e = solid[:, j, :], edt[:, j, :]
    img = np.full((nz, nx, 3), 255, np.uint8)
    img[sl.T] = (200, 205, 215)
    img[(e.T > 0.012)] = (90, 100, 120)
    im = Image.fromarray(img).resize((nx * S, nz * S), Image.NEAREST)
    ox, oy = (k % cols) * nx * S, (k // cols) * (nz * S + 18) + 18
    sheet.paste(im, (ox, oy))
    to_px = lambda x, z: (ox + (x - lo[0]) / h * S, oy + (z - lo[2]) / h * S)
    for g in np.arange(-0.3, 0.31, 0.05):
        x0, _ = to_px(g, 0); d.line([(x0, oy), (x0, oy + nz * S)], fill=(255, 190, 120) if abs(g) < 1e-9 else (225, 225, 225))
    for g in np.arange(-0.1, 0.11, 0.05):
        _, z0 = to_px(0, g); d.line([(ox, z0), (ox + nx * S, z0)], fill=(255, 190, 120) if abs(g) < 1e-9 else (225, 225, 225))
    d.text((ox + 4, oy - 16), f"y={y:.2f}  (front is down)", fill=(0, 0, 0))
    for n, p in pts.items():
        if abs(p[1] - y) < 0.03 and n != "Root":
            px, pz = to_px(p[0], p[2])
            d.ellipse([px - 3, pz - 3, px + 3, pz + 3], fill=(230, 30, 30))
            d.text((px + 4, pz - 5), n.replace("Left", "L").replace("Right", "R"), fill=(160, 0, 0))
sheet.save(a.out)
print("wrote", a.out, sheet.size)
