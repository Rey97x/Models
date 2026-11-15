#!/usr/bin/env python3
"""
Split a GLB into plain web file types: one glTF JSON file (geometry / skin / animation buffer embedded as base64)
plus the textures as separate .jpg files.  This is standard glTF, loadable by any loader, and is what the viewer
(character3d/index.html) uses, because some static hosts do not serve .glb files.

  python3 tools/rig3d/make_gltf_json.py character3d/nomade3d_web.glb character3d/web
"""
import base64, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glb import GLB

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
g = GLB.load(src)
js = g.js
views = js["bufferViews"]
blob = lambda v: bytes(g.bin[v.get("byteOffset", 0): v.get("byteOffset", 0) + v["byteLength"]])
img_views = set()
for im in js.get("images", []):
    v = im.pop("bufferView")
    img_views.add(v)
    ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[im.pop("mimeType")]
    name = os.path.basename(src).split("_web")[0].split(".")[0] + "_" + im.get("name", "tex").split("_")[-1] + ext
    open(os.path.join(out, name), "wb").write(blob(views[v]))
    im["uri"] = name
remap, new_views, binary = {}, [], bytearray()
for i, v in enumerate(views):
    if i in img_views:
        continue
    while len(binary) % 4:
        binary.append(0)
    d = blob(v)
    nv = {k: val for k, val in v.items() if k not in ("byteOffset", "byteLength")}
    nv["byteOffset"], nv["byteLength"] = len(binary), len(d)
    binary.extend(d)
    remap[i] = len(new_views)
    new_views.append(nv)
for a in js["accessors"]:
    a["bufferView"] = remap[a["bufferView"]]
js["bufferViews"] = new_views
js["buffers"] = [{"byteLength": len(binary), "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(binary)).decode()}]
name = os.path.splitext(os.path.basename(src))[0] + ".gltf.json"
json.dump(js, open(os.path.join(out, name), "w"), separators=(",", ":"))
for f in sorted(os.listdir(out)):
    print(f"{f:<32} {os.path.getsize(os.path.join(out, f)) / 1e6:.2f} MB")
