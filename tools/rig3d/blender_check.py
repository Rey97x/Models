#!/usr/bin/env python3
"""
Interop check with Blender (bpy module): import the GLB, report armature / skin / actions, optionally export FBX / .blend.

  python3 tools/rig3d/blender_check.py character3d/nomade3d_rigged.glb [--fbx out.fbx] [--blend out.blend]
"""
import argparse, sys
import bpy

ap = argparse.ArgumentParser()
ap.add_argument("glb"); ap.add_argument("--fbx"); ap.add_argument("--blend")
a = ap.parse_args()
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.fps = 30
bpy.ops.import_scene.gltf(filepath=a.glb)
# the importer adds an "Icosphere" helper object used as custom bone shape: not part of the model
for o in [o for o in bpy.data.objects if o.type == "MESH" and not o.vertex_groups and o.name.startswith("Icosphere")]:
    for pb in [pb for arm in bpy.data.objects if arm.type == "ARMATURE" for pb in arm.pose.bones]:
        pb.custom_shape = None
    bpy.data.objects.remove(o, do_unlink=True)
ok = True
arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
print(f"objects: {len(bpy.data.objects)}  armatures: {len(arms)}  meshes: {len(meshes)}")
for arm in arms:
    print(f"armature '{arm.name}': {len(arm.data.bones)} bones; roots: {[b.name for b in arm.data.bones if b.parent is None]}")
for m in meshes:
    mods = [md for md in m.modifiers if md.type == "ARMATURE"]
    nvg = len(m.vertex_groups)
    unweighted = sum(1 for v in m.data.vertices if not v.groups)
    print(f"mesh '{m.name}': {len(m.data.vertices)} verts, {len(m.data.polygons)} faces, vertex groups {nvg}, armature modifiers {len(mods)}, unweighted verts {unweighted}, dims {[round(d, 3) for d in m.dimensions]}")
    if m.parent_type == "BONE":                 # rigid attachment (weapon in a hand): no skin expected
        print(f"   attached to bone '{m.parent_bone}'")
    elif not mods or unweighted:
        ok = False
print(f"actions: {len(bpy.data.actions)}")
for act in bpy.data.actions:
    fr = act.frame_range
    print(f"  {act.name:<20} frames {fr[0]:.0f}-{fr[1]:.0f}  ({(fr[1]-fr[0])/bpy.context.scene.render.fps:.3f}s @ {bpy.context.scene.render.fps} fps)")
if not arms or not bpy.data.actions:
    ok = False
if a.fbx:
    bpy.ops.export_scene.fbx(filepath=a.fbx, add_leaf_bones=False, bake_anim=True, bake_anim_use_all_actions=True,
                             bake_anim_use_nla_strips=False, path_mode="COPY", embed_textures=True, axis_forward="-Z", axis_up="Y")
    print("exported", a.fbx)
if a.blend:
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=a.blend)
    print("saved", a.blend)
print("BLENDER CHECK:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
