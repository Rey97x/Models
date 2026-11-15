# rig3d pipeline contract

Quick start for a new character: `python3 tools/rig3d/pipeline.py <character dir>` (see `CLAUDE.md` at the repo root).
`auto_skeleton.py` + `template_skeleton.json` give the first-guess skeleton; the cloth joints
(`Left/RightRibbon1-2`, `Sash1-2`, `Cape1-2` with their `_End` points) are OPTIONAL and every tool works without them.
`slice_view.py` draws horizontal cross-sections with the joints marked.
The rest of this file documents the first character (`character3d/`) and the shared conventions.

Goal: turn the static Tripo mesh `character3d/src/desert_assassin_tripo.glb` (desert nomad / assassin, A-pose,
63 492 vertices, 112 781 triangles, one watertight connected shell, 4K PBR textures) into a rigged, skinned and
animated GLB: `character3d/nomade3d_rigged.glb`.

## Coordinate conventions (verified by rendering)
- glTF: **+Y up**, the character **faces +Z**, the character's **LEFT is +X**, right is -X.
- Units are the source model's units: the character is **0.979 tall** (feet on y = 0), ~0.61 wide in A-pose.
  1 unit ≈ 1.8 m of a real person, so 0.001 units ≈ 1.8 mm.
- Every joint has an **identity rest rotation** (joint axes are world-aligned in the bind pose);
  rest local translation = `head - parent.head`.  Inverse bind matrix = translation(-head).
- Quaternions are glTF order `[x, y, z, w]`.  `world = parent_world * local`.
- Rotation signs in the bind pose (right-handed, world-aligned axes):
  - about **+X**: negative = swing a leg/arm **forward** (hip flexion), positive = backward; knee flexion is **positive X**
    on `LeftLeg`/`RightLeg`.
  - about **+Y**: positive = yaw to the character's **left** (counter-clockwise seen from above).
  - about **+Z**: on `LeftArm` negative lowers the arm towards the body; on `RightArm` positive lowers it.

## Skeleton (`tools/rig3d/skeleton_def.py`, `character3d/rig/skeleton.json`)
29 joints, names fixed:
`Root > Hips > Spine > Spine1 > Spine2 > Neck > Head`,
`Spine2 > Left/RightShoulder > Arm > ForeArm > Hand`,
`Hips > Left/RightUpLeg > Leg > Foot > ToeBase`,
cloth helpers `Left/RightForeArm > Ribbon1 > Ribbon2` (strips hanging from the wrist wraps), `Hips > Sash1 > Sash2`
(cloth hanging from the belt on the character's left hip).
`skeleton.json` = `{"points": {name: [x, y, z]}}` holding every joint head plus end sites
(`Head_End`, `Left/RightHand_End`, `Left/RightToe_End`, `Left/RightRibbon_End`, `Sash_End`) and the landmarks
`LeftHeel` / `RightHeel` (bottom-back of the heel on the ground).  `skeleton_def.load_skeleton(path)` parses it.
The skeleton is NOT guaranteed to be perfectly left/right symmetric - always read positions from the file,
never hard-code coordinates.

## Shared tools (do not change their interfaces)
| tool | purpose |
|---|---|
| `tools/rig3d/glb.py` | `GLB.load/save`, `mesh_data()`, `add_skin(joints, joint_idx, weights)`, `add_animation(name, tracks, extras)`, quaternion helpers `q_mul q_conj q_axis_angle q_rotate q_from_matrix q_to_matrix`, `fk(joints, local_rot, local_pos)` |
| `tools/rig3d/build.py` | assembles source mesh + skeleton.json + `character3d/rig/weights.npz` (falls back to a crude preview skin) + clips from `tools/rig3d/animate.py::build_clips(skeleton_path)` into a GLB (`--out`, `--no-anim`, `--quick-skin`, `--clips A,B`) |
| `tools/rig3d/render.py` | headless three.js contact sheets: `--clip NAME --frames N` / `--times a,b,c`, `--views front,left,back,right,top,q,qb,az:el`, `--poses file.json`, `--pose '{...}'`, `--mode pbr|flat|wire|weights`, `--weight-joint NAME`, `--skeleton`, `--grid` (0.1-unit squares, world-fixed camera), `--follow NODE`, `--zoom`, `--center x,y,z`, `--cell WxH`, `--info`.  "front" = camera on +Z looking at the face; "left" = camera on +X (the character then faces screen-left); in "top" the character's front points screen-down. |
| `tools/rig3d/preview_skeleton.py` | overlay a `{name: [x,y,z]}` point set on the mesh |
| `tools/rig3d/validate.cjs` | Khronos glTF validator (`node tools/rig3d/validate.cjs file.glb`) |
Blender is available headless as a Python module (`import bpy`, version 5.3) for import / interop checks.
Python has numpy, scipy, pillow.  Rendering uses software WebGL (a sheet takes a few seconds).

## Intermediate files
- `character3d/rig/weights.npz`: `joint_names` (str array, must equal `skeleton_def.JOINT_NAMES`),
  `joint_idx` (N,4) uint16 indices into JOINT_NAMES, `weights` (N,4) float32 (rows sum to 1, sorted descending),
  N = 63 492 in source vertex order.
- clip dict (returned by `animate.build_clips`): `{"name", "duration", "tracks": {joint: {"t": (K,), "rotation": (K,4)?, "translation": (K,3)?}}, "extras": {...}}`
  - uniform 30 fps sampling, first key at t = 0, last key at t = duration; for looping clips the last key equals the first
    (except accumulated root motion)
  - `rotation` = local quaternion (rest = identity); `translation` = ABSOLUTE local translation (rest offset + delta),
    normally only on `Root` and `Hips`.

## Clips to deliver
| name | type | notes |
|---|---|---|
| `Idle` | loop | relaxed standing, arms lowered from the A-pose, breathing, subtle weight shift |
| `Walk` | in-place loop | feet move backwards at a constant speed during stance; `extras.speed` = ground speed (units/s) at which the feet do not slide |
| `Walk_RootMotion` | loop with root motion | same motion, `Root` translates along +Z by exactly one stride per cycle; stance foot fixed in world space |
| `Turn360_Left`, `Turn360_Right` | one-shot, root motion | full 360° turn on the spot starting and ending in the Idle start pose; the yaw is on `Root` (0 → ±360°); the feet step around and stay planted between steps |
| `TurnLoop_Left`, `TurnLoop_Right` | in-place loop | steady stepping turn with NO root yaw, for engines that rotate the character in code; `extras.turnRate` = deg/s at which the feet do not slide |
Left turn = counter-clockwise seen from above = positive yaw about +Y.

## Foot-contact acceptance criteria (model units)
- While a foot is in flat stance its `Foot` and `ToeBase` joints stay fixed in world space: drift ≤ 0.0005
  (for in-place clips: after applying the declared constant `speed` / `turnRate` to the root).
- Sole points (heel landmark, ball, toe tip rigidly attached to Foot / ToeBase) never go below y = -0.001.
- No knee pops: the leg never reaches full extension abruptly (IK reach ≤ ~0.995 of leg length, smooth knee angle).
- Loops: pose at t = duration equals pose at t = 0 (≤ 1e-4), velocities continuous across the seam.
