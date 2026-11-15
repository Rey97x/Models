"""Joint hierarchy shared by the whole rig3d pipeline (names follow the common Mixamo-style convention)."""

# name -> parent   (parents listed before children)
HIERARCHY = [
    ("Root", None),            # on the ground under the pelvis (0, 0, 0): carries root motion
    ("Hips", "Root"),          # pelvis centre
    ("Spine", "Hips"), ("Spine1", "Spine"), ("Spine2", "Spine1"),
    ("Neck", "Spine2"), ("Head", "Neck"),
    ("LeftShoulder", "Spine2"), ("LeftArm", "LeftShoulder"), ("LeftForeArm", "LeftArm"), ("LeftHand", "LeftForeArm"),
    ("RightShoulder", "Spine2"), ("RightArm", "RightShoulder"), ("RightForeArm", "RightArm"), ("RightHand", "RightForeArm"),
    ("LeftUpLeg", "Hips"), ("LeftLeg", "LeftUpLeg"), ("LeftFoot", "LeftLeg"), ("LeftToeBase", "LeftFoot"),
    ("RightUpLeg", "Hips"), ("RightLeg", "RightUpLeg"), ("RightFoot", "RightLeg"), ("RightToeBase", "RightFoot"),
    # cloth helpers
    ("LeftRibbon1", "LeftForeArm"), ("LeftRibbon2", "LeftRibbon1"),
    ("RightRibbon1", "RightForeArm"), ("RightRibbon2", "RightRibbon1"),
    ("Sash1", "Hips"), ("Sash2", "Sash1"),
]
JOINT_NAMES = [n for n, _ in HIERARCHY]
PARENT = dict(HIERARCHY)

# leaf "tail" points: not joints, only used for bone segments (skin weights) and IK / foot geometry
END_SITES = {
    "Head_End": "Head", "LeftHand_End": "LeftHand", "RightHand_End": "RightHand",
    "LeftToe_End": "LeftToeBase", "RightToe_End": "RightToeBase",
    "LeftRibbon_End": "LeftRibbon2", "RightRibbon_End": "RightRibbon2", "Sash_End": "Sash2",
}
# extra surface landmarks for the animator
LANDMARKS = ["LeftHeel", "RightHeel"]

ALL_POINT_NAMES = JOINT_NAMES + list(END_SITES) + LANDMARKS


def load_skeleton(path):
    """skeleton.json -> (joints list for glb.add_skin / glb.fk, end_sites dict, landmarks dict)"""
    import json
    d = json.load(open(path))
    pts = d["points"]
    joints = [{"name": n, "parent": p, "head": pts[n]} for n, p in HIERARCHY]
    ends = {n: pts[n] for n in END_SITES if n in pts}
    lms = {n: pts[n] for n in LANDMARKS if n in pts}
    return joints, ends, lms
