"""Bone name aliases for the standard human model.

Maps standard snake_case bone names to wire-format names for each target
protocol. The standard name is the single source of truth; aliases are a
serialization concern — bones themselves never carry alias knowledge.

Adding a new target means adding a column to ``BONE_ALIASES``. No bone
definitions change. Missing aliases silently fall back to the standard
name — the stream keeps working while the alias row is added.

Target key conventions:
    ``vrm`` — VRM 1.0 camelCase (also used by VMC Protocol over OSC)
    ``unreal`` — UE5 Mannequin skeleton bone names
"""

# ── Alias table ────────────────────────────────────────────────────────
#
# standard_name → {target: alias}
#
# Standard names follow VRM 1.0 structure with snake_case normalization.
# The VRM target is VRM 1.0 camelCase; the VMC adapter maps from this to
# VRM 0.x names (e.g. thumb metacarpal → proximal) at emit time.
# The Unreal target maps to UE5 Mannequin bone names.
#
# Bones without a standard Unreal equivalent (eye bones, jaw) map to
# ``None`` — the Unreal adapter handles these via morph targets or omits
# them.

BONE_ALIASES: dict[str, dict[str, str | None]] = {
    # ── Torso ──────────────────────────────────────────────────────
    "hips": {
        "vrm": "hips",
        "unreal": "pelvis",
    },
    "spine": {
        "vrm": "spine",
        "unreal": "spine_01",
    },
    "chest": {
        "vrm": "chest",
        "unreal": "spine_03",
    },
    "upper_chest": {
        "vrm": "upperChest",
        "unreal": "spine_05",
    },
    "neck": {
        "vrm": "neck",
        "unreal": "neck_01",
    },

    # ── Head ───────────────────────────────────────────────────────
    "head": {
        "vrm": "head",
        "unreal": "head",
    },
    "left_eye": {
        "vrm": "leftEye",
        "unreal": None,  # no standard UE mannequin eye bone
    },
    "right_eye": {
        "vrm": "rightEye",
        "unreal": None,
    },
    "jaw": {
        "vrm": "jaw",
        "unreal": None,  # no standard UE mannequin jaw bone
    },
    # FreeMoCap face-detail segments — no VRM 1.0 humanoid or UE mannequin
    # equivalent; both targets resolve to None so adapters omit them.
    "nose": {
        "vrm": None,
        "unreal": None,
    },
    "left_ear": {
        "vrm": None,
        "unreal": None,
    },
    "right_ear": {
        "vrm": None,
        "unreal": None,
    },
    "left_mouth": {
        "vrm": None,
        "unreal": None,
    },
    "right_mouth": {
        "vrm": None,
        "unreal": None,
    },

    # ── Left arm ───────────────────────────────────────────────────
    "left_shoulder": {
        "vrm": "leftShoulder",
        "unreal": "clavicle_l",
    },
    "left_upper_arm": {
        "vrm": "leftUpperArm",
        "unreal": "upperarm_l",
    },
    "left_lower_arm": {
        "vrm": "leftLowerArm",
        "unreal": "lowerarm_l",
    },
    "left_hand": {
        "vrm": "leftHand",
        "unreal": "hand_l",
    },

    # ── Right arm ──────────────────────────────────────────────────
    "right_shoulder": {
        "vrm": "rightShoulder",
        "unreal": "clavicle_r",
    },
    "right_upper_arm": {
        "vrm": "rightUpperArm",
        "unreal": "upperarm_r",
    },
    "right_lower_arm": {
        "vrm": "rightLowerArm",
        "unreal": "lowerarm_r",
    },
    "right_hand": {
        "vrm": "rightHand",
        "unreal": "hand_r",
    },

    # ── Left leg ───────────────────────────────────────────────────
    "left_upper_leg": {
        "vrm": "leftUpperLeg",
        "unreal": "thigh_l",
    },
    "left_lower_leg": {
        "vrm": "leftLowerLeg",
        "unreal": "calf_l",
    },
    "left_foot": {
        "vrm": "leftFoot",
        "unreal": "foot_l",
    },
    "left_toes": {
        "vrm": "leftToes",
        "unreal": "ball_l",
    },

    # ── Right leg ──────────────────────────────────────────────────
    "right_upper_leg": {
        "vrm": "rightUpperLeg",
        "unreal": "thigh_r",
    },
    "right_lower_leg": {
        "vrm": "rightLowerLeg",
        "unreal": "calf_r",
    },
    "right_foot": {
        "vrm": "rightFoot",
        "unreal": "foot_r",
    },
    "right_toes": {
        "vrm": "rightToes",
        "unreal": "ball_r",
    },

    # ── Left hand fingers ──────────────────────────────────────────
    # Thumb (VRM 1.0: metacarpal/proximal/distal)
    "left_thumb_metacarpal": {
        "vrm": "leftThumbMetacarpal",
        "unreal": "thumb_01_l",
    },
    "left_thumb_proximal": {
        "vrm": "leftThumbProximal",
        "unreal": "thumb_02_l",
    },
    "left_thumb_distal": {
        "vrm": "leftThumbDistal",
        "unreal": "thumb_03_l",
    },
    # Index
    "left_index_proximal": {
        "vrm": "leftIndexProximal",
        "unreal": "index_01_l",
    },
    "left_index_intermediate": {
        "vrm": "leftIndexIntermediate",
        "unreal": "index_02_l",
    },
    "left_index_distal": {
        "vrm": "leftIndexDistal",
        "unreal": "index_03_l",
    },
    # Middle
    "left_middle_proximal": {
        "vrm": "leftMiddleProximal",
        "unreal": "middle_01_l",
    },
    "left_middle_intermediate": {
        "vrm": "leftMiddleIntermediate",
        "unreal": "middle_02_l",
    },
    "left_middle_distal": {
        "vrm": "leftMiddleDistal",
        "unreal": "middle_03_l",
    },
    # Ring
    "left_ring_proximal": {
        "vrm": "leftRingProximal",
        "unreal": "ring_01_l",
    },
    "left_ring_intermediate": {
        "vrm": "leftRingIntermediate",
        "unreal": "ring_02_l",
    },
    "left_ring_distal": {
        "vrm": "leftRingDistal",
        "unreal": "ring_03_l",
    },
    # Little
    "left_little_proximal": {
        "vrm": "leftLittleProximal",
        "unreal": "pinky_01_l",
    },
    "left_little_intermediate": {
        "vrm": "leftLittleIntermediate",
        "unreal": "pinky_02_l",
    },
    "left_little_distal": {
        "vrm": "leftLittleDistal",
        "unreal": "pinky_03_l",
    },

    # ── Right hand fingers ─────────────────────────────────────────
    # Thumb
    "right_thumb_metacarpal": {
        "vrm": "rightThumbMetacarpal",
        "unreal": "thumb_01_r",
    },
    "right_thumb_proximal": {
        "vrm": "rightThumbProximal",
        "unreal": "thumb_02_r",
    },
    "right_thumb_distal": {
        "vrm": "rightThumbDistal",
        "unreal": "thumb_03_r",
    },
    # Index
    "right_index_proximal": {
        "vrm": "rightIndexProximal",
        "unreal": "index_01_r",
    },
    "right_index_intermediate": {
        "vrm": "rightIndexIntermediate",
        "unreal": "index_02_r",
    },
    "right_index_distal": {
        "vrm": "rightIndexDistal",
        "unreal": "index_03_r",
    },
    # Middle
    "right_middle_proximal": {
        "vrm": "rightMiddleProximal",
        "unreal": "middle_01_r",
    },
    "right_middle_intermediate": {
        "vrm": "rightMiddleIntermediate",
        "unreal": "middle_02_r",
    },
    "right_middle_distal": {
        "vrm": "rightMiddleDistal",
        "unreal": "middle_03_r",
    },
    # Ring
    "right_ring_proximal": {
        "vrm": "rightRingProximal",
        "unreal": "ring_01_r",
    },
    "right_ring_intermediate": {
        "vrm": "rightRingIntermediate",
        "unreal": "ring_02_r",
    },
    "right_ring_distal": {
        "vrm": "rightRingDistal",
        "unreal": "ring_03_r",
    },
    # Little
    "right_little_proximal": {
        "vrm": "rightLittleProximal",
        "unreal": "pinky_01_r",
    },
    "right_little_intermediate": {
        "vrm": "rightLittleIntermediate",
        "unreal": "pinky_02_r",
    },
    "right_little_distal": {
        "vrm": "rightLittleDistal",
        "unreal": "pinky_03_r",
    },
}


# ── Resolution ─────────────────────────────────────────────────────────


def resolve_alias(bone_name: str, target: str) -> str | None:
    """Resolve a standard bone name to its alias for a given target.

    Args:
        bone_name: Standard snake_case bone name (e.g. ``left_upper_arm``).
        target: Target protocol key (e.g. ``"vrm"``, ``"unreal"``).

    Returns:
        The alias string for that target, or ``None`` if the bone has no
        equivalent in the target skeleton (e.g. ``left_eye`` → Unreal).

        If the bone name itself is not in the alias table, logs a warning
        and returns the standard name unchanged — this lets a newly-added
        bone stream without crashing while someone adds the alias row.
    """
    target_aliases = BONE_ALIASES.get(bone_name)

    if target_aliases is None:
        return bone_name

    alias = target_aliases.get(target)

    if alias is None and target in target_aliases:
        return None

    if alias is None:
        return bone_name

    return alias


def resolve_all_aliases(
    bone_names: list[str], target: str
) -> dict[str, str | None]:
    """Resolve a list of standard bone names to their aliases for a target.

    Args:
        bone_names: Standard snake_case bone names.
        target: Target protocol key.

    Returns:
        Dict mapping bone_name → alias (or ``None`` for bones with no
        equivalent in that target).
    """
    return {name: resolve_alias(name, target) for name in bone_names}
