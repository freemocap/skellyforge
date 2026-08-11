"""Standard human model — the canonical VRM-1.0-aligned humanoid.

Pydantic ``BaseModel`` holding the full skeleton definition: every bone
with its reference geometry, the joint hierarchy, T-pose marker positions,
and blendshape channel declarations. Loaded once at startup, validated,
then consumed by the orientation solver and streaming schema builder.

The model is a pure data description — it carries no runtime state, no
per-frame data, and no tracker knowledge (that's SkellyTracker's domain).
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from skellyforge.skellymodels.standard_human.human_bones import (
    BoneReferenceGeometry,
    CoordinateFrameDefinition,
    HumanBone,
    TwistPolicy,
    TwistTier,
)
from skellyforge.skellymodels.standard_human.human_blendshapes import (
    BlendShapeChannel,
    get_blendshape_names,
)


class StandardHuman(BaseModel):
    """The canonical humanoid skeleton.

    A validated, self-consistent definition of every bone in a
    VRM-1.0-aligned humanoid at T-pose. This is the single source of
    truth that the orientation solver compares live landmarks against
    and that the streaming schema enumerates.

    Bones subsume the old ``segment_connections`` concept — every bone
    carries its proximal/distal joint centers and coordinate frame
    as part of its ``BoneReferenceGeometry``.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # numpy arrays in HumanBone fields
        frozen=False,  # mutable for now — may freeze once stable
    )

    name: str
    """Identifier for this model (e.g. ``"standard_human_v1"``)."""

    bones: list[HumanBone]
    """All bones in hierarchy order (root first)."""

    blendshape_channels: list[str]
    """ARKit blendshape channel names (always 52)."""

    subject_height_mm: float = 1700.0
    """Nominal subject height in millimeters, used to scale bone lengths
    from anthropometric ratios when T-pose positions are auto-generated.
    Override for subject-specific models.
    """

    # ── Validation ─────────────────────────────────────────────────

    @model_validator(mode="after")
    def validate_bone_hierarchy_is_a_tree(self) -> "StandardHuman":
        """Ensure bones form a single-rooted tree with no orphans."""
        bone_names = {b.name for b in self.bones}

        if len(bone_names) != len(self.bones):
            raise ValueError(
                f"Duplicate bone names detected. "
                f"Unique names: {len(bone_names)}, bones: {len(self.bones)}"
            )

        # Every parent reference must point to an existing bone
        root_count = 0
        for bone in self.bones:
            if bone.parent is None:
                root_count += 1
            elif bone.parent not in bone_names:
                raise ValueError(
                    f"Bone '{bone.name}' references parent "
                    f"'{bone.parent}' which is not in the bone list"
                )

        if root_count != 1:
            raise ValueError(
                f"Skeleton must have exactly one root bone (parent=None), "
                f"got {root_count}"
            )

        # No cycles (follow parent chain from each bone to root)
        for bone in self.bones:
            visited: set[str] = set()
            current = bone.name
            while current is not None:
                if current in visited:
                    raise ValueError(
                        f"Cycle detected in bone hierarchy at '{current}'"
                    )
                visited.add(current)
                parent_bone = self._get_bone_by_name(current)
                if parent_bone is None:
                    break
                current = parent_bone.parent

        return self

    @model_validator(mode="after")
    def validate_twist_sources_exist(self) -> "StandardHuman":
        """Ensure CHAIN_RESOLVED twist sources reference real bones."""
        bone_names = {b.name for b in self.bones}

        for bone in self.bones:
            if bone.twist_policy.tier == TwistTier.CHAIN_RESOLVED:
                source = bone.twist_policy.twist_source_bone
                if source is None:
                    raise ValueError(
                        f"Bone '{bone.name}' has CHAIN_RESOLVED twist tier "
                        f"but twist_source_bone is None"
                    )
                if source not in bone_names:
                    raise ValueError(
                        f"Bone '{bone.name}' references twist source "
                        f"'{source}' which is not in the bone list"
                    )

        return self

    @model_validator(mode="after")
    def validate_required_bones_present(self) -> "StandardHuman":
        """Ensure all required VRM bones are present."""
        bone_names = {b.name for b in self.bones}
        required = {b.name for b in self.bones if b.required}

        missing_required = required - bone_names
        if missing_required:
            raise ValueError(
                f"Required bones missing from model: "
                f"{sorted(missing_required)}"
            )
        # This should never trigger (we just built bone_names from the
        # bones list), but kept as a structural check for subclasses or
        # partial models.

        return self

    # ── Accessors ──────────────────────────────────────────────────

    def _get_bone_by_name(self, name: str) -> HumanBone | None:
        """Look up a bone by canonical name."""
        for bone in self.bones:
            if bone.name == name:
                return bone
        return None

    @property
    def bone_names(self) -> list[str]:
        """Canonical bone names in declaration order."""
        return [b.name for b in self.bones]

    @property
    def joint_hierarchy(self) -> dict[str, list[str]]:
        """Parent → children mapping for the skeleton tree."""
        hierarchy: dict[str, list[str]] = {}
        for bone in self.bones:
            parent_key = bone.parent if bone.parent is not None else "__root__"
            hierarchy.setdefault(parent_key, []).append(bone.name)
        return hierarchy

    @property
    def root_bone(self) -> HumanBone:
        """The single root bone (parent is None)."""
        for bone in self.bones:
            if bone.parent is None:
                return bone
        raise ValueError("No root bone found — skeleton is invalid")

    @property
    def t_pose_markers(self) -> dict[str, NDArray[np.float64]]:
        """T-pose joint center positions for every bone.

        Keyed by bone name. Each value is the proximal joint center
        (i.e. the origin of the bone in the skeleton).
        """
        return {
            bone.name: bone.reference_geometry.proximal_joint_center.copy()
            for bone in self.bones
        }

    def get_children(self, bone_name: str) -> list[HumanBone]:
        """Return the child bones of the named bone."""
        return [
            b for b in self.bones
            if b.parent == bone_name
        ]

    def get_bone_chain(
        self, bone_name: str
    ) -> list[HumanBone]:
        """Return the chain from root to the named bone (inclusive)."""
        chain: list[HumanBone] = []
        current = self._get_bone_by_name(bone_name)
        while current is not None:
            chain.append(current)
            current = (
                self._get_bone_by_name(current.parent)
                if current.parent is not None
                else None
            )
        chain.reverse()
        return chain

    # ── Construction ───────────────────────────────────────────────

    @classmethod
    def from_bone_definitions(
        cls,
        name: str,
        bone_defs: list[dict[str, Any]],
        subject_height_mm: float = 1700.0,
    ) -> "StandardHuman":
        """Build from a list of bone definition dicts.

        Each dict must contain the fields needed to construct a
        ``HumanBone``: ``name``, ``parent``, ``required``,
        ``proximal_joint``, ``distal_joint``, ``exact_axis``,
        ``approximate_axis``, ``twist_tier``, and optionally
        ``twist_source_bone`` and ``damping_factor``.

        This is the primary construction path — bone definitions
        typically come from a YAML config or a programmatic builder.
        """
        bones: list[HumanBone] = []
        for bd in bone_defs:
            proximal = np.array(bd["proximal_joint"], dtype=np.float64)
            distal = np.array(bd["distal_joint"], dtype=np.float64)
            exact = np.array(bd["exact_axis"], dtype=np.float64)
            approx = np.array(bd["approximate_axis"], dtype=np.float64)

            ref_geom = BoneReferenceGeometry(
                proximal_joint_center=proximal,
                distal_joint_center=distal,
                coordinate_frame=CoordinateFrameDefinition(
                    exact_axis=exact,
                    approximate_axis=approx,
                ),
            )

            twist_tier = TwistTier(bd["twist_tier"])
            twist = TwistPolicy(
                tier=twist_tier,
                twist_source_bone=bd.get("twist_source_bone"),
                damping_factor=bd.get("damping_factor", 0.95),
            )

            bones.append(HumanBone(
                name=bd["name"],
                parent=bd.get("parent"),
                required=bd.get("required", True),
                reference_geometry=ref_geom,
                twist_policy=twist,
            ))

        return cls(
            name=name,
            bones=bones,
            blendshape_channels=get_blendshape_names(),
            subject_height_mm=subject_height_mm,
        )


# ── Anthropometric seed data ───────────────────────────────────────────
#
# Bone-length-to-height ratios from Winter (2009) and Drillis & Contini
# (1966), copied from the existing canonical_body.yaml. Used to seed
# T-pose joint centers when no explicit coordinates are provided.
#
# Ratios are bone_length / total_standing_height.

_BONE_LENGTH_RATIOS: dict[str, float] = {
    # Arms (Winter 2009)
    "shoulder_to_upper_arm": 0.186,   # clavicle length ≈ upper arm
    "upper_arm_to_lower_arm": 0.146,
    "lower_arm_to_hand": 0.108,       # hand length

    # Legs (Winter 2009)
    "hip_to_upper_leg": 0.245,        # thigh
    "upper_leg_to_lower_leg": 0.246,  # shank
    "lower_leg_to_foot": 0.039,       # foot height (ankle→toe)
    "hip_half_width": 0.057,

    # Torso (Winter 2009, approximate)
    "hips_to_spine": 0.145,
    "spine_to_chest": 0.100,
    "chest_to_upper_chest": 0.055,
    "upper_chest_to_neck": 0.090,     # neck base → head center
    "neck_to_head": 0.040,            # head center → top

    # Shoulder width
    "shoulder_half_width": 0.117,     # neck_center → shoulder (half biacromial)

    # Hand finger ratios (Buryanov & Kotiuk 2010, scaled to height)
    "hand_root": 0.108,               # hand length / height
    "thumb_metacarpal": 0.015,
    "thumb_proximal": 0.018,
    "thumb_distal": 0.017,
    "finger_proximal": 0.028,
    "finger_intermediate": 0.018,
    "finger_distal": 0.014,
    "finger_metacarpal": 0.050,       # wrist → MCP
}
"""Anthropometric bone-length-to-height ratios.

Keys use a ``parent_to_child`` naming convention for lookup during
T-pose construction. Values are dimensionless ratios.
"""
