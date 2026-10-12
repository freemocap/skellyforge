"""The whole body's inertial properties: total mass, center of mass, and inertia tensor.

Computed at the de Leva 16-segment level: each anatomical segment carries its de Leva mass
fraction, a center of mass resolved from the skeleton's landmarks (center_of_mass.py), and
an inertia tensor about that center (segment_inertia.py). The whole-body center of mass is
the mass-weighted mean of the segment centers, and the whole-body inertia tensor is the
sum of each segment's tensor about the body center, by the parallel-axis theorem.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from skellyforge.core.biomechanics.anthropometric_parameters import AnthropometricParameters
from skellyforge.core.biomechanics.center_of_mass import (
    CenterOfMassDefinitions,
    landmark_world_positions,
    segment_com,
)
from skellyforge.core.biomechanics.segment_inertia import segment_inertia_tensors
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import SkeletonPose
from skellyforge.type_overloads import FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class BodyInertialProperties:
    """The whole body's inertial properties at one instant.

    Attributes:
        mass: the total body mass.
        center_of_mass: the world position of the whole-body center of mass.
        inertia_tensor: the 3x3 inertia tensor about the center of mass, world frame.
    """

    mass: float
    center_of_mass: FloatArray
    inertia_tensor: FloatArray


def body_inertial_properties(
    *,
    skeleton: SkeletonDefinition,
    pose: SkeletonPose,
    body_mass: float,
    anthropometric: AnthropometricParameters,
    com_definitions: CenterOfMassDefinitions,
    segment_scales: Mapping[str, float],
) -> BodyInertialProperties:
    """The whole body's mass, center of mass and inertia tensor at this pose.

    Args:
        skeleton: the skeleton the pose hydrates.
        pose: the skeleton's pose at one instant.
        body_mass: the subject's total body mass.
        anthropometric: de Leva mass fractions and radii of gyration.
        com_definitions: the per-segment COM and long-axis landmark definitions.
        segment_scales: each segment's fitted scale, in world units per unit of the model's reference unit —
            `ModelScaleFit.segment_scales`. Everything below reads world positions, and the
            template has no size without this.

    Raises:
        ValueError: body_mass is not positive.
    """
    if body_mass <= 0.0:
        raise ValueError(f"body_mass must be positive, got {body_mass}")

    world = landmark_world_positions(
        skeleton=skeleton, pose=pose, segment_scales=segment_scales
    )

    segment_masses: dict[str, float] = {}
    segment_coms: dict[str, FloatArray] = {}
    segment_inertias: dict[str, FloatArray] = {}
    tensor_inputs = []
    for definition in com_definitions.definitions.values():
        parameters = anthropometric.get(name=definition.name)
        mass = parameters.mass_fraction * body_mass
        for full_name, side in definition.side_entries:
            proximal_name = definition.resolve_landmark(
                side=side, landmark=definition.proximal
            )
            distal_name = definition.resolve_landmark(
                side=side, landmark=definition.distal
            )
            if proximal_name not in world or distal_name not in world:
                # An occluded anchor leaves the long axis unmeasurable, so this
                # segment contributes neither a COM nor an inertia - it is skipped
                # exactly like a segment whose whole pose is missing.
                continue
            com = segment_com(definition=definition, side=side, world=world)
            if com is None:
                # Every COM-defining landmark is occluded; the mass is invisible.
                continue
            proximal, distal = world[proximal_name], world[distal_name]
            radii = parameters.radii_of_gyration
            tensor_inputs.append((full_name, mass, [radii.transverse, radii.sagittal, radii.longitudinal], proximal, distal))
            segment_masses[full_name] = mass
            segment_coms[full_name] = com
    if tensor_inputs:
        tensors = segment_inertia_tensors(masses=np.asarray([entry[1] for entry in tensor_inputs]),
            radii=np.asarray([entry[2] for entry in tensor_inputs]), proximal=np.asarray([entry[3] for entry in tensor_inputs]),
            distal=np.asarray([entry[4] for entry in tensor_inputs]))
        segment_inertias = {entry[0]: tensor for entry, tensor in zip(tensor_inputs, tensors, strict=True)}

    center_of_mass = whole_body_center_of_mass(
        segment_coms=segment_coms, segment_masses=segment_masses
    )
    inertia_tensor = whole_body_inertia_tensor(
        segment_inertias=segment_inertias,
        segment_coms=segment_coms,
        segment_masses=segment_masses,
        body_center_of_mass=center_of_mass,
    )
    return BodyInertialProperties(
        mass=body_mass, center_of_mass=center_of_mass, inertia_tensor=inertia_tensor
    )


def whole_body_center_of_mass(
    *,
    segment_coms: Mapping[str, FloatArray | None],
    segment_masses: Mapping[str, float],
) -> FloatArray | None:
    """The mass-weighted mean of the segment centers of mass.

    Segments absent from ``segment_coms`` (or whose COM is ``None`` — fully
    occluded) are skipped and the remaining masses re-normalized, so the
    whole-body CoM rolls up over the visible segments. Returns ``None`` when no
    segment COM is available.
    """
    entries = [(com, segment_masses[name]) for name, com in segment_coms.items()
               if com is not None and segment_masses.get(name) is not None]
    if not entries:
        return None
    masses = np.asarray([mass for _, mass in entries])
    total_mass = np.add.accumulate(masses)[-1]
    if total_mass <= 0.0:
        return None
    weighted = masses[:, None] * np.asarray([com for com, _ in entries])
    return np.add.accumulate(weighted, axis=0)[-1] / total_mass


def whole_body_inertia_tensor(
    *,
    segment_inertias: dict[str, FloatArray],
    segment_coms: dict[str, FloatArray],
    segment_masses: dict[str, float],
    body_center_of_mass: FloatArray,
) -> FloatArray:
    """Sum the segment inertia tensors, translated to the body center of mass.

    Every entry of `segment_inertias` must have matching entries in `segment_coms`
    and `segment_masses` - an inertia without the COM it is being translated about
    is a caller bug and raises rather than guessing.
    """
    if not segment_inertias:
        return np.zeros((3, 3), dtype=np.float64)
    for name, inertia in segment_inertias.items():
        segment_com = segment_coms.get(name)
        if name not in segment_coms or name not in segment_masses:
            raise KeyError(
                f"segment {name!r} has an inertia tensor but no matching COM/mass "
                "entry - build the three maps together (see "
                "`body_inertial_properties`)"
            )
        if segment_com is None:
            raise ValueError(
                f"segment {name!r} has a None COM - occluded segments must be left "
                "out of all three maps entirely, not carried as None"
            )
    names = list(segment_inertias)
    offsets = np.asarray([segment_coms[name] for name in names]) - np.asarray(body_center_of_mass, dtype=np.float64)
    squared = (offsets[:, None, :] @ offsets[:, :, None])[:, 0, 0]
    shifts = squared[:, None, None] * np.eye(3) - offsets[:, :, None] * offsets[:, None, :]
    shifts *= np.asarray([segment_masses[name] for name in names])[:, None, None]
    # Preserve original sequential addition: total + inertia + displacement.
    terms = np.empty((2 * len(names), 3, 3), dtype=np.float64)
    terms[::2] = np.asarray([segment_inertias[name] for name in names])
    terms[1::2] = shifts
    return np.add.accumulate(terms, axis=0)[-1]


def _parallel_axis_term(*, offset: FloatArray) -> FloatArray:
    """The parallel-axis displacement term: ||d||^2 I - d d^T."""
    d = np.asarray(offset, dtype=np.float64)
    return float(np.dot(d, d)) * np.eye(3) - np.outer(d, d)
