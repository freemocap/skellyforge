"""Biomechanics derived from a skeleton and its pose: mass, center of mass, inertia.

Layering: core/math/ is pure algebra; core/skeleton/ is the typed model; this package is
the DERIVED layer - it reads a SkeletonDefinition and a SkeletonPose and produces the
quantities that follow from them (segment masses, the whole-body center of mass and
inertia tensor, and ground-reference quantities). It never mutates the skeleton, and the
skeleton never imports it.

The anthropometric source of truth is de Leva (1996); the mass and radii-of-gyration
table lives in definitions/human_skeleton/anthropometric_parameters.yaml, and each
segment's 3D center-of-mass definition (a weighted sum of skeleton landmarks, which is
what places the trunk's COM anterior to the spine) lives in
definitions/human_skeleton/center_of_mass.yaml.
"""

from __future__ import annotations

from typing import Final

from skellyforge.core.math.geometry.numeric_tolerances import MINIMUM_VECTOR_NORM

# Mass fractions and COM weights are authored to four decimals and sum to 1.0 exactly;
# this tolerance is a thousand times the math base, tight enough to catch a transcription
# error (>= 1e-4) while tolerating float64 rounding.
FRACTION_SUM_TOLERANCE: Final[float] = 1000.0 * MINIMUM_VECTOR_NORM

from skellyforge.core.biomechanics.anthropometric_parameters import (
    AnthropometricParameters,
    AnthropometricSegment,
    RadiiOfGyration,
)
from skellyforge.core.biomechanics.center_of_mass import (
    CenterOfMassDefinitions,
    ComLandmarkWeight,
    SegmentComDefinition,
    compute_segment_coms,
    landmark_world_positions,
    segment_com,
)
from skellyforge.core.biomechanics.composite_inertia import (
    BodyInertialProperties,
    body_inertial_properties,
    whole_body_center_of_mass,
    whole_body_inertia_tensor,
)
from skellyforge.core.biomechanics.derived_kinematics import (
    center_of_mass_acceleration,
    center_of_mass_velocity,
)
from skellyforge.core.biomechanics.ground_reference import (
    GRAVITY_ACCELERATION,
    center_of_pressure,
    centroidal_moment_pivot,
    extrapolated_center_of_mass,
)
from skellyforge.core.biomechanics.segment_inertia import segment_inertia_tensor
from skellyforge.core.biomechanics.segment_mapping import (
    anatomical_segment_name,
    distribute_segment_masses,
    map_skeleton_segments,
)

__all__ = [
    "AnthropometricParameters",
    "AnthropometricSegment",
    "BodyInertialProperties",
    "CenterOfMassDefinitions",
    "ComLandmarkWeight",
    "FRACTION_SUM_TOLERANCE",
    "GRAVITY_ACCELERATION",
    "RadiiOfGyration",
    "SegmentComDefinition",
    "anatomical_segment_name",
    "body_inertial_properties",
    "center_of_mass_acceleration",
    "center_of_mass_velocity",
    "center_of_pressure",
    "centroidal_moment_pivot",
    "compute_segment_coms",
    "distribute_segment_masses",
    "extrapolated_center_of_mass",
    "landmark_world_positions",
    "map_skeleton_segments",
    "segment_com",
    "segment_inertia_tensor",
    "whole_body_center_of_mass",
    "whole_body_inertia_tensor",
]
