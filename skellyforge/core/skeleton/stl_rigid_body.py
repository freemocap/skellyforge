"""STL feature geometry adapted to Forge's existing rigid reconstruction.

The mesh/object frame is not the segment frame. Explicit affine transforms retain
its origin, axes and source units while the segment uses measured feature names
for its reference triad. All features remain rigid-fit observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import (
    MINIMUM_VECTOR_NORM,
    ORTHONORMALITY_TOLERANCE,
)
from skellyforge.core.math.geometry.orthonormal_basis.calculate_orthonormal_basis import (
    calculate_orthonormal_basis,
)
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.math.geometry.transform_math import Transform
from skellyforge.core.skeleton.loading.reference_frame_building import (
    build_reference_frame_definition,
)
from skellyforge.core.skeleton.rigid_marker_skeleton import build_rigid_marker_skeleton
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


@dataclass(frozen=True, slots=True, eq=False)
class StlRigidBodyDefinition:
    """Prepared geometry; matrices multiply homogeneous column coordinates.

    Segment coordinates are dimensionless reference units. The fitted scale is
    measured millimetres per reference unit, comparable to reference_unit_mm.
    source_mesh refers to the original STL, not a normalized rendering copy.
    Arrays are owned/read-only. No inferred human/board identities are introduced.
    """

    skeleton: SkeletonDefinition
    feature_names: tuple[str, ...]
    reference_unit_mm: float
    object_to_segment_reference: np.ndarray
    source_mesh_to_segment_reference: np.ndarray
    segment_reference_to_source_mesh: np.ndarray
    source_mesh_units: str
    mesh_sha256: str | None
    segment_name: str
    scale_reference_name: str = "object_reference_length"

    def source_mesh_to_world_matrix(
        self, *, segment_to_world: Transform, scale_mm: float
    ) -> np.ndarray:
        """Compose the measured segment pose/scale with the original mesh frame.

        Use the pipeline's fitted scale, not an assumed nominal STL size. The
        result maps original source-unit mesh vertices to measured world mm.
        """
        if not np.isfinite(scale_mm) or scale_mm <= 0:
            raise ValueError("scale_mm must be finite and positive")
        matrix = np.eye(4)
        matrix[:3, :3] = segment_to_world.rotation.to_rotation_matrix() * scale_mm
        matrix[:3, 3] = segment_to_world.translation.array
        return matrix @ self.source_mesh_to_segment_reference


def _owned_readonly(*, values: object) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _validate_similarity(*, matrix: np.ndarray, scale: float) -> None:
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("source mesh transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=ORTHONORMALITY_TOLERANCE):
        raise ValueError("source mesh transform must have homogeneous last row")
    rotation = matrix[:3, :3] / scale
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0, atol=ORTHONORMALITY_TOLERANCE
    ) or not np.isclose(
        np.linalg.det(rotation), 1, rtol=0, atol=ORTHONORMALITY_TOLERANCE
    ):
        raise ValueError(
            "source mesh transform must preserve orientation and declared units"
        )


def build_stl_rigid_body(
    *,
    name: str,
    feature_names: tuple[str, ...],
    object_positions_mm: np.ndarray,
    reference_unit_mm: float,
    source_mesh_to_object_mm: np.ndarray,
    source_mesh_units: str,
    segment_name: str = "rigid_object",
    mesh_sha256: str | None = None,
) -> StlRigidBodyDefinition:
    """Prepare a rigid model without asking users to author a frame triad.

    Farthest pair plus the feature farthest from its line define a deterministic
    frame at load time. Existing Gram-Schmidt defines its axes. This does not
    restrict fitting to those three features, nor make a symmetry observable.
    """
    names = tuple(feature_names)
    positions = np.asarray(object_positions_mm, dtype=np.float64)
    if (
        len(names) < 3
        or any(not isinstance(item, str) or not item.strip() for item in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            "feature_names must contain at least three unique nonempty strings"
        )
    if positions.shape != (len(names), 3) or not np.isfinite(positions).all():
        raise ValueError(
            "object_positions_mm must be finite (feature_count, 3) coordinates"
        )
    if len(np.unique(positions, axis=0)) != len(names):
        raise ValueError("feature coordinates must be distinct")
    if not np.isfinite(reference_unit_mm) or reference_unit_mm <= MINIMUM_VECTOR_NORM:
        raise ValueError("reference_unit_mm must be finite and positive")
    unit_scales = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}
    if source_mesh_units not in unit_scales:
        raise ValueError("source_mesh_units must explicitly name mm, cm, m or in")
    source_to_object = np.asarray(source_mesh_to_object_mm, dtype=np.float64)
    _validate_similarity(matrix=source_to_object, scale=unit_scales[source_mesh_units])

    reference = positions / reference_unit_mm
    distances = np.sum((reference[:, None] - reference[None, :]) ** 2, axis=-1)
    origin, primary = np.unravel_index(np.argmax(distances), distances.shape)
    direction = reference[primary] - reference[origin]
    areas = np.linalg.norm(np.cross(reference - reference[origin], direction), axis=1)
    secondary = int(np.argmax(areas))
    frame_names = names[origin], names[primary], names[secondary]
    definition = build_reference_frame_definition(
        segment_name=segment_name,
        reference_geometry={
            "origin": frame_names[0],
            "x_axis": {"landmark": frame_names[1], "type": "exact"},
            "y_axis": {"landmark": frame_names[2], "type": "approximate"},
        },
    )
    basis = calculate_orthonormal_basis(
        points={
            key: Point.from_array(values=value) for key, value in zip(names, reference)
        },
        definition=definition,
    )
    object_to_segment = np.eye(4)
    object_to_segment[:3, :3] = basis.local_from_world_matrix / reference_unit_mm
    object_to_segment[:3, 3] = -basis.local_from_world_matrix @ reference[origin]
    local = (positions - positions[origin]) @ object_to_segment[:3, :3].T
    skeleton = build_rigid_marker_skeleton(
        name=name,
        segment_name=segment_name,
        marker_positions=dict(zip(names, local)),
        origin_marker_name=frame_names[0],
        primary_marker_name=frame_names[1],
        secondary_marker_name=frame_names[2],
        marker_definition="named geometric feature on the source STL rigid body",
    )
    source_to_segment = object_to_segment @ source_to_object
    return StlRigidBodyDefinition(
        skeleton=skeleton,
        feature_names=names,
        reference_unit_mm=float(reference_unit_mm),
        object_to_segment_reference=_owned_readonly(values=object_to_segment),
        source_mesh_to_segment_reference=_owned_readonly(values=source_to_segment),
        segment_reference_to_source_mesh=_owned_readonly(
            values=np.linalg.inv(source_to_segment)
        ),
        source_mesh_units=source_mesh_units,
        mesh_sha256=mesh_sha256,
        segment_name=segment_name,
    )


def load_stl_rigid_body(
    *, path: Path, name: str, segment_name: str = "rigid_object"
) -> StlRigidBodyDefinition:
    """Load generated definitions or the retained authored feature definition.

    Generated schema v1 carries the original STL unit transform. Legacy authored
    definitions explicitly carry millimetres, source origin and source rotation.
    This adapter does not import rendering/training tools or inference libraries.
    """
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("units") != "mm":
        raise ValueError("STL definition must declare object positions in mm")
    if "schema_version" in data and data["schema_version"] != 1:
        raise ValueError("Unsupported STL object definition schema_version")
    required = ("feature_names", "object_positions_mm", "reference_unit_mm")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"STL definition missing fields: {missing}")
    if not isinstance(data["feature_names"], list):
        raise ValueError("feature_names must be an ordered array of identities")
    if "source_mesh_to_object" in data:
        if "original_mesh_units" not in data:
            raise ValueError("original_mesh_units required with source_mesh_to_object")
        transform = np.asarray(data["source_mesh_to_object"], dtype=float)
        units = data["original_mesh_units"]
        mesh_hash = data.get("original_mesh_sha256")
        if "object_to_source_mesh" in data:
            inverse = np.asarray(data["object_to_source_mesh"], dtype=float)
            if (
                transform.shape != (4, 4)
                or inverse.shape != (4, 4)
                or not np.allclose(
                    transform @ inverse,
                    np.eye(4),
                    rtol=0,
                    atol=ORTHONORMALITY_TOLERANCE,
                )
            ):
                raise ValueError("source/object transforms are not mutually inverse")
    else:
        if "source_to_object_rotation" not in data or "origin_source_mm" not in data:
            raise ValueError(
                "Explicit source rotation and origin required for authored STL geometry"
            )
        rotation = np.asarray(data["source_to_object_rotation"], dtype=float)
        origin = np.asarray(data["origin_source_mm"], dtype=float)
        if rotation.shape != (3, 3) or origin.shape != (3,):
            raise ValueError("Invalid source rotation/origin shape")
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = -rotation @ origin
        units = "mm"
        mesh_hash = data.get("source_mesh_sha256")
    return build_stl_rigid_body(
        name=name,
        feature_names=tuple(data["feature_names"]),
        object_positions_mm=np.asarray(data["object_positions_mm"], dtype=np.float64),
        reference_unit_mm=float(data["reference_unit_mm"]),
        source_mesh_to_object_mm=transform,
        source_mesh_units=units,
        segment_name=segment_name,
        mesh_sha256=mesh_hash,
    )
