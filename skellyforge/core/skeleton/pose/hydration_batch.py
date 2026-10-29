"""Recover segment poses from named landmark arrays over bounded frame chunks."""

from dataclasses import dataclass, field

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import (
    MINIMUM_VECTOR_NORM,
    MINIMUM_SINE_BETWEEN_DEFINING_VECTORS,
)
from skellyforge.core.math.geometry.orthonormal_basis.calculate_orthonormal_basis import (
    calculate_orthonormal_basis,
)
from skellyforge.core.math.geometry.orthonormal_basis.reference_frame_definition import (
    ReferenceFrameDefinition,
)
from skellyforge.core.math.geometry.rotation_quaternion import (
    rotation_matrices_to_quaternions,
)
from skellyforge.core.math.geometry.spatial_vectors import Point, UnitVector
from skellyforge.core.math.kinematics.coordinate_frame_ops import (
    direction_pair_quaternions,
)
from skellyforge.core.math.kinematics.rigid_point_set import (
    align_point_sets_batch,
    MINIMUM_POINTS_FOR_RIGID_FIT,
)
from skellyforge.core.skeleton.pose.hydration import (
    MissingLandmarkObservations,
    DegenerateObservations,
)
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.pose.pose_tensor import (
    PoseTensor,
    SolutionCode,
    decode_solutions,
    POSITION_COMPONENTS,
    ROTATION_COMPONENTS,
)


def _immutable_array(values: object, *, dtype: object) -> np.ndarray:
    """Own prepared metadata in immutable storage, independent of authored arrays."""
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class ObservationGeometry:
    """Segments sharing signed axes; point columns retain each segment's names.

    Landmark roles are observation origin/primary/secondary, segment origin/primary.
    The shared definition supplies axis conventions and names for the basis adapter.
    """

    segment_columns: np.ndarray
    landmark_columns: np.ndarray
    segment_lengths: np.ndarray
    frame_definition: ReferenceFrameDefinition


@dataclass(frozen=True)
class DirectionGeometry:
    """All direction candidates, including underobserved rigid segments."""

    segment_columns: np.ndarray
    origin_columns: np.ndarray
    primary_columns: np.ndarray
    primary_signs: np.ndarray
    reference_directions: np.ndarray
    segment_lengths: np.ndarray


@dataclass(frozen=True)
class RigidGeometry:
    """One authored marker geometry; visibility groups remain sample-dependent."""

    segment_column: int
    landmark_columns: np.ndarray
    reference_positions: np.ndarray


@dataclass(frozen=True)
class HydratedPoseBatch:
    """Dense (F,S) axes, with origins (F,S,3) and unit wxyz rotations (F,S,4).

    Status: 0 missing observations, 1 valid, 2 degenerate observations. Absent
    numeric values are NaN and provenance is None. No roll continuity is inferred.
    """

    origins: np.ndarray
    orientations: np.ndarray
    scale_estimates: np.ndarray
    solutions: np.ndarray
    status: np.ndarray

    @property
    def segment_present(self) -> np.ndarray:
        return self.status == 1


@dataclass(frozen=True)
class LandmarkPoseGeometry:
    """Immutable solver metadata for fixed named axes and model definitions.

    Recovery shares prepared references, never mutable output or scratch arrays.
    Each returned tensor stays independent of subsequent calls, including when
    this geometry is shared by multiple callers. Model definitions and named axes
    must stay unchanged during use.
    """

    skeleton: SkeletonDefinition
    landmark_names: tuple[str, ...]
    segment_names: tuple[str, ...]
    landmark_indices: dict[str, int]
    observation_geometry: tuple[ObservationGeometry, ...] = field(
        init=False, repr=False, compare=False
    )
    direction_geometry: DirectionGeometry = field(init=False, repr=False, compare=False)
    rigid_geometry: tuple[RigidGeometry, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Compile names, axes and reference vectors once. Recovery only indexes arrays.
        observation_groups: dict[tuple, list] = {}
        direction_columns, origin_columns, primary_columns = [], [], []
        primary_signs, reference_directions, direction_lengths = [], [], []
        rigid_geometry = []
        for column, name in enumerate(self.segment_names):
            segment = self.skeleton.segments[name]
            origin_name = segment.frame_definition.origin_point_name
            primary_name = segment.frame_definition.primary_point_name
            definition = segment.observation_frame
            if definition is not None:
                required = (*definition.point_names, origin_name, primary_name)
                if all(name in self.landmark_indices for name in required):
                    key = (
                        definition.primary_axis,
                        definition.secondary_axis,
                        definition.handedness,
                    )
                    observation_groups.setdefault(key, []).append(
                        (column, required, segment.length, definition)
                    )
                # Explicit observation frames never fall back to another solver.
                continue
            if segment.supports_rigid_fit:
                owned = tuple(
                    name for name in segment.landmarks if name in self.landmark_indices
                )
                rigid_geometry.append(
                    RigidGeometry(
                        column,
                        _immutable_array(
                            [self.landmark_indices[name] for name in owned],
                            dtype=np.intp,
                        ),
                        _immutable_array(
                            [
                                segment.landmarks[name].local_position.array
                                for name in owned
                            ],
                            dtype=np.float64,
                        ).reshape((-1, len(POSITION_COMPONENTS))),
                    )
                )
            if (
                origin_name not in self.landmark_indices
                or primary_name not in self.landmark_indices
            ):
                continue
            local = segment.landmarks[primary_name].local_position.array
            local_length = np.linalg.norm(local)
            sign = segment.frame_definition.primary_axis.sign
            direction_columns.append(column)
            origin_columns.append(self.landmark_indices[origin_name])
            primary_columns.append(self.landmark_indices[primary_name])
            primary_signs.append(sign)
            # Defer invalid-reference errors until an observed direction needs it.
            reference_directions.append(
                sign * local / local_length
                if local_length >= MINIMUM_VECTOR_NORM
                else np.full_like(local, np.nan)
            )
            direction_lengths.append(segment.length)
        observation_geometry = tuple(
            ObservationGeometry(
                _immutable_array([item[0] for item in group], dtype=np.intp),
                _immutable_array(
                    [
                        [self.landmark_indices[name] for name in item[1]]
                        for item in group
                    ],
                    dtype=np.intp,
                ),
                _immutable_array([item[2] for item in group], dtype=np.float64),
                group[0][3],
            )
            for group in observation_groups.values()
        )
        direction_geometry = DirectionGeometry(
            _immutable_array(direction_columns, dtype=np.intp),
            _immutable_array(origin_columns, dtype=np.intp),
            _immutable_array(primary_columns, dtype=np.intp),
            _immutable_array(primary_signs, dtype=np.float64),
            _immutable_array(reference_directions, dtype=np.float64).reshape(
                (-1, len(POSITION_COMPONENTS))
            ),
            _immutable_array(direction_lengths, dtype=np.float64),
        )
        object.__setattr__(self, "observation_geometry", observation_geometry)
        object.__setattr__(self, "direction_geometry", direction_geometry)
        object.__setattr__(self, "rigid_geometry", tuple(rigid_geometry))

    @classmethod
    def from_skeleton(
        cls, *, skeleton: SkeletonDefinition, landmark_names: tuple[str, ...]
    ) -> "LandmarkPoseGeometry":
        if len(set(landmark_names)) != len(landmark_names):
            raise ValueError("Landmark names must be unique")
        return cls(
            skeleton,
            landmark_names,
            tuple(skeleton.segments),
            {n: i for i, n in enumerate(landmark_names)},
        )

    def recover(
        self,
        *,
        positions: np.ndarray,
        landmark_present: np.ndarray,
        require_all: bool = False,
    ) -> HydratedPoseBatch:
        """Adapt existing object-provenance callers to the shared tensor solver."""
        tensor = self.recover_tensor(
            positions=positions,
            landmark_present=landmark_present,
            require_all=require_all,
        )
        return HydratedPoseBatch(
            tensor.origins,
            tensor.orientations,
            tensor.scale_estimates,
            decode_solutions(tensor.solution_codes),
            tensor.hydration_status,
        )

    def recover_tensor(
        self,
        *,
        positions: np.ndarray,
        landmark_present: np.ndarray,
        require_all: bool = False,
    ) -> PoseTensor:
        """Batch samples AND compatible segments; group rigid fits by visible landmarks.

        Input (F,L,3) and boolean (F,L). Present positions must be finite. Missing
        observations are explicitly masked, not substituted. Observation-frame
        precedence and direction fallback for underobserved rigid segments match
        hydrate_segment. Both frames and landmarks participate in array operations.
        """
        points = np.asarray(positions, dtype=np.float64)
        present = np.asarray(landmark_present)
        if points.ndim != 3 or points.shape[1:] != (
            len(self.landmark_names),
            len(POSITION_COMPONENTS),
        ):
            raise ValueError("Positions must have shape (frames, landmarks, 3)")
        if present.dtype != np.bool_ or present.shape != points.shape[:2]:
            raise ValueError("Presence must be boolean (frames, landmarks)")
        if not np.isfinite(points[present]).all():
            raise ValueError("Present positions must be finite")
        frames, segments = len(points), len(self.segment_names)
        origins = np.full((frames, segments, len(POSITION_COMPONENTS)), np.nan)
        rotations = np.full((frames, segments, len(ROTATION_COMPONENTS)), np.nan)
        scales = np.full((frames, segments), np.nan)
        solutions = np.zeros((frames, segments), dtype=np.uint8)
        status = np.zeros((frames, segments), dtype=np.uint8)
        for geometry in self.observation_geometry:
            frame_rows, group_columns = np.nonzero(
                present[:, geometry.landmark_columns].all(axis=-1)
            )
            if not len(frame_rows):
                continue
            segment_columns = geometry.segment_columns[group_columns]
            status[frame_rows, segment_columns] = 2
            observed = points[
                frame_rows[:, None], geometry.landmark_columns[group_columns]
            ]
            (
                frame_origin,
                frame_primary,
                frame_secondary,
                segment_origin,
                segment_primary,
            ) = np.moveaxis(observed, 1, 0)
            primary = frame_primary - frame_origin
            secondary = frame_secondary - frame_origin
            length = np.linalg.norm(segment_primary - segment_origin, axis=-1)
            a, b = np.linalg.norm(primary, axis=-1), np.linalg.norm(secondary, axis=-1)
            valid = (
                (a >= MINIMUM_VECTOR_NORM)
                & (b >= MINIMUM_VECTOR_NORM)
                & (length >= MINIMUM_VECTOR_NORM)
            )
            # Same degeneracy screen and validated solver as scalar hydration. Flatten
            # sample/segment selections together so each axis convention needs one call.
            aa = primary / np.where(a > 0, a, 1)[:, None]
            bb = secondary / np.where(b > 0, b, 1)[:, None]
            residual = bb - np.sum(bb * aa, axis=-1)[:, None] * aa
            valid &= (
                np.linalg.norm(residual, axis=-1)
                >= MINIMUM_SINE_BETWEEN_DEFINING_VECTORS
            )
            if not valid.any():
                continue
            definition = geometry.frame_definition
            basis = calculate_orthonormal_basis(
                points={
                    definition.origin_point_name: Point.from_array(
                        values=frame_origin[valid]
                    ),
                    definition.primary_point_name: Point.from_array(
                        values=frame_primary[valid]
                    ),
                    definition.secondary_point_name: Point.from_array(
                        values=frame_secondary[valid]
                    ),
                },
                definition=definition,
            )
            chosen = frame_rows[valid]
            columns = segment_columns[valid]
            rotations[chosen, columns] = rotation_matrices_to_quaternions(
                matrices=basis.world_from_local_matrix
            )
            origins[chosen, columns] = segment_origin[valid]
            scales[chosen, columns] = (
                length[valid] / geometry.segment_lengths[group_columns[valid]]
            )
            solutions[chosen, columns] = SolutionCode.OBSERVATION_FRAME
            status[chosen, columns] = 1
        for geometry in self.rigid_geometry:
            column = geometry.segment_column
            visible = present[:, geometry.landmark_columns]
            rows = np.flatnonzero(visible.sum(axis=1) >= MINIMUM_POINTS_FOR_RIGID_FIT)
            if not len(rows):
                continue
            selections, group = np.unique(visible[rows], axis=0, return_inverse=True)
            # A degenerate rigid fit must never fall back to direction recovery.
            status[rows, column] = 2
            for group_number, selection in enumerate(selections):
                chosen = rows[group == group_number]
                reference = geometry.reference_positions[selection]
                observed = points[chosen[:, None], geometry.landmark_columns[selection]]
                matrices, translations, fitted_scales, valid = align_point_sets_batch(
                    reference=reference, observed=observed
                )
                valid_rows = chosen[valid]
                rotations[valid_rows, column] = rotation_matrices_to_quaternions(
                    matrices=matrices[valid]
                )
                origins[valid_rows, column] = translations[valid]
                scales[valid_rows, column] = fitted_scales[valid]
                solutions[valid_rows, column] = SolutionCode.RIGID_FIT
                status[valid_rows, column] = 1
        geometry = self.direction_geometry
        eligible = (
            present[:, geometry.origin_columns]
            & present[:, geometry.primary_columns]
            & (status[:, geometry.segment_columns] == 0)
        )
        frame_rows, direction_columns = np.nonzero(eligible)
        segment_columns = geometry.segment_columns[direction_columns]
        displacement = (
            points[frame_rows, geometry.primary_columns[direction_columns]]
            - points[frame_rows, geometry.origin_columns[direction_columns]]
        )
        length = np.linalg.norm(displacement, axis=-1)
        valid = length >= MINIMUM_VECTOR_NORM
        status[frame_rows, segment_columns] = 2
        chosen = frame_rows[valid]
        if len(chosen):
            columns = segment_columns[valid]
            selected = direction_columns[valid]
            reference = geometry.reference_directions[selected]
            if not np.isfinite(reference).all():
                raise ValueError("Local primary landmark has zero length")
            rotations[chosen, columns] = direction_pair_quaternions(
                from_directions=UnitVector.from_array(values=reference),
                to_directions=UnitVector.from_array(
                    values=geometry.primary_signs[selected, None]
                    * displacement[valid]
                    / length[valid, None]
                ),
            )
            origins[chosen, columns] = points[chosen, geometry.origin_columns[selected]]
            scales[chosen, columns] = length[valid] / geometry.segment_lengths[selected]
            solutions[chosen, columns] = SolutionCode.DIRECTION
            status[chosen, columns] = 1
        if require_all and np.any(status != 1):
            frame, column = np.argwhere(status != 1)[0]
            error = (
                MissingLandmarkObservations
                if status[frame, column] == 0
                else DegenerateObservations
            )
            raise error(
                f"Frame {frame}, segment {self.segment_names[column]}: cannot recover pose"
            )
        return PoseTensor(
            self.segment_names, origins, rotations, scales, solutions, status
        )
