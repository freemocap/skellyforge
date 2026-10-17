"""Recover segment poses from named landmark arrays over bounded frame chunks."""

from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import MINIMUM_VECTOR_NORM, MINIMUM_SINE_BETWEEN_DEFINING_VECTORS
from skellyforge.core.math.geometry.orthonormal_basis.calculate_orthonormal_basis import calculate_orthonormal_basis
from skellyforge.core.math.geometry.rotation_quaternion import rotation_matrices_to_quaternions
from skellyforge.core.math.geometry.spatial_vectors import Point, UnitVector
from skellyforge.core.math.kinematics.coordinate_frame_ops import direction_pair_quaternions
from skellyforge.core.math.kinematics.rigid_point_set import align_point_sets_batch
from skellyforge.core.skeleton.pose.hydration import MissingLandmarkObservations, DegenerateObservations
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import PoseSolution


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
    """Fixed named axes; model definitions must stay unchanged during use."""
    skeleton: SkeletonDefinition
    landmark_names: tuple[str, ...]
    segment_names: tuple[str, ...]
    landmark_indices: dict[str, int]

    @classmethod
    def from_skeleton(cls, *, skeleton: SkeletonDefinition, landmark_names: tuple[str, ...]) -> "LandmarkPoseGeometry":
        if len(set(landmark_names)) != len(landmark_names):
            raise ValueError("Landmark names must be unique")
        return cls(skeleton, landmark_names, tuple(skeleton.segments), {n: i for i, n in enumerate(landmark_names)})

    def recover(self, *, positions: np.ndarray, landmark_present: np.ndarray,
                require_all: bool = False) -> HydratedPoseBatch:
        """Batch frames for each solver; group rigid fits by visible landmarks.

        Input (F,L,3) and boolean (F,L). Present positions must be finite. Missing
        observations are explicitly masked, not substituted. Observation-frame
        precedence and direction fallback for underobserved rigid segments match
        hydrate_segment. Both frames and landmarks participate in array operations.
        """
        points = np.asarray(positions, dtype=np.float64)
        present = np.asarray(landmark_present)
        if points.ndim != 3 or points.shape[1:] != (len(self.landmark_names), 3):
            raise ValueError("Positions must have shape (frames, landmarks, 3)")
        if present.dtype != np.bool_ or present.shape != points.shape[:2]:
            raise ValueError("Presence must be boolean (frames, landmarks)")
        if not np.isfinite(points[present]).all():
            raise ValueError("Present positions must be finite")
        frames, segments = len(points), len(self.segment_names)
        origins = np.full((frames, segments, 3), np.nan)
        rotations = np.full((frames, segments, 4), np.nan)
        scales = np.full((frames, segments), np.nan)
        solutions = np.full((frames, segments), None, dtype=object)
        status = np.zeros((frames, segments), dtype=np.uint8)
        for column, name in enumerate(self.segment_names):
            segment = self.skeleton.segments[name]
            origin_name = segment.frame_definition.origin_point_name
            primary_name = segment.frame_definition.primary_point_name
            definition = segment.observation_frame
            if definition is not None:
                required = tuple(dict.fromkeys((*definition.point_names, origin_name, primary_name)))
                if not set(required).issubset(self.landmark_indices):
                    continue
                columns = [self.landmark_indices[n] for n in required]
                rows = np.flatnonzero(present[:, columns].all(axis=1))
                if not len(rows):
                    continue
                status[rows, column] = 2
                p = {n: points[rows, self.landmark_indices[n]] for n in required}
                primary = p[definition.primary_point_name] - p[definition.origin_point_name]
                secondary = p[definition.secondary_point_name] - p[definition.origin_point_name]
                length = np.linalg.norm(p[primary_name] - p[origin_name], axis=1)
                a, b = np.linalg.norm(primary, axis=1), np.linalg.norm(secondary, axis=1)
                valid = (a >= MINIMUM_VECTOR_NORM) & (b >= MINIMUM_VECTOR_NORM) & (length >= MINIMUM_VECTOR_NORM)
                # Screen degeneracies before invoking the shared validated basis solver.
                aa = primary / np.where(a > 0, a, 1)[:, None]
                bb = secondary / np.where(b > 0, b, 1)[:, None]
                residual = bb - np.sum(bb * aa, axis=1)[:, None] * aa
                valid &= np.linalg.norm(residual, axis=1) >= MINIMUM_SINE_BETWEEN_DEFINING_VECTORS
                if not valid.any():
                    continue
                basis = calculate_orthonormal_basis(points={n: Point.from_array(values=v[valid]) for n, v in p.items()}, definition=definition)
                chosen = rows[valid]
                rotations[chosen, column] = rotation_matrices_to_quaternions(matrices=basis.world_from_local_matrix)
                origins[chosen, column] = p[origin_name][valid]
                scales[chosen, column] = length[valid] / segment.length
                solutions[chosen, column] = PoseSolution.OBSERVATION_FRAME
                status[chosen, column] = 1
                continue
            direction_rows = np.ones(frames, dtype=bool)
            if segment.supports_rigid_fit:
                owned = [n for n in segment.landmarks if n in self.landmark_indices]
                columns = [self.landmark_indices[n] for n in owned]
                visible = present[:, columns]
                rigid_rows = visible.sum(axis=1) >= 3
                direction_rows &= ~rigid_rows
                if rigid_rows.any():
                    rows = np.flatnonzero(rigid_rows)
                    selections, group = np.unique(visible[rows], axis=0, return_inverse=True)
                    status[rows, column] = 2
                    for group_number, selection in enumerate(selections):
                        chosen = rows[group == group_number]
                        selected_names = [n for n, keep in zip(owned, selection) if keep]
                        reference = np.asarray([segment.landmarks[n].local_position.array for n in selected_names])
                        observed = points[chosen[:, None], np.asarray(columns)[selection]]
                        matrices, translations, fitted_scales, valid = align_point_sets_batch(reference=reference, observed=observed)
                        valid_rows = chosen[valid]
                        rotations[valid_rows, column] = rotation_matrices_to_quaternions(matrices=matrices[valid])
                        origins[valid_rows, column] = translations[valid]
                        scales[valid_rows, column] = fitted_scales[valid]
                        solutions[valid_rows, column] = PoseSolution.RIGID_FIT
                        status[valid_rows, column] = 1
            if origin_name not in self.landmark_indices or primary_name not in self.landmark_indices:
                continue
            oi, pi = self.landmark_indices[origin_name], self.landmark_indices[primary_name]
            rows = np.flatnonzero(direction_rows & present[:, oi] & present[:, pi])
            displacement = points[rows, pi] - points[rows, oi]
            length = np.linalg.norm(displacement, axis=1)
            valid = length >= MINIMUM_VECTOR_NORM
            status[rows, column] = 2
            chosen = rows[valid]
            if not len(chosen):
                continue
            local = segment.landmarks[primary_name].local_position.array
            local_length = np.linalg.norm(local)
            if local_length < MINIMUM_VECTOR_NORM:
                raise ValueError("Local primary landmark has zero length")
            sign = segment.frame_definition.primary_axis.sign
            rotations[chosen, column] = direction_pair_quaternions(
                from_directions=UnitVector.from_array(values=np.broadcast_to(sign * local / local_length, (len(chosen), 3))),
                to_directions=UnitVector.from_array(values=sign * displacement[valid] / length[valid, None]))
            origins[chosen, column] = points[chosen, oi]
            scales[chosen, column] = length[valid] / segment.length
            solutions[chosen, column] = PoseSolution.DIRECTION
            status[chosen, column] = 1
        if require_all and np.any(status != 1):
            frame, column = np.argwhere(status != 1)[0]
            error = MissingLandmarkObservations if status[frame, column] == 0 else DegenerateObservations
            raise error(f"Frame {frame}, segment {self.segment_names[column]}: cannot recover pose")
        return HydratedPoseBatch(origins, rotations, scales, solutions, status)
