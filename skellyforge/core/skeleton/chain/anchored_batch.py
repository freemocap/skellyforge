"""Connected skeleton positions across frames, with explicit named array axes.

Prepare geometry once per fixed recording fit and anchor, then pass bounded frame
chunks. Temporal orientation recovery remains the caller's responsibility. This
module neither fills missing segments nor changes their measured orientations.
"""

from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import UNIT_QUATERNION_TOLERANCE
from skellyforge.core.math.geometry.rotation_quaternion import rotate_vector_pairs
from skellyforge.core.skeleton.chain.synthesis import _validate_fitted_geometry
from skellyforge.core.skeleton.pose.model_scale_fitting import ModelScaleFit
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


@dataclass(frozen=True)
class ConnectedPositionBatch:
    """Positions in the input world frame/units; NaN means absent, never inferred.

    Axes come from the producing AnchoredSkeletonGeometry. Origins are (F,S,3),
    landmarks (F,L,3), and presence arrays (F,S)/(F,L). Inputs are not retained.
    """

    segment_origins: np.ndarray
    landmark_positions: np.ndarray
    segment_present: np.ndarray
    landmark_present: np.ndarray


def _readonly(values, dtype):
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class AnchoredSkeletonGeometry:
    """Snapshot of fitted geometry, independent of mutable model/fit dictionaries.

    Use from_skeleton to validate and prepare. The only execution loop traverses
    tree depths: all frames and all branches at that depth run as NumPy arrays.
    """

    segment_names: tuple[str, ...]
    landmark_names: tuple[str, ...]
    anchor_index: int
    attachment_owners: np.ndarray
    attachment_vectors: np.ndarray
    landmark_owners: np.ndarray
    landmark_vectors: np.ndarray
    traversal_levels: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]

    @classmethod
    def from_skeleton(cls, *, skeleton: SkeletonDefinition, fit: ModelScaleFit,
                      anchor_segment_name: str) -> "AnchoredSkeletonGeometry":
        _validate_fitted_geometry(skeleton=skeleton, fit=fit)
        names = tuple(skeleton.segments)
        indices = {name: i for i, name in enumerate(names)}
        if anchor_segment_name not in indices:
            raise ValueError("Anchor must name a known segment")
        adjacency: list[list[tuple[int, int, int]]] = [[] for _ in names]
        owners, vectors = [], []
        for edge, joint in enumerate(skeleton.joints.values()):
            parent, child = indices[joint.parent.name], indices[joint.child.name]
            owners.append(parent)
            vectors.append(joint.connect_at.local_position.array * fit.segment_scales[joint.parent.name])
            adjacency[parent].append((child, edge, 1))
            adjacency[child].append((parent, edge, -1))
        anchor = indices[anchor_segment_name]
        visited, frontier, levels = {anchor}, [anchor], []
        while frontier:
            rows, next_frontier = [], []
            for parent in frontier:
                for child, edge, direction in adjacency[parent]:
                    if child in visited:
                        continue
                    visited.add(child)
                    next_frontier.append(child)
                    rows.append((parent, child, edge, direction))
            if rows:
                levels.append(tuple(_readonly(column, np.int64) for column in zip(*rows)))
            frontier = next_frontier
        if len(visited) != len(names) or len(owners) != len(names) - 1:
            raise ValueError("Batch synthesis requires a connected skeleton tree")
        landmarks = tuple(skeleton.landmarks)
        return cls(
            segment_names=names, landmark_names=landmarks, anchor_index=anchor,
            attachment_owners=_readonly(owners, np.int64),
            attachment_vectors=_readonly(vectors, np.float64).reshape((-1, 3)),
            landmark_owners=_readonly([indices[skeleton.landmarks[n].segment] for n in landmarks], np.int64),
            landmark_vectors=_readonly([
                skeleton.landmarks[n].local_position.array * fit.segment_scales[skeleton.landmarks[n].segment]
                for n in landmarks], np.float64).reshape((-1, 3)),
            traversal_levels=tuple(levels),
        )

    def positions(self, *, world_orientations: np.ndarray, segment_present: np.ndarray,
                  anchor_origins: np.ndarray) -> ConnectedPositionBatch:
        """Calculate one chunk without a per-frame or per-landmark Python loop.

        Orientations: (F,S,4), unit wxyz; presence: boolean (F,S); anchors: (F,3).
        Selected segments must connect to the anchor in each nonempty frame.
        Empty frames are allowed. Values in absent slots are ignored. No inputs
        are normalized, mutated, filled, or silently dropped.
        """
        rotations = np.asarray(world_orientations, dtype=np.float64)
        present = np.asarray(segment_present)
        anchors = np.asarray(anchor_origins, dtype=np.float64)
        if rotations.ndim != 3 or rotations.shape[1:] != (len(self.segment_names), 4):
            raise ValueError("Orientations must have shape (frames, segments, 4)")
        frames = len(rotations)
        if present.dtype != np.bool_ or present.shape != rotations.shape[:2]:
            raise ValueError("Presence must be boolean (frames, segments)")
        if anchors.shape != (frames, 3):
            raise ValueError("Anchor origins must have shape (frames, 3)")
        active = present.any(axis=1)
        if np.any(active & ~present[:, self.anchor_index]):
            raise ValueError("Selected segments must include the anchor")
        if not np.isfinite(anchors[active]).all():
            raise ValueError("Active anchor origins must be finite")
        selected = rotations[present]
        norms = np.sqrt(np.sum(selected * selected, axis=1))
        if not np.isfinite(selected).all() or np.any(np.abs(norms - 1) > UNIT_QUATERNION_TOLERANCE):
            raise ValueError("Selected orientations must be finite unit quaternions")
        safe_rotations = np.where(present[..., None], rotations, np.array([1., 0., 0., 0.]))

        def rotate_offsets(owners, vectors):
            return rotate_vector_pairs(
                quaternions=safe_rotations[:, owners].reshape((-1, 4)),
                vectors=np.broadcast_to(vectors, (frames, len(owners), 3)).reshape((-1, 3)),
            ).reshape((frames, len(owners), 3))

        offsets = rotate_offsets(self.attachment_owners, self.attachment_vectors)
        origins = np.full((frames, len(self.segment_names), 3), np.nan)
        origins[active, self.anchor_index] = anchors[active]
        reached = np.zeros_like(present)
        reached[:, self.anchor_index] = active
        for parents, children, edges, signs in self.traversal_levels:
            valid = reached[:, parents] & present[:, children]
            reached[:, children] = valid
            origins[:, children] = np.where(valid[..., None],
                origins[:, parents] + offsets[:, edges] * signs[None, :, None], np.nan)
        if not np.array_equal(reached, present):
            raise ValueError("Selected segments must be connected to the anchor without gaps")
        landmark_present = present[:, self.landmark_owners]
        landmarks = np.where(landmark_present[..., None],
            origins[:, self.landmark_owners] + rotate_offsets(self.landmark_owners, self.landmark_vectors), np.nan)
        if not np.isfinite(origins[present]).all() or not np.isfinite(landmarks[landmark_present]).all():
            raise ValueError("Connected positions must remain finite")
        return ConnectedPositionBatch(origins, landmarks, present.copy(), landmark_present)
