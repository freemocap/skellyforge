"""Translation-only weighted attachments, with fixed observed anchor.

Minimize ||t-o||² + strength ||B t-d||² over present segments. Every recovered
position has unit weight. Rotations, fitted offsets, and presence are untouched.
The small operator depends only on topology, presence, and fixed strength; its
SVD is cached by presence pattern, never repeated for each frame in a chunk.
"""

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import UNIT_QUATERNION_TOLERANCE
from skellyforge.core.math.geometry.rotation_quaternion import rotate_vector_pairs
from skellyforge.core.skeleton.chain.anchored_batch import AnchoredSkeletonGeometry, ConnectedPositionBatch, _readonly
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


@dataclass(frozen=True)
class WeightedAttachmentGeometry:
    """Prepared weighted placement; cache holds at most 128 presence patterns.

    Absent segments/edges contribute no terms. Disconnected present components
    retain their observed translation preference; no missing pose is completed.
    Any named anchor is supported, and is fixed only when it is present.
    Prepared read-only topology must not be mutated, including by temporarily
    changing array writeability. Fitted geometry vectors may change independently.
    """

    segment_names: tuple[str, ...]
    anchor_index: int
    edge_parents: np.ndarray
    edge_children: np.ndarray
    landmark_names: tuple[str, ...]
    landmark_owners: np.ndarray
    strength: float = 1.0
    _operators: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False, compare=False)
    _validated_topology: tuple | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not np.isfinite(self.strength) or self.strength < 0:
            raise ValueError("Attachment strength must be finite and nonnegative")

    @classmethod
    def from_skeleton(cls, *, skeleton: SkeletonDefinition, anchor_segment_name: str,
                      strength: float = 1.0) -> "WeightedAttachmentGeometry":
        names = tuple(skeleton.segments)
        indices = {name: i for i, name in enumerate(names)}
        if anchor_segment_name not in indices:
            raise ValueError("Anchor must name a known segment")
        return cls(names, indices[anchor_segment_name],
                   _readonly([indices[j.parent.name] for j in skeleton.joints.values()], np.int64),
                   _readonly([indices[j.child.name] for j in skeleton.joints.values()], np.int64),
                   tuple(skeleton.landmarks),
                   _readonly([indices[landmark.segment] for landmark in skeleton.landmarks.values()], np.int64),
                   float(strength))

    def _validate_geometry(self, geometry):
        if (geometry.attachment_vectors.shape != (len(self.edge_parents), 3)
                or geometry.landmark_vectors.shape != (len(self.landmark_names), 3)):
            raise ValueError("Fitted geometry vectors must match declared edge and landmark axes")
        topology = (geometry.attachment_owners, geometry.landmark_owners,
                    *(column for level in geometry.traversal_levels for column in level))
        signature = None
        if all(not column.flags.writeable for column in topology):
            # Prepared read-only topology is immutable by contract. Retain one
            # set of references so replacement arrays cannot reuse cached IDs.
            # Never retain fitted vectors here: scales may change every sample.
            signature = (geometry.segment_names, geometry.anchor_index, geometry.landmark_names,
                         tuple(len(level) for level in geometry.traversal_levels),
                         topology)
            cached = self._validated_topology
            if (cached is not None and signature[:4] == cached[:4]
                    and len(topology) == len(cached[4])
                    and all(column is previous for column, previous in zip(topology, cached[4]))):
                return
        if geometry.segment_names != self.segment_names or geometry.anchor_index != self.anchor_index:
            raise ValueError("Geometry segment axis and anchor must match prepared topology")
        if (geometry.attachment_owners.dtype.kind not in "iu"
                or not np.array_equal(geometry.attachment_owners, self.edge_parents)):
            raise ValueError("Geometry attachment owners must match prepared topology")
        if (geometry.landmark_names != self.landmark_names
                or geometry.landmark_owners.dtype.kind not in "iu"
                or not np.array_equal(geometry.landmark_owners, self.landmark_owners)):
            raise ValueError("Geometry landmark axis and owners must match prepared topology")
        # Same owners alone cannot distinguish two joints leaving one segment.
        # Verify both endpoints by edge identity, honoring traversal rerooting.
        rows = []
        for level in geometry.traversal_levels:
            if (len(level) != 4 or any(column.ndim != 1 or column.dtype.kind not in "iu" for column in level)
                    or any(column.shape != level[0].shape for column in level)):
                raise ValueError("Geometry edge topology must match prepared topology")
            rows.append(np.column_stack(level))
        traversal = np.concatenate(rows) if rows else np.empty((0, 4), dtype=np.int64)
        edge_count = len(self.edge_parents)
        if (traversal.shape != (edge_count, 4)
                or not np.array_equal(np.sort(traversal[:, 2]), np.arange(edge_count))
                or not np.isin(traversal[:, 3], (-1, 1)).all()):
            raise ValueError("Geometry edge topology must match prepared topology")
        parents, children, edges, signs = traversal.T
        forward = signs == 1
        if (not np.array_equal(parents, np.where(forward, self.edge_parents[edges], self.edge_children[edges]))
                or not np.array_equal(children, np.where(forward, self.edge_children[edges], self.edge_parents[edges]))):
            raise ValueError("Geometry edge topology must match prepared topology")
        object.__setattr__(self, "_validated_topology", signature)

    def _operator(self, present):
        key = present.tobytes()
        cached = self._operators.get(key)
        if cached is not None:
            self._operators.move_to_end(key)
            return cached
        free = np.flatnonzero(present & (np.arange(len(present)) != self.anchor_index))
        edges = np.flatnonzero(present[self.edge_parents] & present[self.edge_children])
        incidence = (self.edge_children[edges, None] == free[None, :]).astype(float)
        incidence -= self.edge_parents[edges, None] == free[None, :]
        if len(edges):
            # Forest incidence rows are independent, including the fixed anchor
            # elimination. Singular values avoid cancellation in normal equations.
            left, singular, right = np.linalg.svd(incidence, full_matrices=False)
            if self.strength <= 1:
                gain = self.strength * singular / (1 + self.strength * singular**2)
            else:
                gain = singular / (1 / self.strength + singular**2)
            correction = (right.T * gain) @ left.T
        else:
            correction = np.empty((len(free), 0))
        cached = free, edges, correction
        self._operators[key] = cached
        if len(self._operators) > 128:
            self._operators.popitem(last=False)
        return cached

    def positions(self, *, geometry: AnchoredSkeletonGeometry, world_orientations: np.ndarray,
                  segment_present: np.ndarray, observed_origins: np.ndarray) -> ConnectedPositionBatch:
        """Batch matching presence patterns and preserve caller-owned arrays."""
        self._validate_geometry(geometry)
        rotations = np.asarray(world_orientations, dtype=np.float64)
        present = np.asarray(segment_present)
        observed = np.asarray(observed_origins, dtype=np.float64)
        if rotations.ndim != 3 or rotations.shape[1:] != (len(self.segment_names), 4):
            raise ValueError("Orientations must have shape (frames, segments, 4)")
        if present.dtype != np.bool_ or present.shape != rotations.shape[:2]:
            raise ValueError("Presence must be boolean (frames, segments)")
        if observed.shape != (*rotations.shape[:2], 3):
            raise ValueError("Observed origins must have shape (frames, segments, 3)")
        if not np.isfinite(observed[present]).all():
            raise ValueError("Present observed origins must be finite")
        selected = rotations[present]
        if (not np.isfinite(selected).all()
                or np.any(np.abs(np.linalg.norm(selected, axis=-1) - 1) > UNIT_QUATERNION_TOLERANCE)):
            raise ValueError("Present orientations must be finite unit quaternions")
        if not np.isfinite(geometry.attachment_vectors).all() or not np.isfinite(geometry.landmark_vectors).all():
            raise ValueError("Fitted geometry vectors must be finite")
        frames = len(rotations)
        safe = np.where(present[..., None], rotations, np.array([1., 0., 0., 0.]))

        def rotate(owners, vectors):
            return rotate_vector_pairs(quaternions=safe[:, owners].reshape((-1, 4)),
                vectors=np.broadcast_to(vectors, (frames, len(owners), 3)).reshape((-1, 3))
            ).reshape((frames, len(owners), 3))

        origins = np.where(present[..., None], observed, np.nan)
        if self.strength > 0 and frames:
            offsets = rotate(geometry.attachment_owners, geometry.attachment_vectors)
            patterns, groups = np.unique(present, axis=0, return_inverse=True)
            # Loop over unique patterns, not frames; every matching frame batches.
            for group, pattern in enumerate(patterns):
                free, edges, correction = self._operator(pattern)
                rows = np.flatnonzero(groups == group)
                residual = (observed[rows[:, None], self.edge_children[edges]]
                            - observed[rows[:, None], self.edge_parents[edges]] - offsets[rows[:, None], edges])
                origins[rows[:, None], free] -= np.einsum("se,fec->fsc", correction, residual)
        landmark_present = present[:, geometry.landmark_owners].copy()
        landmarks = np.where(landmark_present[..., None], origins[:, geometry.landmark_owners]
                             + rotate(geometry.landmark_owners, geometry.landmark_vectors), np.nan)
        if not np.isfinite(origins[present]).all() or not np.isfinite(landmarks[landmark_present]).all():
            raise ValueError("Weighted positions must remain finite")
        return ConnectedPositionBatch(origins, landmarks, present.copy(), landmark_present)
