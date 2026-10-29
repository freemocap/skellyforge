"""Numerical temporal roll state and prepared topology; no per-pose objects."""
from dataclasses import replace

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import MINIMUM_VECTOR_NORM
from skellyforge.core.math.geometry.rotation_quaternion import (
    conjugate_quaternion_array, hamilton_product, normalize_quaternion_array,
    rotate_vector_pairs, rotation_vectors_to_quaternions,
)
from skellyforge.core.skeleton.pose.pose_tensor import PoseTensor, SolutionCode


def _multiply(left, right):
    return normalize_quaternion_array(quaternions=hamilton_product(left=left, right=right))


class RollTensorState:
    """Prepared dependency groups plus dense previous-orientation state.

    Frames are chronological because transport carries history. Segments at one tree
    depth and disjoint chains run together. Those dependency loops do not materialize
    individual pose, vector, or quaternion objects.
    """
    def __init__(self, *, skeleton, references, rest_relative, rest_world):
        self.segment_names = tuple(skeleton.segments)
        indices = {name: column for column, name in enumerate(self.segment_names)}
        self.parent_indices = np.full(len(indices), -1, np.int64)
        for joint in skeleton.joints.values():
            self.parent_indices[indices[joint.child.name]] = indices[joint.parent.name]
        self.rest_relative = np.stack([rest_relative[name].as_array() for name in self.segment_names])
        self.rest_world = np.stack([rest_world[name].as_array() for name in self.segment_names])
        self.primary_axes = np.stack([references[name].primary_local for name in self.segment_names])
        self.secondary_axes = np.stack([references[name].secondary_local for name in self.segment_names])
        depths = np.full(len(indices), -1, np.int64)
        for level in range(len(indices)):
            ready = (depths < 0) & ((self.parent_indices < 0) | (depths[np.maximum(self.parent_indices, 0)] >= 0))
            if not ready.any():
                break
            depths[ready] = level
        if np.any(depths < 0):
            raise ValueError("Skeleton parent cycle")
        self.depth_groups = tuple(np.flatnonzero(depths == level) for level in range(int(depths.max()) + 1))
        # Only disjoint consecutive chains may commute; preserve authored order
        # whenever chains share segments and may observe each other's updates.
        chain_groups, group, occupied = [], [], set()
        for chain in skeleton.chains.values():
            columns = tuple(indices[segment.name] for segment in chain.segments)
            if occupied.intersection(columns):
                chain_groups.append(group)
                group, occupied = [], set()
            group.append(columns)
            occupied.update(columns)
        if group:
            chain_groups.append(group)
        self.chain_groups = []
        for group in chain_groups:
            columns = np.full((len(group), max(map(len, group))), -1, np.int64)
            for row, chain in enumerate(group):
                columns[row, :len(chain)] = chain
            self.chain_groups.append(columns)
        self.previous_rotations = np.full_like(self.rest_world, np.nan)
        self.previous_present = np.zeros(len(indices), bool)

    def reset(self):
        self.previous_present.fill(False)
        self.previous_rotations.fill(np.nan)

    def _aim(self, *, source, reference, columns):
        before = rotate_vector_pairs(quaternions=reference, vectors=self.primary_axes[columns])
        after = rotate_vector_pairs(quaternions=source, vectors=self.primary_axes[columns])
        cross = np.cross(before, after)
        norms = np.linalg.norm(cross, axis=-1)
        dots = np.sum(before * after, axis=-1)
        regular = norms >= MINIMUM_VECTOR_NORM
        vectors = np.zeros_like(before)
        vectors[regular] = cross[regular] / norms[regular, None] * np.arctan2(norms[regular], dots[regular])[:, None]
        opposite = ~regular & (dots < 0)
        vectors[opposite] = rotate_vector_pairs(quaternions=reference[opposite],
            vectors=self.secondary_axes[columns[opposite]]) * np.pi
        swing = rotation_vectors_to_quaternions(vectors=vectors)
        return _multiply(swing, reference)

    def _backfill(self, *, rotations, codes, present):
        for chain_columns in self.chain_groups:
            valid_columns = chain_columns >= 0
            lookup = np.maximum(chain_columns, 0)
            measured = valid_columns & present[lookup] & (codes[lookup] == SolutionCode.RIGID_FIT)
            anchor = np.max(np.where(measured, np.arange(chain_columns.shape[1]), -1), axis=1)
            active = anchor > 0
            # Each step consumes the distal step's updated orientation; distinct
            # chains within this group share no segment and run together.
            for distance in range(chain_columns.shape[1] - 1):
                rows = np.flatnonzero(active & (anchor - distance > 0))
                if not len(rows):
                    break
                child = chain_columns[rows, anchor[rows] - distance]
                parent = chain_columns[rows, anchor[rows] - distance - 1]
                eligible = present[parent] & present[child] & ~np.isin(codes[parent],
                    (SolutionCode.RIGID_FIT, SolutionCode.OBSERVATION_FRAME))
                active[rows[~eligible]] = False
                parent, child = parent[eligible], child[eligible]
                if not len(parent):
                    continue
                relative = _multiply(conjugate_quaternion_array(quaternions=rotations[parent]), rotations[child])
                change = _multiply(relative, conjugate_quaternion_array(quaternions=self.rest_relative[child]))
                axes = self.primary_axes[parent]
                axes = axes / np.linalg.norm(axes, axis=-1)[:, None]
                projection = np.sum(change[:, 1:] * axes, axis=-1)
                twist = np.concatenate((change[:, :1], axes * projection[:, None]), axis=-1)
                degenerate = np.linalg.norm(twist, axis=-1) < 1e-9
                twist[degenerate] = (1., 0., 0., 0.)
                twist = normalize_quaternion_array(quaternions=twist)
                rotations[parent] = _multiply(rotations[parent], twist)
                codes[parent] = SolutionCode.TRANSPORTED_ROLL

    def resolve(self, *, poses: PoseTensor, reset_before: np.ndarray | None = None) -> PoseTensor:
        if poses.segment_names != self.segment_names:
            raise ValueError("Pose segment identities must match the roll resolver")
        resets = np.zeros(poses.sample_count, bool) if reset_before is None else np.asarray(reset_before)
        if resets.dtype != bool or resets.shape != (poses.sample_count,):
            raise ValueError("Reset flags must be boolean on the sample axis")
        rotations, codes = poses.orientations.copy(), poses.solution_codes.copy()
        presence = poses.segment_present
        for sample in range(poses.sample_count):
            if resets[sample]:
                self.reset()
            present = presence[sample]
            if not present.any():
                continue  # Absence alone is not a new take; caller owns reset policy.
            current = rotations[sample]
            current_codes = codes[sample]
            for group in self.depth_groups:
                columns = group[present[group] & (current_codes[group] == SolutionCode.DIRECTION)]
                if not len(columns):
                    continue
                previous = self.previous_present[columns]
                reference = self.rest_world[columns].copy()
                reference[previous] = self.previous_rotations[columns[previous]]
                parents = self.parent_indices[columns]
                parent_present = (parents >= 0) & present[np.maximum(parents, 0)]
                initial = parent_present & ~previous
                reference[initial] = _multiply(current[parents[initial]], self.rest_relative[columns[initial]])
                carried = parent_present & previous & self.previous_present[np.maximum(parents, 0)]
                reference[carried] = _multiply(_multiply(current[parents[carried]],
                    conjugate_quaternion_array(quaternions=self.previous_rotations[parents[carried]])), reference[carried])
                current[columns] = self._aim(source=current[columns], reference=reference, columns=columns)
                current_codes[columns] = SolutionCode.TRANSPORTED_ROLL
            self._backfill(rotations=current, codes=current_codes, present=present)
            self.previous_rotations[:] = current
            self.previous_present[:] = present
        return replace(poses, orientations=rotations, solution_codes=codes)

    def resolve_independent(self, *, poses: PoseTensor) -> PoseTensor:
        """Single-segment object adapter semantics: retain other segment histories."""
        if poses.segment_names != self.segment_names or poses.sample_count != 1:
            raise ValueError("Independent segment resolution requires one matching sample")
        columns = np.flatnonzero(poses.segment_present[0])
        rotations, codes = poses.orientations.copy(), poses.solution_codes.copy()
        direction = columns[codes[0, columns] == SolutionCode.DIRECTION]
        reference = self.rest_world[direction].copy()
        previous = self.previous_present[direction]
        reference[previous] = self.previous_rotations[direction[previous]]
        rotations[0, direction] = self._aim(source=rotations[0, direction], reference=reference, columns=direction)
        codes[0, direction] = SolutionCode.TRANSPORTED_ROLL
        self.previous_rotations[columns] = rotations[0, columns]
        self.previous_present[columns] = True
        return replace(poses, orientations=rotations, solution_codes=codes)
