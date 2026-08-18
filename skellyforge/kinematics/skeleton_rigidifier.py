"""Single-frame forward-pass rigidifier for realtime skeleton correction.

This is the streaming counterpart of the posthoc ``enforce_rigid_bones``
step: for one frame, anchor each present root at its observed position,
walk the tree from the roots outward in BFS order, and for each bone take
the direction from the *corrected* parent toward the observed child,
normalize it, and place the child exactly ``length`` away.

Keeping the observed direction but overriding the length holds rigid
segment lengths while tracking the subject's pose. Per-bone last-good
directions are carried across frames so a joint that drops out for a few
frames is gap-filled along its last direction instead of collapsing onto
its parent.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# Direction used for a bone that has never been observed.
_FALLBACK_DIRECTION: np.ndarray = np.array([0.0, 1.0, 0.0])

from typing import TYPE_CHECKING

from skellyforge.kinematics.rigid_point_set import (
    RigidPointTemplate,
    fit_template_to_observed,
)

if TYPE_CHECKING:
    from skellyforge.skellymodels.standard_human.human_skeleton import HumanSkeleton
    from skellyforge.skellymodels.standard_human.standard_human_tpose import StandardHumanTPose


@dataclass
class TreeRigidifier:
    """Forward-pass rigidify over a fixed joint hierarchy.

    The tree topology (roots + BFS edge order) is computed once at
    construction; ``rigidify`` is the per-frame hot path. Stateful
    across calls: remembers each bone's last-good direction for
    gap-filling.

    Parameters
    ----------
    joint_hierarchy : dict[str, list[str]]
        Parent → children mapping. Roots are nodes that appear as parents
        but never as children.
    """

    joint_hierarchy: dict[str, list[str]]

    _roots: tuple[str, ...] = ()
    _edges: tuple[tuple[str, str], ...] = ()
    _last_direction: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        children_of = {
            parent: list(children)
            for parent, children in self.joint_hierarchy.items()
        }
        all_children = {
            c for children in children_of.values() for c in children
        }
        roots = [
            parent
            for parent in children_of
            if parent not in all_children
        ]

        edges: list[tuple[str, str]] = []
        visited: set[str] = set(roots)
        queue: deque[str] = deque(roots)
        while queue:
            parent = queue.popleft()
            for child in children_of.get(parent, []):
                edges.append((parent, child))
                if child not in visited:
                    visited.add(child)
                    queue.append(child)

        self._roots = tuple(roots)
        self._edges = tuple(edges)
        self._last_direction = {}

    def rigidify(
        self,
        positions: dict[str, np.ndarray],
        bone_lengths: dict[str, float],
        *,
        return_directions: bool = False,
    ) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Rigidify one frame of observed joint positions.

        Parameters
        ----------
        positions : dict[str, (3,) ndarray]
            Observed joint positions this frame. Missing joints are
            simply absent from the dict — they are gap-filled along
            the last-good direction.
        bone_lengths : dict[str, float]
            ``node_name → length (mm)`` — the length to enforce FOR the
            named node, keyed by the node's name (the parent is known from
            the hierarchy). Bones without a positive length are skipped
            (their subtree is not placed).
        return_directions : bool
            When ``False`` (default), returns only the corrected positions
            (backwards-compatible). When ``True``, returns ``(corrected,
            directions)`` where ``directions`` maps each CHILD node name
            to the unit direction actually used this frame (observed-
            normalized, else last-good/fallback).

        Returns
        -------
        dict[str, (3,) ndarray]
            Rigidified positions for every joint reachable from a
            present root.
        tuple[dict[str, (3,) ndarray], dict[str, (3,) ndarray]]
            When ``return_directions`` is ``True``: the corrected positions
            plus the per-edge unit directions used this frame (keyed by the
            child node's name).
        """
        corrected: dict[str, np.ndarray] = {}
        directions: dict[str, np.ndarray] = {}
        for root in self._roots:
            obs = positions.get(root)
            if obs is not None:
                corrected[root] = np.asarray(obs, dtype=float).copy()

        for parent, child in self._edges:
            parent_pos = corrected.get(parent)
            if parent_pos is None:
                continue
            length = bone_lengths.get(child)
            if length is None or length <= 0.0:
                continue

            direction: np.ndarray | None = None
            child_obs = positions.get(child)
            if child_obs is not None:
                vector = np.asarray(child_obs, dtype=float) - parent_pos
                norm = float(np.linalg.norm(vector))
                if math.isfinite(norm) and norm > 1e-6:
                    direction = vector / norm
                    self._last_direction[child] = direction
            if direction is None:
                direction = self._last_direction.get(
                    child, _FALLBACK_DIRECTION
                )

            directions[child] = direction
            corrected[child] = parent_pos + direction * length

        if return_directions:
            return corrected, directions
        return corrected

    def reset(self) -> None:
        """Forget all carried directions (e.g. on calibration reload)."""
        self._last_direction.clear()


def rigidify_landmarks(
    skeleton: "HumanSkeleton",
    tpose: "StandardHumanTPose",
    landmarks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Rigidify one frame of hydrated landmarks against the skeleton's rest shape.

    Derived entirely from the loaded model: the joint hierarchy is the parent
    edges, the bone lengths are the derived segment lengths, and the rigid
    bodies are the segments whose landmark set has 3+ points (a rotation-pinned
    Procrustes of their rest positions onto the observed set).
    """
    joint_hierarchy: dict[str, list[str]] = {}
    for segment in skeleton.segments:
        if segment.parent is None:
            joint_hierarchy.setdefault(segment.name, [])
        else:
            joint_hierarchy.setdefault(segment.parent.name, []).append(segment.name)

    # The tree nodes are SEGMENT ORIGINS (their shared landmark position).
    origins = {
        segment.name: landmarks[segment.origin_landmark.name]
        for segment in skeleton.segments
        if segment.origin_landmark.name in landmarks
    }
    # The bone from a parent to a child is the REST distance between the two
    # segments' origins (a distal-attached child == the parent's length; an
    # origin-attached child like the clavicle is a different span).
    bone_lengths: dict[str, float] = {}
    for segment in skeleton.segments:
        if segment.parent is None:
            continue
        parent_origin = tpose.landmarks[segment.parent.origin_landmark.name]
        child_origin = tpose.landmarks[segment.origin_landmark.name]
        bone_lengths[segment.name] = float(np.linalg.norm(child_origin - parent_origin))
    corrected_origins = TreeRigidifier(joint_hierarchy).rigidify(origins, bone_lengths)

    result = dict(landmarks)
    for name, pos in corrected_origins.items():
        segment = next(s for s in skeleton.segments if s.name == name)
        result[segment.origin_landmark.name] = pos

    # Rigid bodies: rotation-pinned Procrustes of the rest shape onto observed.
    for segment in skeleton.segments:
        if len(segment.landmarks) < 3:
            continue
        names = tuple(lm.name for lm in segment.landmarks)
        rest = np.asarray([tpose.landmarks[n] for n in names], dtype=np.float64)
        anchor = segment.origin_landmark.name
        anchor_idx = names.index(anchor) if anchor in names else 0
        template = RigidPointTemplate(
            point_names=names,
            positions=rest - rest[anchor_idx],
            pair_distances=_pairwise_distances(names, rest),
        )
        result.update(fit_template_to_observed(template, result, anchor_name=anchor))

    return result


def _pairwise_distances(
    names: tuple[str, ...], positions: np.ndarray
) -> dict[tuple[str, str], float]:
    distances: dict[tuple[str, str], float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = (names[i], names[j]) if names[i] < names[j] else (names[j], names[i])
            distances[key] = float(np.linalg.norm(positions[i] - positions[j]))
    return distances
