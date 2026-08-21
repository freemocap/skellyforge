"""Single-frame forward-pass rigidifier for realtime skeleton correction.

This is the streaming counterpart of the posthoc ``enforce_rigid_bones``
step: for one frame, anchor each present root at its observed position,
walk the tree from the roots outward in BFS order, and for each bone take
the direction from the *corrected* parent toward the observed child,
normalize it, and place the child exactly ``length`` away.

Keeping the observed direction but overriding the length holds rigid segment
lengths while tracking the subject's pose. A bone whose child origin is missing
this frame is placed along its own T-pose rest direction (supplied per call), so
an unobserved segment follows its parent at the reference-pose angle instead of
collapsing onto it or shooting off along a fixed world axis. The action is
STATELESS: it carries nothing across frames.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from typing import TYPE_CHECKING

from skellyforge.kinematics.rigid_point_set import (
    RigidPointTemplate,
    fit_template_to_observed,
)
from skellyforge.standard_human.human_skeleton import HumanSkeleton
from skellyforge.standard_human.standard_human_tpose import StandardHumanTPose


@dataclass
class TreeRigidifier:
    """Forward-pass rigidify over a fixed joint hierarchy.

    The tree topology (roots + BFS edge order) is computed once at
    construction; ``rigidify`` is the per-frame hot path. The action is
    STATELESS: it carries nothing across frames, so it may be constructed
    fresh per frame or reused freely with identical results.

    Parameters
    ----------
    joint_hierarchy : dict[str, list[str]]
        Parent -> children mapping. Roots are nodes that appear as parents
        but never as children.
    """

    joint_hierarchy: dict[str, list[str]]

    _roots: tuple[str, ...] = ()
    _edges: tuple[tuple[str, str], ...] = ()

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

    def rigidify(
        self,
        positions: dict[str, np.ndarray],
        bone_lengths: dict[str, float],
        fallback_directions: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Rigidify one frame of observed joint positions.

        Parameters
        ----------
        positions : dict[str, (3,) ndarray]
            Observed joint positions this frame. A joint absent from the dict
            is placed along its ``fallback_directions`` entry.
        bone_lengths : dict[str, float]
            ``node_name -> length (mm)`` -- the length to enforce FOR the named
            node, keyed by the node's name (the parent is known from the
            hierarchy). Bones without a positive length are skipped (their
            subtree is not placed).
        fallback_directions : dict[str, np.ndarray]
            ``node_name -> (3,) unit ndarray`` -- the world direction from the
            parent toward this node at the T-pose, used when the node is not
            observed this frame (so an unobserved segment follows its parent at
            the reference-pose angle). Required for every non-root node.

        Returns
        -------
        dict[str, (3,) ndarray]
            Rigidified positions for every joint reachable from a
            present root.
        """
        corrected: dict[str, np.ndarray] = {}
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
            if direction is None:
                direction = fallback_directions[child]

            corrected[child] = parent_pos + direction * length

        return corrected


def rigidify_landmarks(
    skeleton: HumanSkeleton,
    tpose: StandardHumanTPose,
    landmarks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Rigidify one frame of hydrated landmarks against the skeleton's rest shape.

    Two passes, in dependency order:

    1. Multi-point rigid bodies (3+ landmarks) are Procrustes-fitted FIRST, off
       the raw hydrated landmarks, so their DERIVED landmarks (carpals, knuckles)
       are correct before anything hangs off them.
    2. The two-point chains are then forward-passed from those corrected origins
       (enforcing bone lengths along the observed directions), and each two-point
       segment's distal landmark (fingertips, toes) is placed at the enforced
       segment length so a terminal bone is not left at its raw keypoint.
    """
    joint_hierarchy: dict[str, list[str]] = {}
    origin_name_by_segment: dict[str, str] = {}
    for segment in skeleton.segments:
        origin_name_by_segment[segment.name] = segment.origin_landmark.name
        if segment.parent is None:
            joint_hierarchy.setdefault(segment.name, [])
        else:
            joint_hierarchy.setdefault(segment.parent.name, []).append(segment.name)

    # The bone from a parent to a child is the REST offset between the two
    # segments' origins (a distal-attached child == the parent's length; an
    # origin-attached child like the clavicle is a different span). Its length is
    # enforced every frame; its normalized direction is the fallback used when
    # the child origin is missing this frame, so an unobserved segment follows
    # its parent at the reference-pose angle instead of a fixed world axis.
    bone_lengths: dict[str, float] = {}
    rest_directions: dict[str, np.ndarray] = {}
    for segment in skeleton.segments:
        if segment.parent is None:
            continue
        parent_origin = tpose.landmarks[segment.parent.origin_landmark.name]
        child_origin = tpose.landmarks[segment.origin_landmark.name]
        offset = child_origin - parent_origin
        length = float(np.linalg.norm(offset))
        bone_lengths[segment.name] = length
        rest_directions[segment.name] = (
            offset / length if length > 1e-9 else np.zeros(3, dtype=float)
        )

    result = dict(landmarks)

    # 1. Fit multi-point rigid bodies first, so their derived landmarks are
    #    correct before the two-point chains hang off them.
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

    # 2. Forward-pass the segment origins (two-point chains) from the corrected
    #    multi-point origins.
    origins = {
        segment.name: result[segment.origin_landmark.name]
        for segment in skeleton.segments
        if segment.origin_landmark.name in result
    }
    corrected_origins = TreeRigidifier(joint_hierarchy).rigidify(
        origins, bone_lengths, rest_directions
    )
    for name, pos in corrected_origins.items():
        result[origin_name_by_segment[name]] = pos

    # 3. Place each two-point segment's distal landmark (fingertips, toes) at the
    #    enforced segment length along the observed direction. The forward pass
    #    only places ORIGINS, so a terminal landmark would otherwise stay at its
    #    raw keypoint and the terminal bone length would not be enforced.
    for segment in skeleton.segments:
        if len(segment.landmarks) != 2:
            continue
        origin_name = segment.origin_landmark.name
        origin = result.get(origin_name)
        if origin is None:
            continue
        distal = next(
            (lm for lm in segment.landmarks if lm.name != origin_name), None
        )
        if distal is None or distal.name not in landmarks:
            continue
        length = tpose.segments[segment.name].length
        if length <= 0.0:
            continue
        vector = (
            np.asarray(landmarks[distal.name], dtype=float)
            - np.asarray(origin, dtype=float)
        )
        norm = float(np.linalg.norm(vector))
        if math.isfinite(norm) and norm > 1e-6:
            result[distal.name] = (
                np.asarray(origin, dtype=float) + (vector / norm) * length
            )

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
