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
from dataclasses import dataclass

import numpy as np

# Direction used for a bone that has never been observed.
_FALLBACK_DIRECTION: np.ndarray = np.array([0.0, 1.0, 0.0])


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
    _last_direction: dict[str, np.ndarray] = None  # type: ignore[assignment]

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
    ) -> dict[str, np.ndarray]:
        """Rigidify one frame of observed joint positions.

        Parameters
        ----------
        positions : dict[str, (3,) ndarray]
            Observed joint positions this frame. Missing joints are
            simply absent from the dict — they are gap-filled along
            the last-good direction.
        bone_lengths : dict[str, float]
            ``"parent->child" → length (mm)`` to enforce. Bones without
            a positive length are skipped (their subtree is not placed).

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
            length = bone_lengths.get(f"{parent}->{child}")
            if length is None or length <= 0.0:
                continue

            bone_key = f"{parent}->{child}"
            direction: np.ndarray | None = None
            child_obs = positions.get(child)
            if child_obs is not None:
                vector = np.asarray(child_obs, dtype=float) - parent_pos
                norm = float(np.linalg.norm(vector))
                if math.isfinite(norm) and norm > 1e-6:
                    direction = vector / norm
                    self._last_direction[bone_key] = direction
            if direction is None:
                direction = self._last_direction.get(
                    bone_key, _FALLBACK_DIRECTION
                )

            corrected[child] = parent_pos + direction * length

        return corrected

    def reset(self) -> None:
        """Forget all carried directions (e.g. on calibration reload)."""
        self._last_direction.clear()
