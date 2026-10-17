"""Center of mass for frame chunks, preserving visible-landmark renormalization."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from skellyforge.core.biomechanics.center_of_mass import CenterOfMassDefinitions


@dataclass(frozen=True)
class CenterOfMassBatch:
    segment_positions: np.ndarray
    segment_present: np.ndarray
    body_positions: np.ndarray
    body_present: np.ndarray


@dataclass(frozen=True)
class CenterOfMassGeometry:
    landmark_names: tuple[str, ...]
    segment_names: tuple[str, ...]
    landmark_indices: np.ndarray
    weights: np.ndarray
    masses: np.ndarray

    @classmethod
    def from_definitions(cls, *, definitions: CenterOfMassDefinitions,
                         landmark_names: tuple[str, ...], segment_masses: Mapping[str, float]) -> "CenterOfMassGeometry":
        if len(set(landmark_names)) != len(landmark_names):
            raise ValueError("Landmark names must be unique")
        lookup = {name: i for i, name in enumerate(landmark_names)}
        entries = [(name, side, definition) for definition in definitions.definitions.values() for name, side in definition.side_entries]
        width = max(len(d.weights) for _, _, d in entries)
        indices = np.full((len(entries), width), len(lookup), dtype=np.int64)
        weights = np.zeros(indices.shape)
        for i, (_, side, definition) in enumerate(entries):
            for j, weight in enumerate(definition.weights):
                indices[i, j] = lookup.get(definition.resolve_landmark(side=side, landmark=weight.landmark), len(lookup))
                weights[i, j] = weight.weight
        masses = np.asarray([segment_masses.get(name, 0.) for name, _, _ in entries], dtype=float)
        if not np.isfinite(masses).all():
            raise ValueError("Masses must be finite")
        for array in (indices, weights, masses):
            array.setflags(write=False)
        return cls(landmark_names, tuple(n for n, _, _ in entries), indices, weights, masses)

    def calculate(self, *, positions: np.ndarray, landmark_present: np.ndarray) -> CenterOfMassBatch:
        positions = np.asarray(positions, dtype=float)
        present = np.asarray(landmark_present)
        if positions.ndim != 3 or positions.shape[1:] != (len(self.landmark_names), 3):
            raise ValueError("Positions must have shape (frames, landmarks, 3)")
        if present.dtype != np.bool_ or present.shape != positions.shape[:2] or not np.isfinite(positions[present]).all():
            raise ValueError("Presence must match positions and present values must be finite")
        # One sentinel column represents missing names and padded weight slots.
        points = np.concatenate((np.where(present[..., None], positions, 0.), np.zeros((len(positions), 1, 3))), axis=1)
        masks = np.concatenate((present, np.zeros((len(present), 1), dtype=bool)), axis=1)
        weights = np.where(masks[:, self.landmark_indices], self.weights, 0.)
        totals = np.add.accumulate(weights, axis=2)[:, :, -1]
        weighted = points[:, self.landmark_indices] * weights[..., None]
        sums = np.add.accumulate(weighted, axis=2)[:, :, -1]
        segment_present = totals > 0
        centers = sums / np.where(segment_present, totals, 1.)[..., None]
        mass = np.where(segment_present, self.masses, 0.)
        total_mass = np.add.accumulate(mass, axis=1)[:, -1]
        body_present = total_mass > 0
        body = np.add.accumulate(centers * mass[..., None], axis=1)[:, -1] / np.where(body_present, total_mass, 1.)[:, None]
        return CenterOfMassBatch(np.where(segment_present[..., None], centers, np.nan), segment_present,
            np.where(body_present[:, None], body, np.nan), body_present)
