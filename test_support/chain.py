"""Synthetic chain inputs; no experiment execution or output generation."""
import itertools
import numpy as np
from .geometry import rotation


def chain_inputs(*, noise, gap, extension=0.0, branching=False):
    local = np.array(
        list(itertools.product([-25.0, 25.0], [-25.0, 25.0], [-80.0, 80.0]))
    )
    parent = np.array([[0.0, 0.0, 80.0], [0.0, 0.0, 80.0]])
    child = -parent
    if branching:
        parent = np.array([[0., -45., 80.], [0., 45., 80.]])
    times = np.linspace(0, 2, 41)
    missing = set(range(20 - gap // 2, 21 + gap // 2)) if gap else set()
    rng = np.random.default_rng(7)
    records = []
    for i, time in enumerate(times):
        phase = 2 * np.pi * time / 2
        # A local bend centered in the gap prevents initialization from being an
        # exact answer: distal observations must influence the middle quaternion.
        bend = 0.45 * np.exp(-(((time - 1) / 0.16) ** 2))
        rotations = [
            rotation([1, 2, 0], 0.2 * np.sin(phase)),
            rotation([1, 0.2, 0], 0.5 + 0.3 * np.sin(phase) + bend),
            rotation([0, 1, 0.2], -0.4 + 0.25 * np.cos(phase)),
        ]
        if branching:
            rotations[1] = rotation([1, 0, 0], 0.8 + 0.3 * np.sin(phase) + bend)
            rotations[2] = rotation([1, 0, 0], -0.8 + 0.25 * np.cos(phase))
        displacement = extension * np.sin(np.pi * time / 2) ** 2
        translations = [np.array([30 * np.sin(phase), 15 * np.cos(phase), 130.0])]
        for b in range(1, 3):
            parent_index = 0 if branching else b - 1
            translations.append(
                translations[parent_index]
                + rotations[parent_index].apply(
                    parent[b - 1] + ([0, 0, displacement] if b == 2 else np.zeros(3))
                )
                - rotations[b].apply(child[b - 1])
            )
        truth = np.array([r.apply(local) + t for r, t in zip(rotations, translations)])
        observed = truth + rng.normal(0, noise, truth.shape)
        records.append(
            dict(
                displacement=float(displacement),
                rotations=rotations,
                translations=translations,
                truth=truth,
                observed=observed,
            )
        )
    return local, parent, child, times, missing, records
