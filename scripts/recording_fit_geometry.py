"""Geometric incompatibilities in direct targets, independent of optimizer weights."""

from itertools import combinations
import numpy as np


def rigid_target_checks(skeleton, fit, targets):
    """Distance discrepancies for targets fixed to the same rigid segment.

    A target at a joint's parent attachment is also the child's origin, and vice
    versa. Propagate only those exact identities; never treat an entire linked
    chain as rigid. The reverse triangle inequality gives the lower bound:
    max(endpoint errors) >= abs(observed span - fixed span) / 2.
    This is incompatibility with the supplied geometry, not measurement truth.
    """
    attached = {n: {} for n in skeleton.segments}
    for name in targets:
        landmark = skeleton.landmarks[name]
        attached[landmark.segment][name] = (
            landmark.local_position.array * fit.segment_scales[landmark.segment]
        )
    changed = True
    while changed:
        changed = False
        for joint in skeleton.joints.values():
            parent, child = joint.parent.name, joint.child.name
            connection = (
                joint.connect_at.local_position.array * fit.segment_scales[parent]
            )
            for source, destination, at_source, at_destination in (
                (parent, child, connection, np.zeros(3)),
                (child, parent, np.zeros(3), connection),
            ):
                for name, position in tuple(attached[source].items()):
                    if name not in attached[destination] and np.array_equal(
                        position, at_source
                    ):
                        attached[destination][name] = at_destination.copy()
                        changed = True
    checks, seen = [], set()
    for segment, points in attached.items():
        for a, b in combinations(sorted(points), 2):
            if (a, b) in seen:
                continue
            seen.add((a, b))
            fixed = float(np.linalg.norm(points[a] - points[b]))
            observed = float(
                np.linalg.norm(targets[a].position.array - targets[b].position.array)
            )
            checks.append(
                dict(
                    segment=segment,
                    targets=[a, b],
                    fixed_span_mm=fixed,
                    observed_span_mm=observed,
                    minimum_possible_max_endpoint_error_mm=abs(observed - fixed) / 2,
                )
            )
    return sorted(
        checks, key=lambda x: x["minimum_possible_max_endpoint_error_mm"], reverse=True
    )


def spine_bend_degrees(segments):
    a = np.asarray(segments["sacrolumbar"]["end"]) - segments["sacrolumbar"]["origin"]
    b = np.asarray(segments["thoracic"]["end"]) - segments["thoracic"]["origin"]
    return float(
        np.rad2deg(
            np.arccos(
                np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)
            )
        )
    )
