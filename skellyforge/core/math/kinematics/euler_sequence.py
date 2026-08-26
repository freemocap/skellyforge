"""Euler-angle decomposition and composition for named rotation sequences.

WHY EULER EXISTS HERE (when the whole pipeline is quaternion-first): a
quaternion IS the rotation but cannot be read by a human, and every consumer of
joint angles - clinicians, the ISB conventions, published gait-analysis data -
speaks euler triples ("flexion / abduction / axial rotation"). Euler is
therefore a NAMING CONVENTION applied once, at the linkage layer's read-out
boundary: `JointPose` carries the authoritative quaternion beside its
decomposed angles. All composing, blending, transporting and transporting-back
stays in quaternions; nothing downstream ever chains euler values together.

Euler's known pathologies are contained by construction:
- gimbal lock only degrades the three NUMBERS near a lock; the quaternion
  beside them is untouched, and decompose/compose remains an exact inverse
  even at the lock (pinned-branch handling, proven by test);
- order ambiguity (twelve sequences, two families) is authored per joint as
  `EulerConvention(sequence=..., angle_names=...)` - explicit, citable data,
  not implicit convention in someone's head.

This is a conversion utility, not a representation. If you are tempted to store,
interpolate, average, or accumulate these angles: don't - do it with the
quaternion and re-decompose for display.

A sequence is a three-letter string over ``x``/``y``/``z`` (e.g. ``"zyx"``,
``"yxy"``) read INTRINSICALLY: ``"zyx"`` means rotate about z, then about the
new y, then about the new z - equivalently the extrinsic reverse order, and
equivalently the quaternion product ``q_z · q_y · q_x``.

Two families exist:

- **Tait-Bryan** (three distinct axes, e.g. ``zyx``): all three angles have a
  geometric reading on their own axis; the middle angle hits a gimbal lock at
  ±90°.
- **Symmetric / classic Euler** (first == last, e.g. ``yxy``): the middle angle
  lives in [0, π] and locks at 0 and π.

Both families are handled exactly by conjugating the rotation matrix into a
canonical form with an axis-permutation matrix, extracting the canonical
angles there, and mapping the angles back to the sequence's own axes. The
decompose/compose pair is an exact inverse everywhere including gimbal lock
(the locked branch pins one angle to zero and folds it into its neighbour),
which is what ``test_euler_sequence.py`` proves across random quaternions,
every sequence, and forced-lock cases.
"""

from __future__ import annotations

import numpy as np

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.type_overloads import FloatArray

_VALID_AXES = frozenset("xyz")
_LOCK_EPSILON = 1e-12


def raise_unless_valid_sequence(sequence: str) -> None:
    """Raise unless `sequence` names a rotation sequence that spans SO(3).

    Three letters over x/y/z whose CONSECUTIVE letters differ - ``zyx`` (all
    distinct, the Tait-Bryan family) or ``yxy`` (first == last, the symmetric
    family). A repeated adjacent axis collapses one factor into its neighbour
    (``xxy == Rx(a+b)·Ry(c)``), leaving a two-degree-of-freedom sequence that
    cannot represent every rotation, so it is refused rather than degraded.
    """
    if (
        not isinstance(sequence, str)
        or len(sequence) != 3
        or any(axis not in _VALID_AXES for axis in sequence.lower())
    ):
        raise ValueError(
            f"an euler sequence must be three letters over x/y/z (e.g. 'zyx', "
            f"'yxy') - got {sequence!r}"
        )
    if sequence[0] == sequence[1] or sequence[1] == sequence[2]:
        raise ValueError(
            f"an euler sequence must not repeat an axis on consecutive letters - "
            f"{sequence!r} would collapse to a two-axis rotation and cannot span "
            f"all rotations. Use e.g. 'zyx' or 'yxy'."
        )


def is_symmetric_sequence(sequence: str) -> bool:
    """Whether the sequence's first and last axes are the same (classic euler)."""
    return sequence[0] == sequence[2]


def compose_euler_angles(
    *, sequence: str, angles: FloatArray
) -> RotationQuaternion:
    """Build the rotation for `angles` applied intrinsically along `sequence`.

    Args:
        sequence: three letters over x/y/z, applied first-to-last.
        angles: three radians values, one per sequence letter.

    Returns:
        The composed rotation, scalar-first quaternion semantics inside.
    """
    raise_unless_valid_sequence(sequence=sequence)
    if len(angles) != 3:
        raise ValueError(f"{sequence!r} takes three angles - got {len(angles)}")
    composed = RotationQuaternion.identity()
    for axis, angle in zip(sequence, angles):
        composed = composed * _elementary_quaternion(axis=axis, angle=float(angle))
    return composed


def decompose_euler_angles(
    *, quaternion: RotationQuaternion, sequence: str
) -> tuple[float, float, float]:
    """Extract the intrinsic euler angles of `quaternion` along `sequence`.

    The exact inverse of :func:`compose_euler_angles`: composing the returned
    angles reproduces the input rotation bit-for-bit up to float rounding,
    including at gimbal lock where one angle is pinned to zero.
    """
    raise_unless_valid_sequence(sequence=sequence)
    matrix = quaternion.to_rotation_matrix()

    if is_symmetric_sequence(sequence=sequence):
        permutation = _symmetric_permutation_matrix(first=sequence[0], middle=sequence[1])
        relabelled = permutation.T @ matrix @ permutation
        angles = _extract_symmetric_zxz_style(matrix=relabelled)
    else:
        permutation = _permutation_matrix(
            image_of={"x": sequence[0], "y": sequence[1], "z": sequence[2]}
        )
        relabelled = permutation.T @ matrix @ permutation
        angles = _extract_tait_bryan_xyz_style(matrix=relabelled)

    # An odd (mirror) permutation conjugates every elementary factor with its
    # angle negated: Q^T Rot(v, t) Q = Rot(Q^T v, det(Q) * t). Undo that so the
    # returned angles belong to the sequence's own positive senses.
    if np.linalg.det(permutation) < 0.0:
        angles = tuple(-angle for angle in angles)
    return tuple(float(angle) for angle in angles)


def angle_axis_quaternion(*, axis: str, angle: float) -> RotationQuaternion:
    """The elementary rotation of `angle` radians about a single named axis."""
    if axis not in _VALID_AXES or len(axis) != 1:
        raise ValueError(f"axis must be one of 'x', 'y', 'z' - got {axis!r}")
    return _elementary_quaternion(axis=axis, angle=angle)


def _elementary_quaternion(*, axis: str, angle: float) -> RotationQuaternion:
    half = 0.5 * angle
    components = {"x": 0.0, "y": 0.0, "z": 0.0}
    components[axis] = float(np.sin(half))
    return RotationQuaternion.from_components(
        w=float(np.cos(half)),
        x=components["x"],
        y=components["y"],
        z=components["z"],
    )


def _permutation_matrix(
    *, image_of: dict[str, str]
) -> FloatArray:
    """The axis-relabeling matrix sending each canonical letter to `image_of` it.

    Column `c` is the image of canonical basis vector e_c: conjugating a
    rotation matrix by this permutes which physical axis each canonical letter
    refers to, turning any sequence of a family into that family's canonical
    order (P^T R P applies the same rotation under the relabelled axes).
    """
    matrix = np.zeros((3, 3), dtype=np.float64)
    for canonical_index, canonical_letter in enumerate("xyz"):
        target_letter = image_of[canonical_letter]
        target_index = "xyz".index(target_letter)
        matrix[target_index, canonical_index] = 1.0
    return matrix


def _symmetric_permutation_matrix(*, first: str, middle: str) -> FloatArray:
    """The relabelling for a symmetric sequence onto the canonical ``zxz`` form.

    Canonical z plays the first/last role, canonical x the middle role, and
    canonical y takes whatever letter the sequence does not use.
    """
    remaining = sorted(_VALID_AXES - {first, middle})[0]
    return _permutation_matrix(image_of={"x": middle, "y": remaining, "z": first})


def _extract_tait_bryan_xyz_style(*, matrix: FloatArray) -> tuple[float, float, float]:
    """Angles (a, b, c) of R = Rx(a)·Ry(b)·Rz(c), from the rotation matrix.

    b = asin(R[0,2]); away from the ±90° lock, a and c come off the stable
    column/row entries. At the lock, c is pinned to zero and a absorbs it -
    the branch choice keeps compose(decompose(R)) == R exact.
    """
    sin_b = float(np.clip(matrix[0, 2], -1.0, 1.0))
    b = float(np.arcsin(sin_b))
    if abs(sin_b) < 1.0 - _LOCK_EPSILON:
        a = float(np.arctan2(-matrix[1, 2], matrix[2, 2]))
        c = float(np.arctan2(-matrix[0, 1], matrix[0, 0]))
        return a, b, c
    # Gimbal lock: only the combination a ∓ c is observable. Pin c to zero.
    c = 0.0
    sign = 1.0 if sin_b > 0.0 else -1.0
    a = float(np.arctan2(sign * matrix[1, 0], matrix[1, 1]))
    return a, b, c


def _extract_symmetric_zxz_style(*, matrix: FloatArray) -> tuple[float, float, float]:
    """Angles (a, b, c) of R = Ez(a)·Ex(b)·Ez(c), from the rotation matrix.

    Read straight off the product's entries: R[2,2] = cos(b); R[0,2] = sin(a)
    sin(b) and R[1,2] = -cos(a) sin(b) give a; R[2,0] = sin(b) cos(c) and
    R[2,1] = sin(b) cos(a) give c. At either lock (b = 0 or π) only the sum /
    difference a ± c is observable - c is pinned to zero and a absorbs it.
    """
    cos_b = float(np.clip(matrix[2, 2], -1.0, 1.0))
    b = float(np.arccos(cos_b))
    if abs(cos_b) < 1.0 - _LOCK_EPSILON:
        a = float(np.arctan2(matrix[0, 2], -matrix[1, 2]))
        c = float(np.arctan2(matrix[2, 0], matrix[2, 1]))
        return a, b, c
    c = 0.0
    if cos_b > 0.0:
        # b = 0: R is Ez(a + c); its upper-left 2x2 block is the rotation by a.
        a = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    else:
        # b = pi: R encodes a - c in its upper-left block.
        a = float(np.arctan2(matrix[0, 1], matrix[0, 0]))
    return a, b, c
