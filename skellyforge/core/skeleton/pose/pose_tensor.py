"""Named numerical poses shared by one-frame and recording operations.

Sample positions are ordered inputs, not a newly invented clock. Coordinates and
scale retain the caller's spatial units. Ownership is borrowed until copy_owned
is requested; sample_view borrows the corresponding slice without reordering.
"""
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import UNIT_QUATERNION_TOLERANCE
from skellyforge.core.skeleton.skeleton_pose import PoseSolution, SkeletonPose, SegmentPose
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point

POSITION_COMPONENTS = ("x", "y", "z")
ROTATION_COMPONENTS = ("w", "x", "y", "z")


class SolutionCode(IntEnum):
    """Numerical provenance; transported roll remains a model convention."""
    MISSING = 0
    RIGID_FIT = 1
    OBSERVATION_FRAME = 2
    DIRECTION = 3
    TRANSPORTED_ROLL = 4


class HydrationStatus(IntEnum):
    MISSING = 0
    VALID = 1
    DEGENERATE = 2


_SOLUTION_MEMBERS = (None, PoseSolution.RIGID_FIT, PoseSolution.OBSERVATION_FRAME,
                     PoseSolution.DIRECTION, PoseSolution.TRANSPORTED_ROLL)


def encode_solutions(values: np.ndarray) -> np.ndarray:
    """Adapt the existing object API at its boundary, never in tensor calculation."""
    values = np.asarray(values)
    result = np.zeros(values.shape, dtype=np.uint8)
    known = values == None  # noqa: E711 - elementwise object-array comparison
    for code, member in enumerate(_SOLUTION_MEMBERS[1:], start=1):
        selected = values == member
        result[selected] = code
        known |= selected
    if not known.all():
        raise ValueError("Unknown pose solution provenance")
    return result


def decode_solutions(codes: np.ndarray) -> np.ndarray:
    """Materialize enum objects only for callers of the existing object API."""
    codes = np.asarray(codes)
    if codes.dtype.kind not in "iu" or np.any(codes < 0) or np.any(codes >= len(_SOLUTION_MEMBERS)):
        raise ValueError("Unknown pose solution code")
    return np.asarray(_SOLUTION_MEMBERS, dtype=object)[codes]


@dataclass(frozen=True, slots=True)
class PoseTensor:
    """Dense sample/segment arrays with explicit authored segment identities.

    origins and orientations add their declared component axes. hydration_status
    describes recovery (missing/valid/degenerate); solution_codes describes the
    current orientation's provenance, including subsequent temporal roll resolution.
    Missing/degenerate segments carry NaN values and MISSING solution codes.
    Constructors borrow arrays: callers must preserve their declared shapes and
    validity. Use copy_owned for independent storage before retaining shared data.
    """
    segment_names: tuple[str, ...]
    origins: np.ndarray
    orientations: np.ndarray
    scale_estimates: np.ndarray
    solution_codes: np.ndarray
    hydration_status: np.ndarray
    _immutable: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self):
        if (not isinstance(self.segment_names, tuple) or not self.segment_names
                or any(not isinstance(name, str) or not name for name in self.segment_names)
                or len(set(self.segment_names)) != len(self.segment_names)):
            raise ValueError("Pose segment identities must be unique nonempty names")
        if self.scale_estimates.ndim != 2:
            raise ValueError("Pose scales require sample and segment axes")
        shape = self.scale_estimates.shape
        if shape[1] != len(self.segment_names):
            raise ValueError("Pose segment axis disagrees with declared identities")
        for values, expected in ((self.origins, (*shape, len(POSITION_COMPONENTS))),
                                 (self.orientations, (*shape, len(ROTATION_COMPONENTS))),
                                 (self.scale_estimates, shape)):
            if values.shape != expected or values.dtype != np.float64:
                raise ValueError("Pose values must be float64 with declared axes")
        for values, maximum in ((self.solution_codes, int(SolutionCode.TRANSPORTED_ROLL)),
                                (self.hydration_status, int(HydrationStatus.DEGENERATE))):
            if values.shape != shape or values.dtype != np.uint8 or np.any(values > maximum):
                raise ValueError("Pose metadata must be declared uint8 codes on sample/segment axes")
        present = self.segment_present
        if np.any((self.solution_codes != SolutionCode.MISSING) != present):
            raise ValueError("Pose solutions must describe exactly the valid segments")
        for values in (self.origins, self.orientations, self.scale_estimates):
            if not np.isfinite(values[present]).all() or not np.isnan(values[~present]).all():
                raise ValueError("Valid pose values must be finite; unavailable values must be NaN")
        if np.any(self.scale_estimates[present] <= 0):
            raise ValueError("Pose scales must be positive")
        if np.any(np.abs(np.linalg.norm(self.orientations[present], axis=-1) - 1) > UNIT_QUATERNION_TOLERANCE):
            raise ValueError("Pose orientations must be unit wxyz quaternions")

    @property
    def segment_present(self) -> np.ndarray:
        return self.hydration_status == HydrationStatus.VALID

    @property
    def sample_count(self) -> int:
        return self.scale_estimates.shape[0]

    @property
    def is_immutable(self) -> bool:
        return self._immutable

    def sample_view(self, selection: slice) -> "PoseTensor":
        """Borrow an ordered sample slice; retain the sample axis even for one frame."""
        if not isinstance(selection, slice) or (selection.step is not None and selection.step < 1):
            raise ValueError("Pose sample selection must be a forward slice")
        selected = tuple(values[selection] for values in self.arrays())
        if not self._immutable:
            return PoseTensor(self.segment_names, *selected)
        # This instance and its bytes-backed arrays were validated at publication.
        # A forward sample view cannot change identities, dtype, or value validity.
        result = object.__new__(PoseTensor)
        object.__setattr__(result, "segment_names", self.segment_names)
        for name, values in zip(("origins", "orientations", "scale_estimates", "solution_codes", "hydration_status"), selected, strict=True):
            object.__setattr__(result, name, values)
        object.__setattr__(result, "_immutable", True)
        return result

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (self.origins, self.orientations, self.scale_estimates, self.solution_codes, self.hydration_status)

    def copy_owned(self, *, writable: bool = False) -> "PoseTensor":
        """Copy into independent storage; default publication uses immutable bytes."""
        if self._immutable and not writable:
            return self
        def copy(values):
            if writable:
                return values.copy()
            return np.frombuffer(values.tobytes(order="C"), dtype=values.dtype).reshape(values.shape)
        result = PoseTensor(self.segment_names, *(copy(values) for values in self.arrays()))
        object.__setattr__(result, "_immutable", not writable)
        return result

    @classmethod
    def from_pose(cls, *, pose: SkeletonPose, segment_names: tuple[str, ...]) -> "PoseTensor":
        """Explicit adapter for an existing one-frame object caller."""
        index = {name: column for column, name in enumerate(segment_names)}
        if not set(pose.segment_poses).issubset(index):
            raise ValueError("Pose contains unknown segments")
        shape = (1, len(segment_names))
        origins = np.full((*shape, len(POSITION_COMPONENTS)), np.nan)
        orientations = np.full((*shape, len(ROTATION_COMPONENTS)), np.nan)
        scales = np.full(shape, np.nan)
        codes, status = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
        for name, segment in pose.segment_poses.items():
            column = index[name]
            origins[0, column] = segment.origin.array
            orientations[0, column] = segment.orientation.as_array()
            scales[0, column] = segment.scale_estimate
            codes[0, column] = _SOLUTION_MEMBERS.index(segment.solved_by)
            status[0, column] = HydrationStatus.VALID
        return cls(segment_names, origins, orientations, scales, codes, status)

    def to_pose(self, sample_position: int = 0) -> SkeletonPose:
        """Explicit adapter for existing object consumers, outside the numeric path."""
        present = self.segment_present[sample_position]
        return SkeletonPose(segment_poses={self.segment_names[column]: SegmentPose(
            segment_name=self.segment_names[column],
            origin=Point.from_array(values=self.origins[sample_position, column]),
            orientation=RotationQuaternion.from_array(array=self.orientations[sample_position, column]),
            scale_estimate=float(self.scale_estimates[sample_position, column]),
            solved_by=_SOLUTION_MEMBERS[self.solution_codes[sample_position, column]])
            for column in np.flatnonzero(present)})
