"""Dense model-scale evidence and fitted values with prepared anatomical groups."""
from dataclasses import dataclass

import numpy as np

from skellyforge.core.skeleton.loading.sided_expansion import unsided_name_of


@dataclass(frozen=True, slots=True)
class ScaleTensor:
    """One fit over named segments; arrays retain input scale/spatial units.

    Construction validates borrowed arrays. ScaleGeometry creates owned result
    arrays, so subsequent observations cannot mutate a previously returned fit.
    Support distinguishes observed evidence from model-proportion predictions.
    """
    segment_names: tuple[str, ...]
    fitted_scale: float
    segment_scales: np.ndarray
    segment_lengths: np.ndarray
    measured_support: np.ndarray
    voting_support: np.ndarray

    def __post_init__(self):
        if (not isinstance(self.segment_names, tuple) or not self.segment_names or len(set(self.segment_names)) != len(self.segment_names)
                or any(not isinstance(name, str) or not name for name in self.segment_names)):
            raise ValueError("Scale segment identities must be unique nonempty names")
        if not np.isfinite(self.fitted_scale) or self.fitted_scale <= 0:
            raise ValueError("Fitted scale must be finite and positive")
        shape = (len(self.segment_names),)
        for values in (self.segment_scales, self.segment_lengths):
            if values.dtype != np.float64 or values.shape != shape or not np.isfinite(values).all():
                raise ValueError("Scale values must be finite float64 on the segment axis")
        if np.any(self.segment_scales <= 0) or np.any(self.segment_lengths < 0):
            raise ValueError("Scales must be positive and lengths nonnegative")
        for support in (self.measured_support, self.voting_support):
            if support.dtype != bool or support.shape != shape:
                raise ValueError("Scale support must be boolean on the segment axis")
        if np.any(self.voting_support & ~self.measured_support):
            raise ValueError("Scale votes require measured evidence")

    def to_fit(self):
        """Adapt existing model-fit consumers at the explicit dictionary boundary."""
        from skellyforge.core.skeleton.pose.model_scale_fitting import ModelScaleFit
        return ModelScaleFit(fitted_scale=self.fitted_scale,
            segment_scales=dict(zip(self.segment_names, self.segment_scales, strict=True)),
            segment_lengths=dict(zip(self.segment_names, self.segment_lengths, strict=True)),
            measured_segment_names=frozenset(name for name, selected in zip(self.segment_names, self.measured_support) if selected),
            voting_segment_names=frozenset(name for name, selected in zip(self.segment_names, self.voting_support) if selected))


class ScaleGeometry:
    """Prepare authored proportions and sided voting groups once, fit dense evidence."""
    def __init__(self, *, skeleton, voting_segment_names):
        self.segment_names = tuple(skeleton.segments)
        indices = {name: index for index, name in enumerate(self.segment_names)}
        if not set(voting_segment_names).issubset(indices):
            raise KeyError("Voting segment identities must belong to the skeleton")
        self.proportions = np.array([segment.length for segment in skeleton.segments.values()], dtype=np.float64)
        self.voting_mask = np.array([name in voting_segment_names for name in self.segment_names], bool)
        groups = {}
        for name in sorted(voting_segment_names):
            groups.setdefault(unsided_name_of(name=name), []).append(indices[name])
        self.group_columns = np.full((len(groups), max(map(len, groups.values()), default=0)), -1, np.int64)
        for row, columns in enumerate(groups.values()):
            self.group_columns[row, :len(columns)] = columns

    def fit(self, *, scales: np.ndarray, segment_present: np.ndarray) -> ScaleTensor:
        from skellyforge.core.skeleton.pose.model_scale_fitting import (
            InsufficientScaleEvidence, SHRINKAGE_PRIOR_SAMPLES, DISPERSION_HALF_TRUST, _weighted_median,
        )
        values, present = np.asarray(scales), np.asarray(segment_present)
        if (values.dtype != np.float64 or values.ndim != 2 or values.shape[1] != len(self.segment_names)
                or present.dtype != bool or present.shape != values.shape):
            raise ValueError("Scale evidence requires float64 values and boolean presence on sample/segment axes")
        if not np.isfinite(values[present]).all() or np.any(values[present] <= 0):
            raise ValueError("Present scale evidence must be finite and positive")
        counts = present.sum(axis=0)
        measured = counts > 0
        voting = measured & self.voting_mask
        if not voting.any():
            raise InsufficientScaleEvidence("no segment entitled to vote has scale evidence")
        columns = np.flatnonzero(measured)
        selected = np.where(present[:, columns], values[:, columns], np.nan)
        medians = np.zeros(len(self.segment_names), np.float64)
        medians[columns] = np.nanmedian(selected, axis=0)
        dispersions = np.zeros_like(medians)
        dispersions[columns] = np.nanmedian(np.abs(selected - medians[columns]), axis=0) / medians[columns]
        lookup = np.maximum(self.group_columns, 0)
        group_present = (self.group_columns >= 0) & measured[lookup]
        active = group_present.any(axis=1)
        lookup, group_present = lookup[active], group_present[active]
        grouped_values = np.nanmedian(np.where(group_present, medians[lookup], np.nan), axis=1)
        grouped_counts = np.sum(np.where(group_present, counts[lookup], 0), axis=1)
        grouped_spreads = np.max(np.where(group_present, dispersions[lookup], -np.inf), axis=1)
        grouped_proportions = np.sum(np.where(group_present, self.proportions[lookup], 0.), axis=1) / group_present.sum(axis=1)
        weights = grouped_proportions**2 * grouped_counts / (1. + grouped_spreads / DISPERSION_HALF_TRUST)
        fitted = _weighted_median(values=grouped_values, weights=weights)
        prior = SHRINKAGE_PRIOR_SAMPLES * (1. + dispersions / DISPERSION_HALF_TRUST)
        share = counts / (counts + prior)
        fitted_values = share * medians + (1. - share) * fitted
        return ScaleTensor(self.segment_names, fitted, fitted_values, self.proportions * fitted_values, measured, voting)
