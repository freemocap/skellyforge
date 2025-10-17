
from skellyforge.post_processing.interpolation.interpolation_config import InterpolationMethod
from skellyforge.post_processing.interpolation.core.linear_interp import linear_interpolate
from collections.abc import Callable

INTERPOLATION_REGISTRY: dict[InterpolationMethod, Callable] = {
    InterpolationMethod.linear: linear_interpolate
}