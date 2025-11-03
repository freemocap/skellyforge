from skellyforge.post_processing.filters.filter_config import FilterMethod
from skellyforge.post_processing.filters.core.butter import butter_from_config
from collections.abc import Callable

FILTER_REGISTRY: dict[FilterMethod, Callable] = {
    FilterMethod.butter_low_pass: butter_from_config
}