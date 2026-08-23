"""The YAML loader pipeline: include -> lowercase -> side -> reference frame -> objects."""
from skellyforge.core.skeleton.loading.component_building import (
    build_component,
    load_component,
)
from skellyforge.core.skeleton.loading.include_resolution import resolve_includes
from skellyforge.core.skeleton.loading.name_lowercasing import lowercase_names
from skellyforge.core.skeleton.loading.reference_frame_building import (
    build_reference_frame_definition,
)
from skellyforge.core.skeleton.loading.sided_expansion import expand_sided_entries

__all__ = [
    "build_component",
    "build_reference_frame_definition",
    "expand_sided_entries",
    "load_component",
    "lowercase_names",
    "resolve_includes",
]
