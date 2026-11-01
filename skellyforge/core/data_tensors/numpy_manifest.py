"""Versioned metadata for the NumPy dataset format."""
from dataclasses import asdict
import json
import math
import re

import numpy as np

from skellyforge.core.data_tensors.data_tensor import DataTensor, DataProvenance
from skellyforge.core.data_tensors.layout_definitions import AxisDefinition, SourceIdentity, TensorLayout
from skellyforge.core.data_tensors.recording_dataset import DefinitionDocument

FORMAT_NAME = "skellyforge.numpy_dataset"
FORMAT_VERSION = 2
DATA_COMPANIONS = frozenset(("entry_presence", "value_validity", "confidence_scores",
                                  "measured_support", "voting_support"))
ROOT_FIELDS = {"format", "version", "recording_id", "run_id", "definition_documents", "timings", "channels"}
SOURCE_FIELDS = {"source_id", "instance_id", "sensor_group", "camera_id", "model_id"}
LAYOUT_FIELDS = {"axes", "dtype", "component_units", "reference_id", "source_identity", "channel_kind"}
TIMING_FIELDS = {"timing_key", "timeline_id", "clock_id", "clock_epoch", "source_identity", "sample_ids_file", "timestamps_s_file"}
CHANNEL_FIELDS = {"storage_kind", "values_file", "tensor_layout", "data_provenance", "companion_tensors", "sample_axis_name", "timing_ref"}
PROVENANCE_FIELDS = {"meaning", "operation_id", "input_refs", "model_id"}


def json_object(value, fields):
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Expected exact manifest fields {sorted(fields)}")
    return value


def json_list(value):
    if not isinstance(value, list):
        raise ValueError("Manifest sequence must be a JSON list")
    return value


def text_value(value, *, nullable=False):
    if nullable and value is None:
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Manifest identity must be a nonempty string")
    return value


def integer_value(value):
    if type(value) is not int or value < 0:
        raise ValueError("Manifest integer must be nonnegative and not boolean")
    return value


def array_filename(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_]+\.npy", value):
        raise ValueError("Array filename must be a safe flat NPY basename")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        _reject_constant(value)
    return number


def parse_json(content):
    return json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_constant, parse_float=_finite_float)


def source_data(source):
    return None if source is None else asdict(source)


def source_from_data(value, *, nullable=False):
    if nullable and value is None:
        return None
    value = json_object(value, SOURCE_FIELDS)
    text_value(value["source_id"])
    integer_value(value["instance_id"])
    for name in ("sensor_group", "camera_id", "model_id"):
        text_value(value[name], nullable=True)
    return SourceIdentity(**value)


def definition_data(documents):
    result, identities = [], set()
    for document in documents:
        if not isinstance(document, DefinitionDocument):
            raise TypeError("Definition documents must be DefinitionDocument instances")
        if document.definition_id in identities:
            raise ValueError("Duplicate definition identity")
        parse_json(document.content_json)
        identities.add(document.definition_id)
        result.append(asdict(document))
    return result


def definitions_from_data(values):
    documents = []
    for item in json_list(values):
        item = json_object(item, {"definition_id", "content_json"})
        text_value(item["definition_id"])
        text_value(item["content_json"])
        parse_json(item["content_json"])
        documents.append(DefinitionDocument(**item))
    definition_data(documents)
    return tuple(documents)


def layout_data(layout, *, timing_key=None, sample_axis_name=None):
    axes = []
    for axis in layout.axes:
        entry = {"axis_name": axis.axis_name}
        if axis.axis_name == sample_axis_name:
            entry["timing_ref"] = timing_key
        else:
            entry["identifiers"] = list(axis.identifiers)
        axes.append(entry)
    return dict(axes=axes, dtype=layout.dtype.str, component_units=list(layout.component_units),
                reference_id=layout.reference_id, source_identity=source_data(layout.source_identity),
                channel_kind=layout.channel_kind)


def layout_from_data(value, *, timings, timing_key=None, sample_axis_name=None):
    value = json_object(value, LAYOUT_FIELDS)
    axes = []
    for axis in json_list(value["axes"]):
        if not isinstance(axis, dict):
            raise ValueError("Invalid manifest axis")
        name = text_value(axis.get("axis_name"))
        if name == sample_axis_name:
            json_object(axis, {"axis_name", "timing_ref"})
            if axis["timing_ref"] != timing_key or timing_key not in timings:
                raise ValueError("Axis timing reference differs from channel timing")
            identifiers = timings[timing_key].sample_ids
        else:
            json_object(axis, {"axis_name", "identifiers"})
            identifiers = json_list(axis["identifiers"])
            if any(type(item) not in (int, str) for item in identifiers):
                raise ValueError("Axis identifiers must be integers or strings")
        axes.append(AxisDefinition(name, identifiers))
    dtype = np.dtype(text_value(value["dtype"]))
    if dtype.str != value["dtype"]:
        raise ValueError("Manifest dtype must use canonical scalar dtype.str")
    units = tuple(text_value(item) for item in json_list(value["component_units"]))
    return TensorLayout(tuple(axes), dtype, units,
                        reference_id=text_value(value["reference_id"], nullable=True),
                        source_identity=source_from_data(value["source_identity"], nullable=True),
                        channel_kind=text_value(value["channel_kind"], nullable=True))


def provenance_from_data(value):
    value = json_object(value, PROVENANCE_FIELDS)
    return DataProvenance(text_value(value["meaning"]), text_value(value["operation_id"]),
                                tuple(text_value(item) for item in json_list(value["input_refs"])),
                                text_value(value["model_id"], nullable=True))


def validate_data_tensor(tensor):
    if not isinstance(tensor, DataTensor):
        raise TypeError("Channels must contain DataTensor instances")
    # Borrowed inputs may have changed since construction; validate before writing.
    tensor.__post_init__()
    for companion in tensor.companion_tensors:
        if companion.metadata_name not in DATA_COMPANIONS:
            raise ValueError(f"Unsupported companion {companion.metadata_name!r}")
        companion.tensor_layout.validate_array(companion.data_values)
        expected = "f" if companion.metadata_name == "confidence_scores" else "b"
        if companion.data_values.dtype.kind != expected:
            raise ValueError(f"Incorrect companion dtype for {companion.metadata_name!r}")


def manifest_inventory(manifest):
    """Validate the outer schema and collect unique payload filenames before IO."""
    json_object(manifest, ROOT_FIELDS)
    if manifest["format"] != FORMAT_NAME or type(manifest["version"]) is not int or manifest["version"] not in (1, FORMAT_VERSION):
        raise ValueError("Unsupported NumPy dataset format/version")
    text_value(manifest["recording_id"])
    integer_value(manifest["run_id"])
    definitions_from_data(manifest["definition_documents"])
    filenames = set()
    def retain(name):
        name = array_filename(name)
        if name in filenames:
            raise ValueError("Payload filename is referenced more than once")
        filenames.add(name)
    for timing in json_list(manifest["timings"]):
        json_object(timing, TIMING_FIELDS)
        retain(timing["sample_ids_file"])
        retain(timing["timestamps_s_file"])
    static_started = False
    for channel in json_list(manifest["channels"]):
        json_object(channel, (CHANNEL_FIELDS - {"data_provenance"}) | {"scientific_provenance"}
                    if manifest["version"] == 1 else CHANNEL_FIELDS)
        kind = channel["storage_kind"]
        if kind not in ("time_series", "static_tensor"):
            raise ValueError("Unknown channel storage kind")
        if kind == "time_series" and static_started:
            raise ValueError("Time series must precede static tensors")
        static_started |= kind == "static_tensor"
        retain(channel["values_file"])
        for companion in json_list(channel["companion_tensors"]):
            json_object(companion, {"metadata_name", "values_file", "tensor_layout"})
            retain(companion["values_file"])
    return filenames | {"dataset.json"}
