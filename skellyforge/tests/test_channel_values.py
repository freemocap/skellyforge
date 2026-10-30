"""Dense named access retains declared components, absence and buffer ownership."""

import numpy as np
import pytest

from skellyforge.core.data_tensors import ChannelValues


def test_named_reads_borrow_declared_components_and_reject_absent_entities():
    components = ("first", "second", "third", "fourth", "fifth")
    values = np.arange(15, dtype=np.float64).reshape(3, len(components))
    present = np.array([True, False, True])
    channel = ChannelValues(("b", "a", "c"), values, present, component_names=components)
    assert channel.data_values is values
    assert channel.entry_present is present
    assert np.shares_memory(channel["b"], values)
    assert tuple(channel) == ("b", "c")
    assert "a" not in channel
    with pytest.raises(KeyError):
        channel["a"]
    aligned = channel.aligned(("c", "unknown", "a", "b"))
    np.testing.assert_array_equal(aligned[[0, 3]], values[[2, 0]])
    assert np.isnan(aligned[1:3]).all()
    np.testing.assert_array_equal(channel.aligned_presence(("c", "unknown", "a", "b")),
                                  [True, False, False, True])


def test_publication_owns_values_and_presence_and_same_layout_admission_reuses_object():
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    presence = np.ones(2, dtype=bool)
    channel = ChannelValues(("a", "b"), values, presence, component_names=("u", "v", "w"))
    assert channel.aligned(channel.names) is values
    assert ChannelValues.from_mapping(channel, component_names=channel.component_names) is channel
    owned = channel.immutable_copy()
    values[:] = -1
    presence[:] = False
    np.testing.assert_array_equal(owned.data_values, np.arange(6).reshape(2, 3))
    assert owned.entry_present.all()
    for array in (owned.data_values, owned.entry_present):
        with pytest.raises(ValueError):
            array.flags.writeable = True


def test_layout_mismatch_is_not_repaired_and_empty_scalar_channels_are_valid():
    with pytest.raises(ValueError, match="shape"):
        ChannelValues(("a",), np.zeros((1, 3)), component_names=("u", "v"))
    channel = ChannelValues.empty(component_names=())
    assert channel.data_values.shape == (0,)
    assert channel.aligned(("missing",)).shape == (1,)
    assert np.isnan(channel.aligned(("missing",))).all()
