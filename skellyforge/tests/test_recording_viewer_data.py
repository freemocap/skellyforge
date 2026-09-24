"""Prepared data discovery and scalar-to-landmark reconstruction contracts."""

import numpy as np
import pytest

from scripts.recording_data import decode_rows, recording_path


def test_fitted_parent_does_not_rotate_unfitted_descendants_again():
    from scripts.generate_real_skeleton_viewer import combine_fitted_world_rotations
    from skellyforge.core.math.geometry.rotation_quaternion import (
        RotationQuaternion as Q,
    )

    parents = {
        "pelvis": None,
        "chest": "pelvis",
        "clavicle": "chest",
        "arm": "clavicle",
        "hand": "arm",
        "head": "chest",
    }
    world = {
        n: Q.from_rotation_vector(
            rotation_vector=np.array([0.13 * i, -0.07 * i, 0.11 * i])
        )
        for i, n in enumerate(parents)
    }
    fitted = {
        n: Q.from_rotation_vector(rotation_vector=np.array([0.4, -0.2, 0.6])) * world[n]
        for n in ("chest", "clavicle")
    }
    local = combine_fitted_world_rotations(
        parents=parents, reference_world=world, fitted_world=fitted
    )
    rebuilt = {"pelvis": world["pelvis"]}
    for n, parent in parents.items():
        if parent is not None:
            rebuilt[n] = rebuilt[parent] * local[n]
    for n in parents:
        assert rebuilt[n].is_same_rotation(other=fitted.get(n, world[n]))


def rows():
    return [
        dict(
            frame_number=f,
            timestamp_s=t,
            name="point",
            component=c,
            value=v,
            units="mm",
        )
        for f, t in [(2, 1.2), (5, 1.9)]
        for c, v in zip("xyz", [1.0, 2.0, 3.0])
    ]


def test_preserves_frame_numbers_timestamps_and_omits_incomplete_points():
    samples = rows()
    samples[-1]["value"] = None
    result = decode_rows(samples, ["point"])
    assert [f["number"] for f in result] == [2, 5]
    assert [f["time"] for f in result] == [1.2, 1.9]
    np.testing.assert_array_equal(result[0]["points"]["point"], [1, 2, 3])
    assert result[1]["points"] == {}


@pytest.mark.parametrize("bad", ["duplicate", "units", "time", "component"])
def test_rejects_ambiguous_or_incompatible_samples(bad):
    samples = rows()
    if bad == "duplicate":
        samples.append(samples[0])
    if bad == "units":
        samples[0]["units"] = "px"
    if bad == "time":
        samples[0]["timestamp_s"] = 2.0
    if bad == "component":
        samples[0]["component"] = "w"
    with pytest.raises(ValueError):
        decode_rows(samples, ["point"])


def test_prefers_prepared_recording_without_creating_or_downloading_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="Prepare it in FreeMoCap"):
        recording_path(home=tmp_path)
    name = "freemocap_test_data"
    canonical = (
        tmp_path / "freemocap_data" / "recordings" / name / f"{name}_data.parquet"
    )
    canonical.parent.mkdir(parents=True)
    canonical.touch()
    assert recording_path(home=tmp_path) == canonical
    prepared = (
        tmp_path
        / "freemocap_data"
        / "testing"
        / "prepared"
        / name
        / "current"
        / "recordings"
        / name
        / f"{name}_data.parquet"
    )
    prepared.parent.mkdir(parents=True)
    prepared.touch()
    assert recording_path(home=tmp_path) == prepared
