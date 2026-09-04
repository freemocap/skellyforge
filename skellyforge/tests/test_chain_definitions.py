"""The chain layer's static face: declared paths over the skeleton's joints."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from skellyforge.core.skeleton.chain import KinematicChain
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition

SHIPPED_DEFINITIONS_DIR: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
)

def _skeleton() -> SkeletonDefinition:
    return SkeletonDefinition.from_default_yaml()


def test_the_shipped_chains_load_and_are_object_referenced() -> None:
    chains = _skeleton().chains
    for chain in chains.values():
        assert len(chain.segments) >= 3
        assert len(chain.joints) == len(chain.segments) - 1
        for segment in chain.segments:
            # Object references, not strings: every resolved segment is real.
            assert segment.name in _skeleton().segments


def test_chain_joints_connect_consecutive_segments() -> None:
    for chain in _skeleton().chains.values():
        for joint, parent, child in zip(chain.joints, chain.segments, chain.segments[1:]):
            assert joint.parent.name == parent.name
            assert joint.child.name == child.name


def test_a_chain_missing_its_middle_link_is_rejected() -> None:
    with pytest.raises(ValueError, match="no joint joins them"):
        KinematicChain.from_yaml_entry(
            name="broken",
            segment_names=["pelvis", "left_upper_arm", "left_lower_arm"],            joints=_skeleton().joints,
        )


def test_an_unknown_segment_is_rejected() -> None:
    with pytest.raises(ValueError, match="named by no joint"):
        KinematicChain.from_yaml_entry(
            name="ghost",
            segment_names=["pelvis", "lumbar_spine", "tail_bone"],
            joints=_skeleton().joints,
        )


def test_a_two_segment_chain_is_rejected() -> None:
    """A chain is 3+ segments; two segments are one joint, i.e. the linkage layer."""
    with pytest.raises(ValueError):
        KinematicChain.from_yaml_entry(
            name="too_short",
            segment_names=["left_upper_leg", "left_lower_leg", "left_foot"][:2],
            joints=_skeleton().joints,
        )


def test_a_repeated_segment_is_rejected() -> None:
    with pytest.raises(ValueError, match="appears twice"):
        KinematicChain.from_yaml_entry(
            name="loop",
            segment_names=["pelvis", "sacrolumbar", "thoracic", "thoracic"],
            joints=_skeleton().joints,
        )


def test_chains_section_may_be_absent(tmp_path: Path) -> None:
    """A skeleton without declared chains loads fine - they are optional authoring."""
    target = tmp_path / "human_skeleton"
    shutil.copytree(SHIPPED_DEFINITIONS_DIR, target)
    # The shipped human declares its chains compositionally, in the components that
    # walk them; strip those AND the top-level section (if present) to author a
    # genuinely chain-less skeleton.
    for component_path in sorted((target / "components").glob("*.yaml")):
        component = yaml.safe_load(component_path.read_text(encoding="utf-8")) or {}
        if "chains" in component:
            del component["chains"]
            component_path.write_text(yaml.safe_dump(component), encoding="utf-8")
    skeleton_path = target / "human_skeleton.yaml"
    document = yaml.safe_load(skeleton_path.read_text(encoding="utf-8"))
    document.pop("chains", None)
    skeleton_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    skeleton = SkeletonDefinition.from_yaml(path=skeleton_path)
    assert skeleton.chains == {}
