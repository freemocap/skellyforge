"""Command-line entry point: load the standard human and print what it contains.

Doubles as the package's smoke test - if the definitions, the loader and the rest pose
all work, this prints a summary; if any of them are broken, it raises here rather than
somewhere downstream.
"""

from __future__ import annotations

from pathlib import Path

from skellyforge.core.skeleton.components.face_blendshapes import FaceBlendShapes
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition

DEFINITIONS_DIRECTORY: Path = (
    Path(__file__).resolve().parent / "definitions" / "human_skeleton"
)
SKELETON_YAML_PATH: Path = DEFINITIONS_DIRECTORY / "human_skeleton.yaml"
REST_POSE_YAML_PATH: Path = DEFINITIONS_DIRECTORY / "rest_pose.yaml"
FACE_YAML_PATH: Path = DEFINITIONS_DIRECTORY / "components" / "face.yaml"


def run() -> None:
    """Load the shipped standard human and print a one-screen summary of it."""
    skeleton = SkeletonDefinition.from_yaml(path=SKELETON_YAML_PATH)
    rest_pose = RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=skeleton)
    face = FaceBlendShapes.from_yaml(path=FACE_YAML_PATH)

    underspecified = skeleton.underspecified_segment_names
    print(f"skeleton `{skeleton.name}`")
    print(f"  segments            {len(skeleton.segments)}")
    print(f"  landmarks           {len(skeleton.landmarks)}")
    print(f"  face blendshapes    {len(face)}")
    print(
        f"  fully specified     {len(skeleton.segments) - len(underspecified)} "
        f"(roll pinned by a secondary landmark)"
    )
    print(f"  underspecified      {len(underspecified)} (roll resolved downstream)")
    print(f"rest pose `{rest_pose.name}`")
    print(f"  root segment        {rest_pose.root_segment_name}")
    print(f"  resolved landmarks  {len(rest_pose.landmark_positions)}")


if __name__ == "__main__":
    run()
