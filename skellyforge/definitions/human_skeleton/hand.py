"""Standalone hands drawn from the authored human skeleton and T-pose."""
from dataclasses import replace

from skellyforge.core.biomechanics.center_of_mass import CenterOfMassDefinitions
from skellyforge.core.skeleton.pose.rest_pose import RestPose, build_rest_pose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


def human_hand_model(*, human: SkeletonDefinition, rest: RestPose, side: str) -> tuple[SkeletonDefinition, RestPose]:
    if side not in ("left", "right"):
        raise ValueError(f"Unknown hand side {side!r}")
    segments = {name: segment for name, segment in human.segments.items()
                if segment.anatomical_segment == "hand" and name.startswith(f"{side}_")}
    root = f"{side}_carpals"
    skeleton = SkeletonDefinition(
        name=f"{side}_hand", segments=segments,
        landmarks={name: landmark for segment in segments.values() for name, landmark in segment.landmarks.items()},
        joints={name: joint for name, joint in human.joints.items()
                if joint.parent.name in segments and joint.child.name in segments},
        chains={name: chain for name, chain in human.chains.items()
                if all(segment.name in segments for segment in chain.segments)},
        coordinate_system=human.coordinate_system, derived_quantities=frozenset({"roll_resolution"}),
    )
    parents = {name: None if name == root else rest.parents[name] for name in segments}
    attachments = {name: rest.connect_ats[name] for name in segments}
    attachments[root] = segments[root].frame_definition.origin_point_name
    rotations = {name: rest.relative_orientations[name] for name in segments}
    rotations[root] = rest.segment_orientations[root]
    orientations, origins, landmarks = build_rest_pose(
        skeleton=skeleton, parents=parents, connect_ats=attachments, orientations=rotations)
    return skeleton, RestPose(
        name=rest.name, root_segment_name=root, parents=parents, connect_ats=attachments,
        relative_orientations=rotations, segment_orientations=orientations,
        segment_origins=origins, landmark_positions=landmarks,
    )


def hand_center_of_mass(*, side: str, definitions: CenterOfMassDefinitions) -> CenterOfMassDefinitions:
    definition = definitions.get(name="hand")

    def sided(landmark):
        return definition.resolve_landmark(side=side, landmark=landmark)

    hand = replace(definition, name=f"{side}_hand", sided=False,
                   proximal=sided(definition.proximal), distal=sided(definition.distal),
                   weights=tuple(replace(weight, landmark=sided(weight.landmark)) for weight in definition.weights))
    return CenterOfMassDefinitions(definitions={hand.name: hand})
