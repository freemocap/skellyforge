"""Export the standard-human REST POSE armature to human-readable JSON.

A debugging artifact: dumps every segment as a bone (head + tail + parent +
length + rest basis) plus every landmark's rest-world position, so the assembled
T-pose can be loaded into Blender and eyeballed for errors (bone placement,
lengths, left/right mirroring, palms-down hands, the toe fan, ...).

Source of truth is ONLY the current model: ``HumanSkeleton.standard_human()`` +
``build_standard_human_tpose``. Coordinates are the model's own -- millimetres,
+Z up, +X forward (anterior), +Y = the subject's left. The companion
``rest_pose_blender_importer.py`` reads this file inside Blender.

Run it:
    uv run python -m skellyforge.skellymodels.standard_human.rest_pose_blender_export [output.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from skellyforge.kinematics.tpose import build_standard_human_tpose
from skellyforge.skellymodels.standard_human.human_skeleton import HumanSkeleton

_DEFAULT_OUTPUT = Path(__file__).with_name("standard_human_rest_pose.json")


def build_rest_armature() -> dict:
    """Build the JSON-able rest-armature document from the standard human."""
    skeleton = HumanSkeleton.standard_human()
    tpose = build_standard_human_tpose(skeleton)

    def point(name: str) -> list[float]:
        return [float(c) for c in tpose.landmarks[name]]

    bones = []
    for segment in skeleton.segments:
        geometry = tpose.segments[segment.name]
        bones.append(
            {
                "name": segment.name,
                "parent": segment.parent.name if segment.parent is not None else None,
                "head": point(segment.origin_landmark.name),
                "tail": point(segment.primary_axis.target_landmark),
                "length_mm": float(geometry.length),
                # rows = [x_hat, y_hat, z_hat] of the rest frame, world coords
                "basis": [[float(c) for c in row] for row in np.asarray(geometry.basis)],
            }
        )

    # each landmark's owning segment (its reference_frame) -- the frame the
    # Blender read-back back-projects a moved landmark's world position into.
    frame_of = {
        landmark.name: landmark.reference_frame
        for segment in skeleton.segments
        for landmark in segment.landmarks
    }
    landmarks = {
        name: {
            "position": [float(c) for c in position],
            "reference_frame": frame_of[name],
        }
        for name, position in tpose.landmarks.items()
    }

    return {
        "model": skeleton.name,
        "convention": {
            "units": "mm",
            "handedness": "right",
            "up": "+z",
            "forward": "+x",
            "left": "+y",
        },
        "counts": {"bones": len(bones), "landmarks": len(landmarks)},
        "bones": bones,
        "landmarks": landmarks,
    }


def export_rest_armature(output_path: str | Path | None = None) -> Path:
    """Write the rest-armature JSON; returns the path written."""
    output_path = Path(output_path) if output_path is not None else _DEFAULT_OUTPUT
    document = build_rest_armature()
    output_path.write_text(json.dumps(document, indent=2))
    return output_path


def _main() -> None:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OUTPUT
    written = export_rest_armature(output_path)
    document = json.loads(written.read_text())
    counts = document["counts"]
    print(f"wrote {written}")
    print(f"  bones: {counts['bones']}  landmarks: {counts['landmarks']}")
    for name in ("pelvis", "left_hand", "left_upper_leg"):
        bone = next(b for b in document["bones"] if b["name"] == name)
        head = np.round(bone["head"], 1)
        tail = np.round(bone["tail"], 1)
        print(f"  {name:16} head {head}  tail {tail}  len {bone['length_mm']:.0f}mm")


if __name__ == "__main__":
    _main()
