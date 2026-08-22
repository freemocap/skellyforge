"""Generate a self-contained three.js animation of the standard-human hydration pipeline.

Loads the skeleton and its rest pose, then synthesizes a smooth, looped upper-limb motion
(shoulder oscillating vertically and horizontally, elbow flexing 0-90 degrees) over a
sequence of frames. Each frame: forward kinematics projects the landmarks into world
space, a little noise is added, and the closed-form hydration (hydrate_skeleton) recovers the
pose. The viewer then plays the sequence in two panels:

  * LEFT - the hydrated segment frames (bones + full-orientation axis gizmos), the observed
    landmarks, and the ground-truth pose as a faded overlay;
  * RIGHT - stacked time-series of the quantities that matter, so the synthetic landmark
    noise can be told apart from solver error: the landmark residual (observed - true), the
    per-segment bone-direction error, the shoulder/elbow/head joint angles, and the
    rigid-fit orientation error (a root + head tracking check).

The hydration is a per-frame, lag-free closed form. The only freedom it leaves is the roll of a
two-landmark segment about its long axis; that roll is resolved here with a lag-free continuous
(parallel-transport) convention so the full orientation is stable across frames without any
temporal filtering.

Run from the repo root: python scripts/generate_skeleton_viewer.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton_parts.rest_pose import build_rest_pose
from skellyforge.core.skeleton_parts.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton_parts.skeleton_hydration import hydrate_skeleton

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = REPO_ROOT / "skellyforge" / "definitions" / "human_skeleton"
OUTPUT_PATH = Path(__file__).resolve().parent / "skeleton_viewer.html"

FRAME_COUNT = 60
FPS = 30.0
LANDMARK_NOISE_MM = 1.0
SHOULDER_VERTICAL_AMP = 0.7
SHOULDER_HORIZONTAL_AMP = 0.55
ELBOW_FLEX_AMP = np.pi / 2.0
HEAD_NOD_AMP = 0.18
HEAD_TURN_AMP = 0.12
SYNTHESIS_SEED = 20260801


def _side_of(name: str) -> str:
    if name.startswith("left_"):
        return "left"
    if name.startswith("right_"):
        return "right"
    return "midline"


def _vec(array) -> list[float]:
    return [round(float(value), 2) for value in array]


def _round_list(values) -> list[float]:
    return [round(float(value), 2) for value in values]


def _angle_between(a, b) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def _read_rest_pose_tree() -> tuple[dict, dict, dict]:
    document = yaml.safe_load((DEFINITIONS / "rest_pose.yaml").read_text(encoding="utf-8"))
    parents: dict[str, str | None] = {}
    connect_ats: dict[str, str | None] = {}
    orientations: dict[str, RotationQuaternion] = {}
    for name, entry in document["segments"].items():
        parents[name] = entry.get("parent")
        connect_ats[name] = entry.get("connect_at")
        orientation = entry.get("orientation", [1.0, 0.0, 0.0, 0.0])
        orientations[name] = RotationQuaternion.from_components(
            w=float(orientation[0]),
            x=float(orientation[1]),
            y=float(orientation[2]),
            z=float(orientation[3]),
        )
    return parents, connect_ats, orientations


def _descendants(parents: dict[str, str | None], root: str) -> list[str]:
    children: dict[str, list[str]] = defaultdict(list)
    for name, parent in parents.items():
        if parent is not None:
            children[parent].append(name)
    result: list[str] = []
    stack = [root]
    while stack:
        name = stack.pop()
        result.append(name)
        stack.extend(children[name])
    return result


def _perpendicular(direction: np.ndarray) -> np.ndarray:
    """A deterministic unit vector perpendicular to direction."""
    reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(direction[0])) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    perpendicular = reference - float(np.dot(reference, direction)) * direction
    return perpendicular / np.linalg.norm(perpendicular)


def _continuous_roll_orientation(
    raw: RotationQuaternion,
    primary_local: np.ndarray,
    secondary_local: np.ndarray,
    previous_secondary: np.ndarray | None,
) -> tuple[RotationQuaternion, np.ndarray]:
    """Repoint a two-landmark segment's free roll with a lag-free parallel transport.

    The hydration fixes the world direction but leaves the roll about it free. Rather than the
    per-frame shortest-arc roll (which jumps when the direction crosses a pole), carry the
    previous frame's secondary axis forward and orthonormalize it against the new direction.
    This picks the roll closest to the previous frame's - continuous, and with no temporal lag.
    """
    world_primary = raw.rotate_vector(primary_local)
    e1 = world_primary / np.linalg.norm(world_primary)

    if previous_secondary is None:
        reference = raw.rotate_vector(secondary_local)
    else:
        reference = previous_secondary

    e2 = reference - float(np.dot(reference, e1)) * e1
    norm = float(np.linalg.norm(e2))
    if norm < 1e-9:
        # The carried secondary is (nearly) parallel to the new direction: pick a fresh one.
        e2 = _perpendicular(e1)
    else:
        e2 = e2 / norm
    e3 = np.cross(e1, e2)

    world_basis = np.column_stack([e1, e2, e3])
    tertiary_local = np.cross(primary_local, secondary_local)
    local_basis = np.column_stack([primary_local, secondary_local, tertiary_local])
    rotation = RotationQuaternion.from_rotation_matrix(world_basis @ local_basis.T)
    return rotation, e2


def _build_data() -> dict:
    skeleton = SkeletonDefinition.from_yaml(path=DEFINITIONS / "human_skeleton.yaml")
    parents, connect_ats, rest_orientations = _read_rest_pose_tree()

    rest_world, _, _ = build_rest_pose(
        skeleton=skeleton,
        parents=parents,
        connect_ats=connect_ats,
        orientations=rest_orientations,
    )

    segment_order = list(skeleton.segments.keys())
    landmark_order = list(skeleton.landmarks.keys())

    left_arm = set(_descendants(parents, "left_upper_arm"))
    right_arm = set(_descendants(parents, "right_upper_arm"))
    left_forearm = set(_descendants(parents, "left_lower_arm"))
    right_forearm = set(_descendants(parents, "right_lower_arm"))

    primary_locals = {
        name: skeleton.landmarks[skeleton.segments[name].frame_definition.primary_point_name].local_position.array
        for name in segment_order
    }

    finger_tip_name = skeleton.segments["left_index_distal_phalanx"].frame_definition.primary_point_name

    fit: dict[str, str] = {}
    primary_units: dict[str, np.ndarray] = {}
    secondary_locals: dict[str, np.ndarray] = {}
    segments_meta = []
    for name in segment_order:
        segment = skeleton.segments[name]
        owned_count = len(segment.landmarks)
        if owned_count >= 3:
            local_positions = np.stack(
                [landmark.local_position.array for landmark in segment.landmarks.values()],
                axis=0,
            )
            centered = local_positions - local_positions.mean(axis=0)
            singular_values = np.linalg.svd(centered, compute_uv=False)
            spans_plane = singular_values[1] > 1e-6 * singular_values[0]
        else:
            spans_plane = False
        kind = "rigid" if spans_plane else "direction"
        fit[name] = kind

        primary_local = primary_locals[name]
        primary_unit = primary_local / np.linalg.norm(primary_local)
        primary_units[name] = primary_unit
        secondary_locals[name] = _perpendicular(primary_unit)

        segments_meta.append(
            {
                "name": name,
                "side": _side_of(name),
                "fit": kind,
                "landmarks": owned_count,
                "length": round(float(np.linalg.norm(primary_local)), 2),
            }
        )

    rng = np.random.default_rng(SYNTHESIS_SEED)
    carried_secondary: dict[str, np.ndarray] = {}

    frames = []
    all_true_positions = []

    # Time-series accumulators (full precision; rounded only when serialized).
    ts_noise: list[list[float]] = [[], [], []]  # fingertip observed - true, x/y/z
    ts_err_upper_arm: list[float] = []
    ts_err_finger: list[float] = []
    ts_shoulder_vertical: list[float] = []
    ts_shoulder_horizontal: list[float] = []
    ts_elbow_true: list[float] = []
    ts_elbow_rec: list[float] = []
    ts_pelvis_err: list[float] = []
    ts_skull_err: list[float] = []

    for t in range(FRAME_COUNT):
        # Three smoothly-looping degrees of freedom: shoulder vertical (about the forward
        # axis), shoulder horizontal (about the vertical axis), and elbow flexion (about the
        # forward axis). The elbow bends first (in the arm's rest frame), then the shoulder
        # moves the whole arm; the trunk and root stay put.
        vertical_angle = SHOULDER_VERTICAL_AMP * np.sin(2.0 * np.pi * t / FRAME_COUNT)
        horizontal_angle = SHOULDER_HORIZONTAL_AMP * np.sin(4.0 * np.pi * t / FRAME_COUNT)
        elbow_angle = ELBOW_FLEX_AMP * (1.0 - np.cos(4.0 * np.pi * t / FRAME_COUNT)) / 2.0

        shoulder_left = RotationQuaternion.from_rotation_vector(
            np.array([0.0, -horizontal_angle, vertical_angle])
        )
        shoulder_right = RotationQuaternion.from_rotation_vector(
            np.array([0.0, horizontal_angle, -vertical_angle])
        )
        elbow_left = RotationQuaternion.from_rotation_vector(np.array([0.0, 0.0, elbow_angle]))
        elbow_right = RotationQuaternion.from_rotation_vector(np.array([0.0, 0.0, -elbow_angle]))

        # A little head nod (pitch) + turn (yaw), skull only, so its landmarks move with it.
        head_nod = HEAD_NOD_AMP * np.sin(2.0 * np.pi * t / FRAME_COUNT + 1.2)
        head_turn = HEAD_TURN_AMP * np.sin(4.0 * np.pi * t / FRAME_COUNT + 0.6)
        head_rotation = RotationQuaternion.from_rotation_vector(np.array([head_nod, head_turn, 0.0]))

        world = {}
        for name in segment_order:
            orientation = rest_world[name]
            if name in left_forearm:
                orientation = elbow_left * orientation
            elif name in right_forearm:
                orientation = elbow_right * orientation
            if name in left_arm:
                orientation = shoulder_left * orientation
            elif name in right_arm:
                orientation = shoulder_right * orientation
            if name == "skull":
                orientation = head_rotation * orientation
            world[name] = orientation

        relative = {}
        for name in segment_order:
            parent = parents[name]
            relative[name] = world[name] if parent is None else world[parent].inverse() * world[name]

        world_orientations, world_origins, true_landmarks = build_rest_pose(
            skeleton=skeleton,
            parents=parents,
            connect_ats=connect_ats,
            orientations=relative,
        )

        observed = {
            name: Point.from_prevalidated_array(
                array=point.array + rng.normal(scale=LANDMARK_NOISE_MM, size=3)
            )
            for name, point in true_landmarks.items()
        }

        hydrated = hydrate_skeleton(skeleton=skeleton, observed=observed)

        frame_segments = []
        dirs_rec = {}
        dirs_true = {}
        for name in segment_order:
            hydrated_pose = hydrated.segment_poses[name]
            raw_rotation = hydrated_pose.orientation

            if fit[name] == "direction":
                rotation, carried = _continuous_roll_orientation(
                    raw=raw_rotation,
                    primary_local=primary_units[name],
                    secondary_local=secondary_locals[name],
                    previous_secondary=carried_secondary.get(name),
                )
                carried_secondary[name] = carried
            else:
                rotation = raw_rotation

            matrix = rotation.to_rotation_matrix()
            primary_local = primary_locals[name]

            rec_dir = rotation.rotate_vector(primary_local)
            true_dir = world_orientations[name].rotate_vector(primary_local)
            dirs_rec[name] = rec_dir
            dirs_true[name] = true_dir

            if name == "left_upper_arm":
                ts_err_upper_arm.append(np.degrees(_angle_between(rec_dir, true_dir)))
            elif name == "left_index_distal_phalanx":
                ts_err_finger.append(np.degrees(_angle_between(rec_dir, true_dir)))
            elif name in ("pelvis", "skull"):
                rigid_error = float(
                    np.degrees(
                        np.linalg.norm(
                            (rotation * world_orientations[name].inverse()).to_rotation_vector()
                        )
                    )
                )
                if name == "pelvis":
                    ts_pelvis_err.append(rigid_error)
                else:
                    ts_skull_err.append(rigid_error)

            frame_segments.append(
                {
                    "origin": _vec(hydrated_pose.origin.array),
                    "end": _vec(hydrated_pose.origin.array + rec_dir),
                    "basis": [
                        _vec(matrix[:, 0]),
                        _vec(matrix[:, 1]),
                        _vec(matrix[:, 2]),
                    ],
                    "gt_origin": _vec(world_origins[name].array),
                    "gt_end": _vec(world_origins[name].array + true_dir),
                }
            )

        ts_shoulder_vertical.append(float(np.degrees(vertical_angle)))
        ts_shoulder_horizontal.append(float(np.degrees(horizontal_angle)))
        ts_elbow_true.append(
            float(np.degrees(_angle_between(dirs_true["left_upper_arm"], dirs_true["left_lower_arm"])))
        )
        ts_elbow_rec.append(
            float(np.degrees(_angle_between(dirs_rec["left_upper_arm"], dirs_rec["left_lower_arm"])))
        )

        noise = observed[finger_tip_name].array - true_landmarks[finger_tip_name].array
        ts_noise[0].append(float(noise[0]))
        ts_noise[1].append(float(noise[1]))
        ts_noise[2].append(float(noise[2]))

        frames.append(
            {
                "segments": frame_segments,
                "landmarks": [_vec(observed[name].array) for name in landmark_order],
            }
        )
        all_true_positions.extend(point.array for point in true_landmarks.values())

    center = np.mean(all_true_positions, axis=0)

    rigid_count = sum(1 for meta in segments_meta if meta["fit"] == "rigid")

    timeseries = {
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "panels": [
            {
                "title": "Fingertip landmark noise - " + finger_tip_name + " (observed - true)",
                "unit": "mm",
                "series": [
                    {"name": "x", "color": "#ff6b6b", "values": _round_list(ts_noise[0])},
                    {"name": "y", "color": "#51cf66", "values": _round_list(ts_noise[1])},
                    {"name": "z", "color": "#5c7cfa", "values": _round_list(ts_noise[2])},
                ],
            },
            {
                "title": "Bone direction error (recovered - true)",
                "unit": "deg",
                "series": [
                    {"name": "left_upper_arm", "color": "#e74c3c", "values": _round_list(ts_err_upper_arm)},
                    {"name": "left_index_distal_phalanx", "color": "#f1c40f", "values": _round_list(ts_err_finger)},
                ],
            },
            {
                "title": "Shoulder angles (true)",
                "unit": "deg",
                "series": [
                    {"name": "vertical", "color": "#e74c3c", "values": _round_list(ts_shoulder_vertical)},
                    {"name": "horizontal", "color": "#3498db", "values": _round_list(ts_shoulder_horizontal)},
                ],
            },
            {
                "title": "Elbow flexion",
                "unit": "deg",
                "series": [
                    {"name": "true", "color": "#ffffff", "values": _round_list(ts_elbow_true)},
                    {"name": "recovered", "color": "#e74c3c", "values": _round_list(ts_elbow_rec)},
                ],
            },
            {
                "title": "Rigid-fit orientation error (pelvis + skull)",
                "unit": "deg",
                "series": [
                    {"name": "pelvis", "color": "#95a5a6", "values": _round_list(ts_pelvis_err)},
                    {"name": "skull", "color": "#e67e22", "values": _round_list(ts_skull_err)},
                ],
            },
        ],
    }

    return {
        "center": _vec(center),
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "segments_meta": segments_meta,
        "landmarks_meta": list(landmark_order),
        "frames": frames,
        "timeseries": timeseries,
        "counts": {
            "segments": len(segments_meta),
            "landmarks": len(landmark_order),
            "rigid": rigid_count,
            "direction": len(segments_meta) - rigid_count,
        },
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Standard Human Skeleton - Hydration</title>
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #1a1a2e; }
  body { display: flex; }
  #left { flex: 1 1 55%; position: relative; min-width: 0; }
  #left canvas { display: block; }
  #right {
    flex: 1 1 45%; min-width: 0; overflow-y: auto;
    background: #12122a; border-left: 1px solid #2a2a44;
    padding: 12px; box-sizing: border-box;
  }
  #charts-head {
    color: #eee; font-family: system-ui, sans-serif; font-size: 14px; font-weight: 600;
    margin: 0 0 2px 0;
  }
  #charts-sub { color: #999; font-family: system-ui, sans-serif; font-size: 11px; margin: 0 0 10px 0; }
  .panel { background: rgba(0, 0, 0, 0.35); border-radius: 6px; padding: 8px 10px; margin-bottom: 10px; }
  .panel-title { color: #ddd; font-family: system-ui, sans-serif; font-size: 12px; margin-bottom: 4px; }
  .panel-legend { margin-bottom: 4px; }
  .legend-item { display: inline-block; margin-right: 12px; color: #bbb; font-size: 11px; font-family: system-ui, sans-serif; }
  .legend-item .dot { width: 8px; height: 8px; margin-right: 3px; }
  .chart-row { display: flex; align-items: stretch; }
  .chart-row svg { flex: 1 1 auto; min-width: 0; height: 116px; display: block; background: #0c0c1c; border-radius: 4px; cursor: crosshair; }
  .yaxis {
    flex: 0 0 42px; width: 42px;
    display: flex; flex-direction: column; justify-content: space-between; align-items: flex-end;
    padding: 0 6px 2px 0; box-sizing: border-box;
    color: #999; font-family: system-ui, sans-serif; font-size: 10px; line-height: 1;
  }
  .yaxis .y-unit { writing-mode: vertical-rl; text-orientation: mixed; align-self: center; color: #aaa; }
  .xaxis {
    display: flex; justify-content: space-between; align-items: baseline;
    color: #999; font-family: system-ui, sans-serif; font-size: 10px;
    padding: 3px 2px 0 48px;
  }

  #info {
    position: absolute; top: 16px; left: 16px; z-index: 10;
    background: rgba(0, 0, 0, 0.6); color: #eee;
    padding: 12px 16px; border-radius: 8px;
    font-family: system-ui, sans-serif; font-size: 13px;
  }
  #info h1 { font-size: 15px; margin: 0 0 6px 0; }
  #info p { margin: 3px 0; }
  #info label { display: inline-block; margin-right: 12px; cursor: pointer; }
  #info input { vertical-align: middle; margin-right: 3px; }
  #info button {
    background: #3498db; color: #fff; border: 0; border-radius: 4px;
    padding: 3px 12px; margin-right: 8px; cursor: pointer; font-size: 13px;
  }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
  #tooltip {
    position: absolute; display: none; z-index: 11;
    background: rgba(0, 0, 0, 0.85); color: #fff;
    padding: 4px 10px; border-radius: 4px;
    font-family: system-ui, sans-serif; font-size: 12px;
    pointer-events: none; white-space: nowrap;
  }
</style>
</head>
<body>
<div id="left">
  <div id="info">
    <h1>Standard Human Skeleton &mdash; Hydration</h1>
    <p id="counts"></p>
    <p><button id="togglePlay">pause</button><span id="frameLabel"></span></p>
    <p>
      <label><input type="checkbox" id="toggleGt" checked> ground truth</label>
      <label><input type="checkbox" id="toggleAxes" checked> orientation axes</label>
      <label><input type="checkbox" id="toggleLandmarks" checked> landmarks</label>
    </p>
    <p>
      <span class="dot" style="background:#e74c3c"></span> left bone
      <span class="dot" style="background:#3498db"></span> right bone
      <span class="dot" style="background:#95a5a6"></span> midline bone
      <span class="dot" style="background:#f1c40f"></span> landmark (observed)
    </p>
    <p style="opacity:.85">
      <span class="dot" style="background:#ff6b6b"></span> local x
      <span class="dot" style="background:#51cf66"></span> local y
      <span class="dot" style="background:#5c7cfa"></span> local z (full orientation)
      <span class="dot" style="background:#ffffff"></span> ground truth (faded)
    </p>
  </div>
  <div id="tooltip"></div>
</div>
<div id="right">
  <p id="charts-head">Time series</p>
  <p id="charts-sub">Click a chart to seek. The vertical cursor is the current frame.</p>
  <div id="charts"></div>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
var DATA = __DATA__;

var COLORS = { midline: 0x95a5a6, left: 0xe74c3c, right: 0x3498db };
var AXIS_COLORS = [0xff6b6b, 0x51cf66, 0x5c7cfa];
var LANDMARK_COLOR = 0xf1c40f;
var GT_COLOR = 0xffffff;
var BONE_RADIUS = 5;
var GT_RADIUS = 3;
var LANDMARK_RADIUS = 9;
var GIZMO_LENGTH = 30;
var GIZMO_RADIUS = 1.5;
var GT_OPACITY = 0.25;

var UP = new THREE.Vector3(0, 1, 0);

var leftEl = document.getElementById("left");
var scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

var camera = new THREE.PerspectiveCamera(50, 1, 10, 20000);

var renderer = new THREE.WebGLRenderer({ antialias: true });
leftEl.appendChild(renderer.domElement);

var controls = new THREE.OrbitControls(camera, renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
var light = new THREE.DirectionalLight(0xffffff, 0.5);
light.position.set(1, 1, 1);
scene.add(light);

scene.add(new THREE.AxesHelper(200));

function vec3(a) { return new THREE.Vector3(a[0], a[1], a[2]); }

function makeCylinder(length, radius, color, opacity) {
  var mat = new THREE.MeshLambertMaterial({ color: color });
  if (opacity < 1) { mat.transparent = true; mat.opacity = opacity; }
  return new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, length, 12), mat);
}

function placeCylinder(cyl, a, b) {
  var d = new THREE.Vector3().subVectors(b, a);
  var len = d.length();
  if (len < 0.5) { cyl.visible = false; return; }
  cyl.visible = true;
  cyl.position.copy(a).addScaledVector(d, 0.5);
  cyl.quaternion.setFromUnitVectors(UP, d.normalize());
}

var CENTER = vec3(DATA.center);
controls.target.copy(CENTER);
camera.position.set(CENTER.x + 1000, CENTER.y + 600, CENTER.z + 1500);
controls.update();

var hoverables = [];
var gtGroup = new THREE.Group();
var segGroup = new THREE.Group();
var axisGroup = new THREE.Group();
var lmGroup = new THREE.Group();

var boneMeshes = [];
var gtMeshes = [];
var axisMeshes = [];

DATA.segments_meta.forEach(function (meta) {
  var bone = makeCylinder(meta.length, BONE_RADIUS, COLORS[meta.side], 1);
  bone.userData = { name: meta.name, kind: "segment", fit: meta.fit, landmarks: meta.landmarks };
  hoverables.push(bone);
  boneMeshes.push(bone);
  segGroup.add(bone);

  var gt = makeCylinder(meta.length, GT_RADIUS, GT_COLOR, GT_OPACITY);
  gtMeshes.push(gt);
  gtGroup.add(gt);

  var axes = [
    makeCylinder(GIZMO_LENGTH, GIZMO_RADIUS, AXIS_COLORS[0], 1),
    makeCylinder(GIZMO_LENGTH, GIZMO_RADIUS, AXIS_COLORS[1], 1),
    makeCylinder(GIZMO_LENGTH, GIZMO_RADIUS, AXIS_COLORS[2], 1)
  ];
  axes.forEach(function (a) { axisGroup.add(a); });
  axisMeshes.push(axes);
});

var landmarkMeshes = [];
DATA.landmarks_meta.forEach(function (name) {
  var sph = new THREE.Mesh(
    new THREE.SphereGeometry(LANDMARK_RADIUS, 10, 10),
    new THREE.MeshLambertMaterial({ color: LANDMARK_COLOR })
  );
  sph.userData = { name: name, kind: "landmark" };
  hoverables.push(sph);
  landmarkMeshes.push(sph);
  lmGroup.add(sph);
});

scene.add(gtGroup);
scene.add(segGroup);
scene.add(axisGroup);
scene.add(lmGroup);

function applyFrame(t) {
  var F = DATA.frames[t];
  for (var i = 0; i < DATA.segments_meta.length; i++) {
    var s = F.segments[i];
    placeCylinder(boneMeshes[i], vec3(s.origin), vec3(s.end));
    placeCylinder(gtMeshes[i], vec3(s.gt_origin), vec3(s.gt_end));
    var o = vec3(s.origin);
    for (var j = 0; j < 3; j++) {
      var d = vec3(s.basis[j]).normalize().multiplyScalar(GIZMO_LENGTH);
      placeCylinder(axisMeshes[i][j], o, o.clone().add(d));
    }
  }
  for (var k = 0; k < F.landmarks.length; k++) {
    var p = F.landmarks[k];
    landmarkMeshes[k].position.set(p[0], p[1], p[2]);
  }
}

document.getElementById("counts").textContent =
  DATA.counts.segments + " segments (" + DATA.counts.rigid + " rigid-fit / " +
  DATA.counts.direction + " direction-only) / " + DATA.counts.landmarks + " landmarks / " +
  DATA.frame_count + " frames";

var frame = 0;
var playing = true;
var accum = 0;
var last = performance.now();
var frameLabel = document.getElementById("frameLabel");

document.getElementById("togglePlay").addEventListener("click", function (e) {
  playing = !playing;
  e.target.textContent = playing ? "pause" : "play";
});

document.getElementById("toggleGt").addEventListener("change", function (e) { gtGroup.visible = e.target.checked; });
document.getElementById("toggleAxes").addEventListener("change", function (e) { axisGroup.visible = e.target.checked; });
document.getElementById("toggleLandmarks").addEventListener("change", function (e) { lmGroup.visible = e.target.checked; });

var raycaster = new THREE.Raycaster();
var mouse = new THREE.Vector2();
var tooltip = document.getElementById("tooltip");
var hovered = null;

renderer.domElement.addEventListener("mousemove", function (event) {
  var rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  var hits = raycaster.intersectObjects(hoverables);
  if (hits.length > 0) {
    var obj = hits[0].object;
    if (hovered !== obj) {
      if (hovered) hovered.material.emissive.setHex(0x000000);
      hovered = obj;
      hovered.material.emissive.setHex(0xffffff);
    }
    tooltip.style.display = "block";
    tooltip.style.left = (event.clientX - rect.left + 14) + "px";
    tooltip.style.top = (event.clientY - rect.top + 14) + "px";
    tooltip.textContent = obj.userData.kind === "segment"
      ? obj.userData.name + " (" + obj.userData.fit + ", " + obj.userData.landmarks + " landmarks)"
      : obj.userData.name;
    renderer.domElement.style.cursor = "pointer";
  } else {
    if (hovered) {
      hovered.material.emissive.setHex(0x000000);
      hovered = null;
    }
    tooltip.style.display = "none";
    renderer.domElement.style.cursor = "default";
  }
});

renderer.domElement.addEventListener("mouseleave", function () {
  if (hovered) {
    hovered.material.emissive.setHex(0x000000);
    hovered = null;
  }
  tooltip.style.display = "none";
});

function resize() {
  var w = leftEl.clientWidth || 1;
  var h = leftEl.clientHeight || 1;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// ---- Right-hand time-series charts ----
var NS = "http://www.w3.org/2000/svg";
var chartsEl = document.getElementById("charts");
var charts = [];

DATA.timeseries.panels.forEach(function (panel) {
  var div = document.createElement("div");
  div.className = "panel";

  var title = document.createElement("div");
  title.className = "panel-title";
  title.textContent = panel.title;
  div.appendChild(title);

  var legend = document.createElement("div");
  legend.className = "panel-legend";
  panel.series.forEach(function (s) {
    var item = document.createElement("span");
    item.className = "legend-item";
    var dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = s.color;
    item.appendChild(dot);
    item.appendChild(document.createTextNode(s.name));
    legend.appendChild(item);
  });
  div.appendChild(legend);

  var N = panel.series[0].values.length;
  var dataMin = Infinity, dataMax = -Infinity;
  panel.series.forEach(function (s) {
    s.values.forEach(function (v) {
      if (v < dataMin) dataMin = v;
      if (v > dataMax) dataMax = v;
    });
  });
  if (!isFinite(dataMin) || !isFinite(dataMax)) { dataMin = -1; dataMax = 1; }
  if (dataMin === dataMax) { dataMin -= 1; dataMax += 1; }
  var pad = (dataMax - dataMin) * 0.08;
  var min = dataMin - pad, max = dataMax + pad;
  var span = dataMax - dataMin;
  var dec = span >= 50 ? 0 : (span >= 5 ? 1 : 2);
  function fmt(v) { return v.toFixed(dec); }

  var W = 400, H = 120;
  function xOf(i) { return (i / (N - 1)) * W; }
  function yOf(v) { return H - ((v - min) / (max - min)) * H; }

  var chartRow = document.createElement("div");
  chartRow.className = "chart-row";

  var yaxis = document.createElement("div");
  yaxis.className = "yaxis";
  var yMax = document.createElement("span"); yMax.textContent = fmt(dataMax);
  var yUnit = document.createElement("span"); yUnit.className = "y-unit"; yUnit.textContent = panel.unit;
  var yMin = document.createElement("span"); yMin.textContent = fmt(dataMin);
  yaxis.appendChild(yMax);
  yaxis.appendChild(yUnit);
  yaxis.appendChild(yMin);
  chartRow.appendChild(yaxis);

  var svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  svg.setAttribute("preserveAspectRatio", "none");
  chartRow.appendChild(svg);

  var zero = document.createElementNS(NS, "line");
  var zy = yOf(0);
  zero.setAttribute("x1", 0); zero.setAttribute("x2", W);
  zero.setAttribute("y1", zy); zero.setAttribute("y2", zy);
  zero.setAttribute("stroke", "#3a3a55"); zero.setAttribute("stroke-dasharray", "4 3");
  svg.appendChild(zero);

  panel.series.forEach(function (s) {
    var path = document.createElementNS(NS, "polyline");
    var pts = s.values.map(function (v, i) { return xOf(i).toFixed(1) + "," + yOf(v).toFixed(1); }).join(" ");
    path.setAttribute("points", pts);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", s.color);
    path.setAttribute("stroke-width", 1.5);
    svg.appendChild(path);
  });

  var cursor = document.createElementNS(NS, "line");
  cursor.setAttribute("y1", 0); cursor.setAttribute("y2", H);
  cursor.setAttribute("stroke", "#888"); cursor.setAttribute("stroke-width", 1);
  svg.appendChild(cursor);

  svg.addEventListener("click", function (e) {
    var rect = svg.getBoundingClientRect();
    var fx = (e.clientX - rect.left) / rect.width;
    frame = Math.max(0, Math.min(N - 1, Math.round(fx * (N - 1))));
    accum = 0;
    applyFrame(frame);
    updateCursors();
  });

  div.appendChild(chartRow);

  var xaxis = document.createElement("div");
  xaxis.className = "xaxis";
  var x0 = document.createElement("span"); x0.textContent = "0";
  var xmid = document.createElement("span"); xmid.textContent = "frame";
  var xN = document.createElement("span"); xN.textContent = String(N - 1);
  xaxis.appendChild(x0);
  xaxis.appendChild(xmid);
  xaxis.appendChild(xN);
  div.appendChild(xaxis);

  chartsEl.appendChild(div);
  charts.push({ cursor: cursor, W: W, N: N });
});

function updateCursors() {
  charts.forEach(function (c) {
    var x = (frame / (c.N - 1)) * c.W;
    c.cursor.setAttribute("x1", x);
    c.cursor.setAttribute("x2", x);
  });
}

function animate(now) {
  requestAnimationFrame(animate);
  var dt = (now - last) / 1000;
  last = now;
  if (playing && dt > 0 && dt < 0.5) {
    accum += dt * DATA.fps;
    var steps = Math.floor(accum);
    if (steps > 0) {
      accum -= steps;
      frame = (frame + steps) % DATA.frame_count;
    }
  }
  applyFrame(frame);
  updateCursors();
  frameLabel.textContent = "frame " + (frame + 1) + " / " + DATA.frame_count;
  controls.update();
  renderer.render(scene, camera);
}
applyFrame(0);
updateCursors();
animate(performance.now());
</script>
</body>
</html>
"""


def main() -> None:
    data = _build_data()
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    counts = data["counts"]
    print(f"wrote {OUTPUT_PATH}")
    print(
        f"  {counts['segments']} segments ({counts['rigid']} rigid-fit / "
        f"{counts['direction']} direction-only), {counts['landmarks']} landmarks, "
        f"{data['frame_count']} frames @ {data['fps']} fps, "
        f"{len(data['timeseries']['panels'])} time-series panels"
    )


if __name__ == "__main__":
    main()
