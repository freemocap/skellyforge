"""Generate a self-contained three.js viewer for the standard-human skeleton.

Loads the skeleton and its rest pose, projects every segment (along its primary axis) and
landmark into world space, and writes a single HTML file that renders them with three.js.
Run from the repo root: python scripts/generate_skeleton_viewer.py
"""

from __future__ import annotations

import json
from pathlib import Path

from skellyforge.core.skeleton_parts.rest_pose import RestPose
from skellyforge.core.skeleton_parts.skeleton_definition import SkeletonDefinition

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = REPO_ROOT / "skellyforge" / "definitions" / "human_skeleton"
OUTPUT_PATH = Path(__file__).resolve().parent / "skeleton_viewer.html"


def _side_of(name: str) -> str:
    if name.startswith("left_"):
        return "left"
    if name.startswith("right_"):
        return "right"
    return "midline"


def _build_data() -> dict:
    skeleton = SkeletonDefinition.from_yaml(path=DEFINITIONS / "human_skeleton.yaml")
    pose = RestPose.from_yaml(path=DEFINITIONS / "rest_pose.yaml", skeleton=skeleton)

    segments = []
    for segment in skeleton.segments.values():
        origin = pose.segment_origins[segment.name].array
        primary_local = skeleton.landmarks[segment.frame_definition.primary_point_name].local_position.array
        end = origin + pose.segment_orientations[segment.name].rotate_vector(primary_local)
        segments.append(
            {
                "name": segment.name,
                "side": _side_of(segment.name),
                "origin": [round(float(v), 2) for v in origin],
                "end": [round(float(v), 2) for v in end],
            }
        )

    landmarks = [
        {
            "name": landmark.name,
            "side": _side_of(landmark.name),
            "position": [round(float(v), 2) for v in pose.landmark_positions[landmark.name].array],
        }
        for landmark in skeleton.landmarks.values()
    ]

    return {"segments": segments, "landmarks": landmarks}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Standard Human Skeleton</title>
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #1a1a2e; }
  canvas { display: block; }
  #info {
    position: absolute; top: 16px; left: 16px;
    background: rgba(0, 0, 0, 0.6); color: #eee;
    padding: 12px 16px; border-radius: 8px;
    font-family: system-ui, sans-serif; font-size: 13px;
    pointer-events: none;
  }
  #info h1 { font-size: 15px; margin: 0 0 6px 0; }
  #info p { margin: 2px 0; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
  #tooltip {
    position: absolute; display: none;
    background: rgba(0, 0, 0, 0.85); color: #fff;
    padding: 4px 10px; border-radius: 4px;
    font-family: system-ui, sans-serif; font-size: 12px;
    pointer-events: none; white-space: nowrap;
  }
</style>
</head>
<body>
<div id="info">
  <h1>Standard Human Skeleton</h1>
  <p id="counts"></p>
  <p>
    <span class="dot" style="background:#e74c3c"></span> left
    <span class="dot" style="background:#3498db"></span> right
    <span class="dot" style="background:#95a5a6"></span> midline
    <span class="dot" style="background:#f1c40f"></span> landmark
  </p>
</div>
<div id="tooltip"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
var DATA = __DATA__;

var COLORS = { midline: 0x95a5a6, left: 0xe74c3c, right: 0x3498db };
var LANDMARK_COLOR = 0xf1c40f;
var BONE_RADIUS = 5;
var LANDMARK_RADIUS = 9;

var scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

var camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 10, 20000);
camera.position.set(1000, 600, 1400);

var renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

var controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(0, 50, 0);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
var light = new THREE.DirectionalLight(0xffffff, 0.5);
light.position.set(1, 1, 1);
scene.add(light);

scene.add(new THREE.AxesHelper(200));

var hoverables = [];

var segGroup = new THREE.Group();
DATA.segments.forEach(function (seg) {
  var a = new THREE.Vector3(seg.origin[0], seg.origin[1], seg.origin[2]);
  var b = new THREE.Vector3(seg.end[0], seg.end[1], seg.end[2]);
  var d = b.clone().sub(a);
  var len = d.length();
  if (len < 0.5) return;
  var cyl = new THREE.Mesh(
    new THREE.CylinderGeometry(BONE_RADIUS, BONE_RADIUS, len, 10),
    new THREE.MeshLambertMaterial({ color: COLORS[seg.side] })
  );
  cyl.position.copy(a).addScaledVector(d, 0.5);
  cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.clone().normalize());
  cyl.userData.name = seg.name;
  hoverables.push(cyl);
  segGroup.add(cyl);
});
scene.add(segGroup);

var lmGroup = new THREE.Group();
DATA.landmarks.forEach(function (lm) {
  var sph = new THREE.Mesh(
    new THREE.SphereGeometry(LANDMARK_RADIUS, 10, 10),
    new THREE.MeshLambertMaterial({ color: LANDMARK_COLOR })
  );
  sph.position.set(lm.position[0], lm.position[1], lm.position[2]);
  sph.userData.name = lm.name;
  hoverables.push(sph);
  lmGroup.add(sph);
});
scene.add(lmGroup);

document.getElementById("counts").textContent =
  DATA.segments.length + " segments / " + DATA.landmarks.length + " landmarks";

var raycaster = new THREE.Raycaster();
var mouse = new THREE.Vector2();
var tooltip = document.getElementById("tooltip");
var hovered = null;

window.addEventListener("mousemove", function (event) {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
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
    tooltip.style.left = (event.clientX + 14) + "px";
    tooltip.style.top = (event.clientY + 14) + "px";
    tooltip.textContent = obj.userData.name;
    document.body.style.cursor = "pointer";
  } else {
    if (hovered) {
      hovered.material.emissive.setHex(0x000000);
      hovered = null;
    }
    tooltip.style.display = "none";
    document.body.style.cursor = "default";
  }
});

window.addEventListener("resize", function () {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
"""


def main() -> None:
    data = _build_data()
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    print(f"  {len(data['segments'])} segments, {len(data['landmarks'])} landmarks")


if __name__ == "__main__":
    main()
