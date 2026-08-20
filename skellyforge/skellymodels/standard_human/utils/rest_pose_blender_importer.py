"""Load the standard-human rest pose into Blender, and read edits back to YAML values.

Runs INSIDE Blender (imports ``bpy``, not skellyforge). Reads the JSON written by
``rest_pose_blender_export.py`` and builds:
  - an Armature "standard_human" -- one named bone per segment (head/tail/parent/
    roll from the rest pose), as read-only visual context;
  - one NAMED Empty per landmark, grouped into collections:
      * "standard_human_editable"  -- left_* + midline landmarks (drag these)
      * "standard_human_mirrored"  -- right_* landmarks (mirror-derived; don't edit)

Workflow (positions only): drag an editable empty to where it should be, then press
"Read back to YAML" in the 3D-viewport sidebar (N) > "StdHuman" tab. It prints, for
every moved empty, the local ``rest_position`` to paste into the matching part YAML
(the left_/right_ prefix is stripped to the authored generic name). Regenerate with
``python -m ...rest_pose_blender_export`` and re-run this script to see the effect.

The model is millimetres, +Z up, +X forward, +Y left; Blender is +Z up, so only a
mm<->m scale is applied.

Run it (registers the panel + loads the pose):
  - Blender Text Editor: open this file, set JSON_PATH below, press Run.
  - CLI: blender --python rest_pose_blender_importer.py
"""

import json
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

# Absolute path to the JSON written by rest_pose_blender_export.py. Blender's Text
# Editor gives a run script no reliable __file__, so this is a hard absolute path
# on purpose -- edit this one line if you move the JSON.
JSON_PATH = r"C:\Users\jonma\code_repos\github\freemocap\project\skellyforge\skellyforge\skellymodels\standard_human\standard_human_rest_pose.json"

_SCALE = 0.001  # millimetres -> metres
_MIN_BONE_LENGTH = 1e-4  # metres; Blender deletes shorter bones
_MOVED_EPSILON_MM = 0.5  # a landmark counts as "moved" past this world distance

_EDITABLE_COLLECTION = "standard_human_editable"
_MIRRORED_COLLECTION = "standard_human_mirrored"
_ARMATURE_NAME = "standard_human"


# ----------------------------------------------------------------------------- #
# scene teardown / build
# ----------------------------------------------------------------------------- #

def _remove_object(obj) -> None:
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if isinstance(data, bpy.types.Armature):
        bpy.data.armatures.remove(data)


def _clear_previous() -> None:
    for obj in list(bpy.data.objects):
        if obj.name == _ARMATURE_NAME or obj.get("standard_human_landmark") is not None:
            _remove_object(obj)
    for collection_name in (_EDITABLE_COLLECTION, _MIRRORED_COLLECTION):
        collection = bpy.data.collections.get(collection_name)
        if collection is not None:
            bpy.data.collections.remove(collection)


def _get_or_make_collection(name: str):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def _build_armature(document: dict, include_mirrored: bool) -> None:
    armature_data = bpy.data.armatures.new(_ARMATURE_NAME)
    armature_object = bpy.data.objects.new(_ARMATURE_NAME, armature_data)
    bpy.context.scene.collection.objects.link(armature_object)
    bpy.ops.object.select_all(action="DESELECT")
    armature_object.select_set(True)
    bpy.context.view_layer.objects.active = armature_object
    bpy.ops.object.mode_set(mode="EDIT")

    # a segment IS its own reference frame, so its name is what _is_editable reads
    bones = [
        bone for bone in document["bones"]
        if include_mirrored or _is_editable(bone["name"])
    ]
    included = {bone["name"] for bone in bones}

    edit_bones = armature_data.edit_bones
    for bone in bones:
        head = Vector(bone["head"]) * _SCALE
        tail = Vector(bone["tail"]) * _SCALE
        if (tail - head).length < _MIN_BONE_LENGTH:
            tail = head + Vector((0.0, 0.0, _MIN_BONE_LENGTH))
        edit_bone = edit_bones.new(bone["name"])
        edit_bone.head = head
        edit_bone.tail = tail
        edit_bone.align_roll(Vector(bone["basis"][2]))

    for bone in bones:
        if bone["parent"] and bone["parent"] in included:
            edit_bones[bone["name"]].parent = edit_bones[bone["parent"]]
            edit_bones[bone["name"]].use_connect = False

    bpy.ops.object.mode_set(mode="OBJECT")


def _is_editable(reference_frame: str) -> bool:
    """True when a frame is AUTHORED rather than mirror-derived.

    The loader instantiates a sided part twice, generating the right side by
    mirroring; only the left/midline data is written in the YAML. Sides are named
    with a ``.R`` / ``.L`` suffix (VRM/Blender convention, e.g. ``hand.R``); the
    ``right_`` prefix form is also accepted. Midline parts (pelvis, spine, ...)
    author BOTH sides explicitly, so their ``right_hip`` / ``right_asis`` landmarks
    are editable -- which is why this keys off the FRAME, not the landmark name.
    """
    return not (reference_frame.endswith(".R") or reference_frame.startswith("right_"))


def _build_landmark_empties(document: dict, include_mirrored: bool) -> None:
    editable = _get_or_make_collection(_EDITABLE_COLLECTION)
    mirrored = (
        _get_or_make_collection(_MIRRORED_COLLECTION) if include_mirrored else None
    )
    for name, entry in document["landmarks"].items():
        if not include_mirrored and not _is_editable(entry["reference_frame"]):
            continue
        world = Vector(entry["position"]) * _SCALE
        empty = bpy.data.objects.new(name, None)
        empty.empty_display_type = "SPHERE"
        empty.empty_display_size = 0.006
        empty.location = world
        empty["standard_human_landmark"] = True
        empty["reference_frame"] = entry["reference_frame"]
        empty["original_location"] = tuple(world)  # metres, for change detection
        collection = editable if _is_editable(entry["reference_frame"]) else mirrored
        collection.objects.link(empty)


def _include_mirrored() -> bool:
    scene = bpy.context.scene
    return bool(getattr(scene, "standard_human_include_mirrored", True))


def load_rest_pose(json_path: str = JSON_PATH, include_mirrored: bool | None = None) -> None:
    """Build the armature + landmark empties.

    include_mirrored=False loads ONLY the authored (unmirrored) data -- the
    left/midline landmarks and bones actually written in the YAML. The right-side
    limbs are generated by the loader's sidedness mirror, so they are noise while
    tuning anatomy. None reads the scene checkbox.
    """
    if include_mirrored is None:
        include_mirrored = _include_mirrored()
    document = json.loads(Path(json_path).read_text())
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    _clear_previous()
    _build_armature(document, include_mirrored)
    _build_landmark_empties(document, include_mirrored)
    bone_count = len(bpy.data.armatures[_ARMATURE_NAME].bones)
    empty_count = sum(
        1 for obj in bpy.data.objects if obj.get("standard_human_landmark") is not None
    )
    scope = "full skeleton" if include_mirrored else "authored (unmirrored) only"
    print(
        f"[standard_human] loaded {bone_count} bones, {empty_count} landmarks "
        f"({scope}) from {json_path}"
    )


# ----------------------------------------------------------------------------- #
# read-back: moved editable empties -> local rest_position (paste into YAML)
# ----------------------------------------------------------------------------- #

def _authored_name(landmark_name: str, reference_frame: str) -> str:
    """The name as written in the part YAML, for a paste-ready line.

    A sided part authors a GENERIC name (``hand_capitate``) that the loader sides
    into ``hand_capitate.L`` / ``left_hand_capitate``; strip that back. A midline
    part authors the full name (``right_hip``), so it is returned unchanged.
    """
    if not _is_sided_frame(reference_frame):
        return landmark_name
    for suffix in (".L", ".R"):
        if landmark_name.endswith(suffix):
            return landmark_name[: -len(suffix)]
    for prefix in ("left_", "right_"):
        if landmark_name.startswith(prefix):
            return landmark_name[len(prefix):]
    return landmark_name


def _is_sided_frame(reference_frame: str) -> bool:
    return reference_frame.endswith((".L", ".R")) or reference_frame.startswith(
        ("left_", "right_")
    )


def read_back_to_yaml(json_path: str = JSON_PATH) -> list[str]:
    """Print + return the local rest_position of every moved editable empty, as
    paste-ready YAML lines. Positions only."""
    document = json.loads(Path(json_path).read_text())
    frames = {  # segment name -> (head_mm Vector, basis Matrix rows [x,y,z])
        bone["name"]: (Vector(bone["head"]), Matrix(bone["basis"]))
        for bone in document["bones"]
    }

    lines: list[str] = []
    for name, entry in document["landmarks"].items():
        if not _is_editable(entry["reference_frame"]):
            continue
        empty = bpy.data.objects.get(name)
        if empty is None or empty.get("standard_human_landmark") is None:
            continue
        original = Vector(empty["original_location"])
        moved_mm = (empty.location - original).length / _SCALE
        if moved_mm < _MOVED_EPSILON_MM:
            continue
        head_mm, basis = frames[entry["reference_frame"]]
        world_mm = empty.location / _SCALE
        # world = origin + basis^T @ local  =>  local = basis @ (world - origin)
        local = basis @ (world_mm - head_mm)
        lines.append(
            f"  {_authored_name(name, entry['reference_frame'])}:\n"
            f"    rest_position: [{local.x:.1f}, {local.y:.1f}, {local.z:.1f}]"
            f"    # moved {moved_mm:.1f} mm  (from {name})"
        )

    print("\n# --- paste these rest_position values into the matching part YAML ---")
    print("\n".join(lines) if lines else "# (no editable landmarks moved)")
    return lines


# ----------------------------------------------------------------------------- #
# Blender operators + sidebar panel (View3D > N > "StdHuman")
# ----------------------------------------------------------------------------- #

class STANDARDHUMAN_OT_load(bpy.types.Operator):
    bl_idname = "standard_human.load"
    bl_label = "Load rest pose"
    bl_description = "Rebuild the armature + landmark empties from the JSON"

    def execute(self, context):
        load_rest_pose()
        return {"FINISHED"}


class STANDARDHUMAN_OT_read_back(bpy.types.Operator):
    bl_idname = "standard_human.read_back"
    bl_label = "Read back to YAML"
    bl_description = "Print the local rest_position of every moved editable empty"

    def execute(self, context):
        lines = read_back_to_yaml()
        self.report({"INFO"}, f"{len(lines)} moved landmark(s) -- see the System Console")
        return {"FINISHED"}


class STANDARDHUMAN_PT_panel(bpy.types.Panel):
    bl_label = "StdHuman rest pose"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "StdHuman"

    def draw(self, context):
        self.layout.operator(STANDARDHUMAN_OT_load.bl_idname, icon="ARMATURE_DATA")
        self.layout.prop(context.scene, "standard_human_include_mirrored")
        self.layout.operator(STANDARDHUMAN_OT_read_back.bl_idname, icon="EXPORT")


_CLASSES = (
    STANDARDHUMAN_OT_load,
    STANDARDHUMAN_OT_read_back,
    STANDARDHUMAN_PT_panel,
)


def register() -> None:
    for cls in _CLASSES:
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
        bpy.utils.register_class(cls)
    bpy.types.Scene.standard_human_include_mirrored = bpy.props.BoolProperty(
        name="Include mirrored right side",
        description=(
            "On: the full 95-bone skeleton. Off: only the authored (unmirrored) "
            "data -- the left/midline bones and landmarks actually written in the YAML"
        ),
        default=False,
    )


def unregister() -> None:
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
    if hasattr(bpy.types.Scene, "standard_human_include_mirrored"):
        del bpy.types.Scene.standard_human_include_mirrored


if __name__ == "__main__":
    register()
    load_rest_pose()
