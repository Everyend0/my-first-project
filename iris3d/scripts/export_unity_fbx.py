"""Export Iris (mesh + rig + 1 s loop) as an FBX that Unity plays correctly.

Usage:
    blender -b model/iris.blend --python scripts/export_unity_fbx.py -- \
        --out unity/Assets/IrisViewer/Models/iris_unity.fbx

Why a separate export rig: the Blender rig folds each arm by scaling joints
along their length with "inherit scale = None". Unity transforms always
inherit their parent's scale, so the original chain would shear in Unity.
Here every joint is re-parented directly under the hub and keyed with its
exact world transform (rotation + scale along its own axis), which Unity can
represent. The script checks the deformed mesh against the original rig on
several frames before exporting.

Also adds an empty 'Iris_FrontMarker' in front of the face so the Unity
viewer can find which way the mascot faces after axis conversion.
"""
import argparse
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args(argv)

LOOP = 30
scene = bpy.context.scene
body = bpy.data.objects["Iris_Body"]
rig = bpy.data.objects["Iris_Rig"]


def deformed(frame):
    scene.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    mw = np.array(body.matrix_world)
    co = co.reshape(-1, 3)
    return co @ mw[:3, :3].T + mw[:3, 3]


check_frames = [1, 5, 9, 13, 20, 27]
ref = {f: deformed(f) for f in check_frames}

# ---- flat export rig --------------------------------------------------------
exp_data = rig.data.copy()
exp_data.name = "Iris_Rig_Export"
exp = bpy.data.objects.new("Iris_Rig_Export", exp_data)
scene.collection.objects.link(exp)
exp.matrix_world = rig.matrix_world.copy()
bpy.ops.object.select_all(action="DESELECT")
bpy.context.view_layer.objects.active = exp
exp.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
for eb in exp_data.edit_bones:
    if eb.name != "hub":
        eb.use_connect = False
        eb.parent = exp_data.edit_bones["hub"]
        eb.inherit_scale = "FULL"
bpy.ops.object.mode_set(mode="POSE")
for pb in exp.pose.bones:
    pb.rotation_mode = "QUATERNION"
    c = pb.constraints.new("COPY_TRANSFORMS")
    c.target = rig
    c.subtarget = pb.name
    c.target_space = "WORLD"
    c.owner_space = "WORLD"
scene.frame_start, scene.frame_end = 1, LOOP + 1
bpy.ops.nla.bake(frame_start=1, frame_end=LOOP + 1, only_selected=False,
                 visual_keying=True, clear_constraints=True, use_current_action=False,
                 bake_types={"POSE"})
bpy.ops.object.mode_set(mode="OBJECT")
exp.animation_data.action.name = "IrisBurstLoop"
for fc in exp.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"

# move the mesh onto the export rig
mod = body.modifiers["Armature"]
mod.object = exp
mw = body.matrix_world.copy()
body.parent = exp
body.matrix_world = mw

err = max(float(np.abs(deformed(f) - ref[f]).max()) for f in check_frames)
print(f"EXPORT_RIG_MATCH max vertex error = {err:.2e} m")
assert err < 1e-4, "export rig does not reproduce the animation"

# influences per vertex (Unity default quality uses 4)
infl = np.array([sum(1 for g in v.groups if g.weight > 1e-4) for v in body.data.vertices])
print("influences per vertex: max", int(infl.max()), "share >4:", float((infl > 4).mean()))

# front marker: 1 m in front of the hub, on the face side (-Y in Blender)
hub_w = exp.matrix_world @ exp.data.bones["hub"].head_local
marker = bpy.data.objects.new("Iris_FrontMarker", None)
scene.collection.objects.link(marker)
marker.location = hub_w + Vector((0.0, -1.0, 0.0))
marker.parent = exp
marker.matrix_parent_inverse = exp.matrix_world.inverted()

# remove everything that is not part of the asset
for ob in list(scene.objects):
    if ob not in (exp, body, marker):
        bpy.data.objects.remove(ob, do_unlink=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
scene.frame_set(1)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.fbx(
    filepath=os.path.abspath(a.out),
    use_selection=True,
    object_types={"ARMATURE", "MESH", "EMPTY"},
    use_mesh_modifiers=False,          # keep the rest-pose mesh; Unity skins it
    mesh_smooth_type="FACE",
    add_leaf_bones=False,
    primary_bone_axis="Y",
    secondary_bone_axis="X",
    armature_nodetype="NULL",
    bake_anim=True,
    bake_anim_use_all_bones=True,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False,
    bake_anim_force_startend_keying=True,
    bake_anim_step=1.0,
    bake_anim_simplify_factor=0.0,     # keep every frame exactly
    apply_unit_scale=True,
    apply_scale_options="FBX_SCALE_UNITS",
    axis_forward="-Z",
    axis_up="Y",
    path_mode="STRIP",
    embed_textures=False,
)
print("wrote", a.out, os.path.getsize(a.out), "bytes")
