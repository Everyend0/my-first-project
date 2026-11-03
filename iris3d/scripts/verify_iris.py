"""Blender-side checks for the Iris scene. Writes a JSON report and a 1:1
front render (same pixel scale as the drawing) for compare_iris.py.

Usage:
    blender -b model/iris.blend --python scripts/verify_iris.py -- \
        --out build/verify [--samples 32] [--no-render]

Checks:
  * loop: mesh at frame f equals mesh at frame f+30 (seamless 1 s loop)
  * face rigidity: pairwise distances between face vertices never change
  * rest pose == traced drawing (deformed mesh at rest matches base mesh)
  * extension range per arm over one loop
  * 1:1 orthographic front render of the rest pose, backdrop hidden,
    transparent film -> silhouette/outline/face comparison in compare_iris.py
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--samples", type=int, default=32)
ap.add_argument("--no-render", action="store_true")
a = ap.parse_args(argv)
os.makedirs(a.out, exist_ok=True)

SCALE = 500.0
scene = bpy.context.scene
body = bpy.data.objects["Iris_Body"]
rig = bpy.data.objects["Iris_Rig"]


def verts_at(frame):
    scene.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    co = co.reshape(-1, 3)
    mw = np.array(body.matrix_world)
    return co @ mw[:3, :3].T + mw[:3, 3]


report = {}
base = np.empty(len(body.data.vertices) * 3)
body.data.vertices.foreach_get("co", base)
base = base.reshape(-1, 3)
mw = np.array(body.matrix_world)
base_w = base @ mw[:3, :3].T + mw[:3, 3]

# ---- face vertices (front faces with black material inside the yellow) ---
me = body.data
# face strokes = black-material vertices fully weighted to the face joint
# (the rigid joint that carries the eyes and smile; stored by build_iris.py)
face_bone = rig["face_bone"]
gi = body.vertex_groups[face_bone].index
face_v = [v.index for v in me.vertices
          if any(g.group == gi and g.weight > 0.9999 for g in v.groups)]
black_v = set()
for p in me.polygons:
    if p.material_index == 1:
        black_v.update(p.vertices)
face_strokes = np.array(sorted(set(face_v) & black_v))
report["face_stroke_vertices"] = int(len(face_strokes))

# ---- rest pose equals base mesh -------------------------------------------
rig.data.pose_position = "REST"
rest = verts_at(1)
report["rest_vs_base_max_err_m"] = float(np.abs(rest - base_w).max())
rig.data.pose_position = "POSE"

# ---- loop seamlessness / rigidity / range ----------------------------------
loop_err, rigid_err = 0.0, 0.0
rng = np.random.default_rng(0)
sel = face_strokes[rng.choice(len(face_strokes), min(300, len(face_strokes)), replace=False)]
ref = None
frames = {}
for f in range(1, 31):
    v = verts_at(f)
    frames[f] = v
    pv = v[sel]
    d = np.linalg.norm(pv[:, None] - pv[None], axis=2)
    if ref is None:
        ref_rest = rest[sel]
        ref = np.linalg.norm(ref_rest[:, None] - ref_rest[None], axis=2)
    rigid_err = max(rigid_err, float(np.abs(d - ref).max()))
for f in (1, 7, 16, 25):
    loop_err = max(loop_err, float(np.abs(verts_at(f + 30) - frames[f]).max()),
                   float(np.abs(verts_at(f + 60) - frames[f]).max()))
report["loop_seam_max_err_m"] = loop_err
report["face_rigidity_max_err_m"] = rigid_err
report["face_rigidity_max_err_px"] = rigid_err * SCALE

# silhouette extent per frame (how far the arms reach), relative to rest
hub = np.array(rig.pose.bones["hub"].head) @ np.array(rig.matrix_world)[:3, :3].T + \
    np.array(rig.matrix_world)[:3, 3]
r_rest = np.linalg.norm((rest - hub)[:, [0, 2]], axis=1).max()
ext = {f: float(np.linalg.norm((frames[f] - hub)[:, [0, 2]], axis=1).max() / r_rest)
       for f in frames}
report["reach_vs_rest"] = ext
report["reach_min"] = min(ext.values())
report["reach_max"] = max(ext.values())
nan = any(not np.isfinite(v).all() for v in frames.values())
report["nan"] = bool(nan)

# ---- folding quality: front triangles that flip, outline loop per frame -----
# front sheet = first len(trace xy) vertices; its triangles are the first half
nfront_tris = len(me.polygons) // 2
tri = np.empty(len(me.polygons) * 3, np.int64)
me.polygons.foreach_get("vertices", tri)
tri = tri.reshape(-1, 3)[:nfront_tris]


def signed_xz_area(v):
    a, b, c = v[tri[:, 0]][:, [0, 2]], v[tri[:, 1]][:, [0, 2]], v[tri[:, 2]][:, [0, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


a_rest = signed_xz_area(rest)
flips = {f: int((np.sign(signed_xz_area(frames[f])) != np.sign(a_rest)).sum()) for f in frames}
report["flipped_front_tris_per_frame"] = flips
report["flipped_front_tris_max"] = max(flips.values())
nloop = int(scene.get("iris_nloop", 0))
if nloop:
    np.savez_compressed(os.path.join(a.out, "loop_frames.npz"),
                        rest=rest[:nloop][:, [0, 2]],
                        **{f"f{f}": frames[f][:nloop][:, [0, 2]] for f in frames})

# ---- cameras: every camera looks at the mascot from outside -------------------
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402

cams = {}
for cam_name in ("CAM_front_tilt", "CAM_front", "CAM_orbit"):
    cam_ob = bpy.data.objects[cam_name]
    worst = dict(min_depth=1e9, min_xy=1e9, max_xy=-1e9)
    for f in (1, 12, 23, 46, 69, 90):
        v = verts_at(f)
        scene.frame_set(f)
        pts = [world_to_camera_view(scene, cam_ob, Vector(p)) for p in v[::97]]
        uvz = np.array([[p.x, p.y, p.z] for p in pts])
        worst["min_depth"] = min(worst["min_depth"], float(uvz[:, 2].min()))
        worst["min_xy"] = min(worst["min_xy"], float(uvz[:, :2].min()))
        worst["max_xy"] = max(worst["max_xy"], float(uvz[:, :2].max()))
    worst["ok"] = worst["min_depth"] > 0.5 and worst["min_xy"] > 0.0 and worst["max_xy"] < 1.0
    cams[cam_name] = worst
report["cameras"] = cams

# ---- 1:1 front render ---------------------------------------------------------
cam = bpy.data.objects["CAM_front"]
report["cam_front"] = dict(location=list(cam.location), ortho_scale=cam.data.ortho_scale)
report["mapping"] = dict(scale=float(scene["iris_scale_px_per_m"]),
                         hub_px=list(scene["iris_hub_px"]), Z0=float(scene["iris_Z0"]))
if not a.no_render:
    scene.camera = cam
    rig.data.pose_position = "REST"
    scene.frame_set(1)
    bpy.data.objects["Cyclorama"].hide_render = True
    scene.render.film_transparent = True
    scene.use_nodes = False
    scene.render.use_motion_blur = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_percentage = 100
    res_x = int(round(cam.data.ortho_scale * SCALE))
    res_y = int(round(res_x * 9 / 16))
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.cycles.samples = a.samples
    scene.cycles.use_adaptive_sampling = False
    scene.render.filepath = os.path.join(os.path.abspath(a.out), "front_1to1.png")
    bpy.ops.render.render(write_still=True)
    report["front_render"] = dict(path=scene.render.filepath, res=[res_x, res_y])
    rig.data.pose_position = "POSE"

with open(os.path.join(a.out, "blender_checks.json"), "w") as f:
    json.dump(report, f, indent=1)
print("VERIFY", json.dumps({k: v for k, v in report.items() if k != "reach_vs_rest"}, indent=1))
