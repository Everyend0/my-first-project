"""Render frames of the Iris scene.

Usage:
    blender -b model/iris.blend --python scripts/render_iris.py -- \
        --camera CAM_front_tilt --start 1 --end 30 --out renders/frames/front_tilt \
        [--pct 100] [--samples 160] [--rest] [--no-blur] [--transparent]

Frames are written as <out>/####.png (frame number). Already rendered frames
are skipped, so an interrupted job can simply be re-run.
"""
import argparse
import os
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--camera", required=True)
ap.add_argument("--start", type=int, default=1)
ap.add_argument("--end", type=int, default=30)
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--pct", type=int, default=100)
ap.add_argument("--samples", type=int, default=0)
ap.add_argument("--rest", action="store_true", help="render the rest pose (no animation)")
ap.add_argument("--no-blur", action="store_true")
ap.add_argument("--transparent", action="store_true", help="RGBA output, no white backdrop")
ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args(argv)

scene = bpy.context.scene
scene.camera = bpy.data.objects[a.camera]
scene.render.resolution_percentage = a.pct
if a.samples:
    scene.cycles.samples = a.samples
    scene.cycles.adaptive_min_samples = min(scene.cycles.adaptive_min_samples, a.samples)
if a.no_blur or a.rest:
    scene.render.use_motion_blur = False
if a.rest:
    bpy.data.objects["Iris_Rig"].data.pose_position = "REST"
if a.threads:
    scene.render.threads_mode = "FIXED"
    scene.render.threads = a.threads
if a.transparent:
    scene.use_nodes = False
    scene.render.image_settings.color_mode = "RGBA"

os.makedirs(a.out, exist_ok=True)
t_all = time.time()
for f in range(a.start, a.end + 1, a.step):
    path = os.path.join(os.path.abspath(a.out), f"{f:04d}.png")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        continue
    scene.frame_set(f)
    scene.render.filepath = path
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"FRAME {a.camera} {f} {time.time() - t:.1f}s -> {path}", flush=True)
print(f"DONE {a.camera} {a.start}-{a.end} in {time.time() - t_all:.1f}s", flush=True)
