"""Render frames of the Iris scene.

Usage:
    blender -b model/iris.blend --python scripts/render_iris.py -- \
        --camera CAM_front_tilt --start 1 --end 30 --out build/frames/x [options]

Passes (--pass):
  full    mascot + caught shadows over white in one render (slow: the
          shadow-catching backdrop covers most of the frame)
  char    mascot only, RGBA with transparent background, full resolution
  shadow  backdrop shadows only (mascot invisible to the camera but still
          casting shadows), RGBA where alpha = shadow density. Shadows are
          soft, so this pass is rendered small and upscaled when composited
          (see composite_iris.py); this cuts 4K render time roughly 5x.

Frames are written as <out>/####.png (frame number). Finished frames are
skipped, so an interrupted job can simply be re-run.
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
ap.add_argument("--pass", dest="rpass", choices=("full", "char", "shadow"), default="full")
ap.add_argument("--pct", type=int, default=0, help="resolution %% (default 100, shadow 25)")
ap.add_argument("--samples", type=int, default=0)
ap.add_argument("--rest", action="store_true", help="render the rest pose (no animation)")
ap.add_argument("--no-blur", action="store_true")
a = ap.parse_args(argv)

scene = bpy.context.scene
scene.camera = bpy.data.objects[a.camera]
body = bpy.data.objects["Iris_Body"]
cyc = bpy.data.objects["Cyclorama"]

if a.rpass == "char":
    cyc.hide_render = True
    scene.use_nodes = False
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
elif a.rpass == "shadow":
    body.visible_camera = False
    scene.use_nodes = False
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.use_motion_blur = False
    scene.cycles.samples = 48
    scene.cycles.adaptive_threshold = 0.03

pct = a.pct or (25 if a.rpass == "shadow" else 100)
scene.render.resolution_percentage = pct
if a.samples:
    scene.cycles.samples = a.samples
    scene.cycles.adaptive_min_samples = min(scene.cycles.adaptive_min_samples, a.samples)
if a.no_blur or a.rest:
    scene.render.use_motion_blur = False
if a.rest:
    bpy.data.objects["Iris_Rig"].data.pose_position = "REST"

os.makedirs(a.out, exist_ok=True)
t_all = time.time()
for f in range(a.start, a.end + 1, a.step):
    path = os.path.join(os.path.abspath(a.out), f"{f:04d}.png")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        continue
    scene.frame_set(f)
    scene.render.filepath = path + ".tmp.png"
    t = time.time()
    bpy.ops.render.render(write_still=True)
    os.replace(path + ".tmp.png", path)   # only complete frames get the final name
    print(f"FRAME {a.camera} {a.rpass} {f} {time.time() - t:.1f}s -> {path}", flush=True)
print(f"DONE {a.camera} {a.rpass} {a.start}-{a.end} in {time.time() - t_all:.1f}s", flush=True)
