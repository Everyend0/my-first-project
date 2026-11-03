"""Build the Iris 3D scene in Blender from the traced drawing.

Usage (Blender 4.x, background):
    blender -b --factory-startup --python build_iris.py -- \
        --trace build/iris_trace.npz --meta build/iris_trace.json \
        --blend model/iris.blend --obj model/iris.obj

Creates:
  * Iris_Body   - closed, smooth mesh; yellow body, black outline band and
                  black face strokes are separate materials on one surface
  * Iris_Rig    - hub bone + 8 arms, each a chain of joints ("마디")
  * a 1-second animation loop (30 frames) modelled on the reference GIF:
    joints fold in towards the hub, burst out with a small overshoot,
    wobble joint by joint while extended, then fold back in
  * key/rim sun lights, white world, shadow-catcher cyclorama
  * three cameras: CAM_front_tilt, CAM_front, CAM_orbit
  * OBJ + MTL of the rest pose (exactly the drawing's shape and colours)
"""
import argparse
import json
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Euler, Matrix, Vector

# --------------------------------------------------------------------------
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--blend", required=True)
ap.add_argument("--obj", required=True)
args = ap.parse_args(argv)

SCALE = 500.0          # drawing pixels per Blender unit (metre)
LOOP = 30              # frames per animation loop (1 s at 30 fps)
LOOPS = 3              # loops in the delivered videos
FLOOR_GAP = 0.30       # gap between the lowest arm tip and the floor (m)
# lighting calibrated so lit, non-highlight yellow renders as the drawing's
# #FEDB02 (see verify_iris.py colour check)
KEY_STRENGTH = float(os.environ.get("IRIS_KEY", 1.68))
WORLD_DIFFUSE = float(os.environ.get("IRIS_WORLD", 0.68))

MAT_YELLOW, MAT_OUTLINE, MAT_FACE = 0, 1, 2

meta = json.load(open(args.meta))
data = np.load(args.trace)
xy = data["xy"].astype(np.float64)
hpx = data["h"].astype(np.float64)
tris = data["tris"].astype(np.int64)
tmat = data["mat"].astype(np.int64)
nloop = int(data["nloop"])

hub_px = np.array(meta["hub"])
Z0 = (xy[:, 1].max() - hub_px[1]) / SCALE + FLOOR_GAP   # hub height above floor


def to_world(p_xy, depth_px=None):
    """image px -> world. x right, image y down -> world Z up, depth -> -Y."""
    p_xy = np.atleast_2d(p_xy)
    X = (p_xy[:, 0] - hub_px[0]) / SCALE
    Z = (hub_px[1] - p_xy[:, 1]) / SCALE + Z0
    Y = np.zeros_like(X) if depth_px is None else -np.asarray(depth_px) / SCALE
    return np.stack([X, Y, Z], 1)


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# --------------------------------------------------------------------------
# reset scene
# --------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.name = "Iris"
# mapping back to the drawing (used by verify_iris.py / compare_iris.py)
scene["iris_scale_px_per_m"] = SCALE
scene["iris_hub_px"] = [float(v) for v in hub_px]
scene["iris_Z0"] = float(Z0)
scene["iris_nloop"] = nloop

# --------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------
YELLOW_SRGB = np.array(meta["yellow_srgb"]) / 255.0      # (254, 219, 2)
BLACK_SRGB = np.array([0.0, 0.0, 0.0])


def make_material(name, srgb, rough, coat, coat_rough, spec_level):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    lin = srgb_to_linear(srgb)
    bsdf.inputs["Base Color"].default_value = (*lin, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Specular IOR Level"].default_value = spec_level
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Coat Roughness"].default_value = coat_rough
    m.diffuse_color = (*lin, 1.0)
    m["srgb_hex"] = "#%02X%02X%02X" % tuple(int(round(v * 255)) for v in srgb)
    return m


# Base specular is (almost) off: a broad base lobe veils the whole yellow in
# white and turns the black outline grey. The sharp clear-coat gives the
# visible light reflections instead.
MAT_PARAMS = {
    "Iris_Yellow": dict(rough=0.45, coat=0.35, coat_rough=0.08, spec_level=0.0),
    "Iris_Black": dict(rough=0.30, coat=0.30, coat_rough=0.10, spec_level=0.05),
}
mat_yellow = make_material("Iris_Yellow", YELLOW_SRGB, **MAT_PARAMS["Iris_Yellow"])
mat_black = make_material("Iris_Black", BLACK_SRGB, **MAT_PARAMS["Iris_Black"])

# --------------------------------------------------------------------------
# body mesh: front sheet (faces camera, -Y) + back sheet, glued on the loop
# --------------------------------------------------------------------------
N = len(xy)
inner_idx = np.arange(nloop, N)
front = to_world(xy, hpx)
back = to_world(xy[inner_idx], -hpx[inner_idx])
verts = np.vstack([front, back])
back_index = np.arange(N)
back_index[nloop:] = N + np.arange(len(inner_idx))   # loop verts are shared

# orient front triangles so their normals face -Y (towards the camera)
p0, p1, p2 = front[tris[:, 0]], front[tris[:, 1]], front[tris[:, 2]]
nrm_y = np.cross(p1 - p0, p2 - p0)[:, 1]
flip = nrm_y > 0
ftris = tris.copy()
ftris[flip] = ftris[flip][:, ::-1]
btris = back_index[ftris][:, ::-1]
faces = np.vstack([ftris, btris])
fmat = np.concatenate([tmat, np.where(tmat == MAT_FACE, MAT_YELLOW, tmat)])
slot = np.where(fmat == MAT_YELLOW, 0, 1)  # slot 0 yellow, slot 1 black

mesh = bpy.data.meshes.new("Iris_Body")
mesh.from_pydata(verts.tolist(), [], faces.tolist())
mesh.update()
mesh.materials.append(mat_yellow)
mesh.materials.append(mat_black)
mesh.polygons.foreach_set("material_index", slot.astype(np.int32))
mesh.polygons.foreach_set("use_smooth", np.ones(len(faces), bool))
mesh.validate()
body = bpy.data.objects.new("Iris_Body", mesh)
scene.collection.objects.link(body)

# keep image coordinates per vertex for weighting
vxy = np.vstack([xy, xy[inner_idx]])
v_is_face = np.zeros(len(verts), bool)
face_v = np.unique(tris[tmat == MAT_FACE])
v_is_face[face_v] = True

# --------------------------------------------------------------------------
# rig
# --------------------------------------------------------------------------
arms = meta["arms"]
r_hub = meta["hub_radius"]
R_arm = meta["arm_halfwidth"]
top_k = int(np.argmin([abs(a["angle_deg"] - 90.0) for a in arms]))
COMPASS = {0: "E", 45: "NE", 90: "N", 135: "NW", 180: "W", -180: "W",
           -45: "SE", -90: "S", -135: "SW"}


def compass(angle):
    best = min(COMPASS, key=lambda a: abs((angle - a + 180) % 360 - 180))
    return COMPASS[best]


def polyline_project(P, cl):
    """closest point on polyline cl for each P: returns (dist, arc_s)."""
    a, b = cl[:-1], cl[1:]
    ab = b - a
    seglen = np.linalg.norm(ab, axis=1)
    cum = np.concatenate([[0], np.cumsum(seglen)])
    best_d = np.full(len(P), np.inf)
    best_s = np.zeros(len(P))
    for i in range(len(a)):
        t = np.clip(((P - a[i]) @ ab[i]) / max(seglen[i] ** 2, 1e-9), 0, 1)
        q = a[i] + t[:, None] * ab[i]
        d = np.linalg.norm(P - q, axis=1)
        better = d < best_d
        best_d[better] = d[better]
        best_s[better] = cum[i] + t[better] * seglen[i]
    return best_d, best_s


def point_at(cl, s):
    seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    return np.array([np.interp(s, cum, cl[:, 0]), np.interp(s, cum, cl[:, 1])])


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


S0_HUB = 0.92 * r_hub      # where an arm leaves the hub (arc length from hub)
CAP = 1.05 * R_arm         # rounded tip kept rigid
BLEND_FACE = 11.0          # joint blend half-width on the face arm (px)
BLEND_ARM = 40.0           # wider blends elsewhere: smooth joints, no creases

arm_specs = []
for k, a in enumerate(arms):
    cl = np.array(a["centerline"])
    L = a["length"]
    name = "arm_" + compass(a["angle_deg"])
    # Fold only beyond the point where this arm separates from its closest
    # neighbour, so a folded tip never pushes into the neighbouring arm.
    gap = min(abs((a["angle_deg"] - arms[j]["angle_deg"] + 180) % 360 - 180)
              for j in range(len(arms)) if j != k)
    S0 = max(S0_HUB, R_arm / math.sin(math.radians(gap) / 2))
    face_joint = None
    if k == top_k:
        # the face sits on this arm: one rigid joint covers the whole face
        blend = BLEND_FACE
        fd, fs = polyline_project(xy[face_v], cl)
        f0, f1 = fs.min() - 2.2 * blend, fs.max() + 2.2 * blend
        if f0 - S0 >= 40:
            bounds, rigid = [S0, 0.5 * (S0 + f0), f0], [False, False]
        else:
            bounds, rigid = [S0, f0], [False]
        face_joint = len(rigid)
        if (L - CAP) - f1 >= 25:
            bounds += [f1, L - CAP, L]
            rigid += [True, False, True]
        else:                      # not enough room above the face
            bounds += [L]
            rigid += [True]
    else:
        blend = BLEND_ARM
        free = (L - CAP) - S0
        bounds = [S0, S0 + 0.25 * free, S0 + 0.5 * free, S0 + 0.75 * free, L - CAP, L]
        rigid = [False, False, False, False, True]
    arm_specs.append(dict(name=name, cl=cl, L=L, bounds=bounds, rigid=rigid, S0=S0,
                          blend=blend, angle=a["angle_deg"], face=(k == top_k),
                          face_joint=face_joint))
    print(f"{name:8s} L={L:6.1f} S0={S0:5.1f} bounds={[round(b) for b in bounds]} rigid={rigid}")

arm_data = bpy.data.armatures.new("Iris_Rig")
arm_data.display_type = "STICK"
rig = bpy.data.objects.new("Iris_Rig", arm_data)
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
eb = arm_data.edit_bones
hub_w = Vector(to_world(hub_px)[0])
root = eb.new("hub")
root.head = hub_w
root.tail = hub_w + Vector((0, 0.0, 0.15))
root.align_roll(Vector((0, -1, 0)))
bone_names = []   # per arm list of bone names
for spec in arm_specs:
    names = []
    parent = root
    for j in range(len(spec["rigid"])):
        b = eb.new(f"{spec['name']}.{j + 1}")
        b.head = Vector(to_world(point_at(spec["cl"], spec["bounds"][j]))[0])
        b.tail = Vector(to_world(point_at(spec["cl"], spec["bounds"][j + 1]))[0])
        b.align_roll(Vector((0, -1, 0)))   # local Z towards the camera
        b.parent = parent
        b.use_connect = j > 0
        b.inherit_scale = "NONE" if j > 0 else "FULL"
        b.use_deform = True
        parent = b
        names.append(b.name)
    bone_names.append(names)
bpy.ops.object.mode_set(mode="OBJECT")

# ---- weights ---------------------------------------------------------------
NV = len(verts)
dists, arcs = [], []
for spec in arm_specs:
    d, s = polyline_project(vxy, spec["cl"])
    dists.append(d)
    arcs.append(s)
dists = np.array(dists)       # (8, NV)
arcs = np.array(arcs)
# soft arm membership (only matters close to the bisector between two arms)
dmin = dists.min(0)
member = np.exp(-((dists - dmin) / 14.0) ** 2)
member /= member.sum(0)
weights = {"hub": np.zeros(NV)}
for k, spec in enumerate(arm_specs):
    s = arcs[k]
    S0k, bl = spec["S0"], spec["blend"]
    # the root up to S0 (incl. the web between neighbouring arms) stays on the hub
    w_arm = smoothstep(S0k - 1.6 * BLEND_FACE, S0k + 0.6 * BLEND_FACE, s) * member[k]
    b = spec["bounds"]
    nb = len(spec["rigid"])
    # telescoping partition of unity along the arm: w_j = c_(j-1) - c_j, where
    # c_j = smoothstep across cut j (always sums to exactly 1, even where two
    # wide joint blends overlap)
    cuts = b[1:-1]
    c = [np.ones(NV)] + [smoothstep(cut - bl, cut + bl, s) for cut in cuts] + [np.zeros(NV)]
    for j in range(1, len(c) - 1):          # keep c non-increasing along j
        c[j] = np.minimum(c[j], c[j - 1])
    for j in range(nb):
        weights[bone_names[k][j]] = (c[j] - c[j + 1]) * w_arm
total_arm = sum(v for kname, v in weights.items() if kname != "hub")
weights["hub"] = np.clip(1.0 - total_arm, 0, 1)

# the face must move rigidly: give face vertices exactly the face-joint weight
top = arm_specs[top_k]
face_bone = bone_names[top_k][top["face_joint"]]
for kname in weights:
    weights[kname][v_is_face] = 0.0
weights[face_bone][v_is_face] = 1.0
rig["face_bone"] = face_bone

for kname, w in weights.items():
    vg = body.vertex_groups.new(name=kname)
    idx = np.nonzero(w > 1e-4)[0]
    for wv in np.unique(np.round(w[idx], 4)):
        sel = idx[np.round(w[idx], 4) == wv]
        vg.add(sel.tolist(), float(wv), "REPLACE")

mod = body.modifiers.new("Armature", "ARMATURE")
mod.object = rig
body.parent = rig
print("weights: max sum dev",
      float(np.abs(sum(weights.values()) - 1).max()))

# --------------------------------------------------------------------------
# animation (1-second loop modelled on the reference GIF)
# --------------------------------------------------------------------------
scene.render.fps = 30
scene.frame_start = 1
scene.frame_end = LOOP * LOOPS

# extension curve e(tau), tau in [0,1): folded (hold) -> burst out with
# overshoot -> settle and wobble -> fold back in, after the reference GIF
# (blob ~130 ms, burst to a peak at ~400 ms, settle/jitter, retract ~200 ms).
# Every joint runs the same curve with a delay, hub side first, so with the
# delays below all joints are folded together for ~5 frames around frame 1
# and fully out together for ~4 frames mid-loop.
# n(tau) is normalised: 0 = fully folded, 1 = the drawing's length.
KEYS = [  # (tau, n, easing into this key)
    (0.000, 0.000, None),
    (0.120, 0.000, "smooth"),
    (0.360, 1.070, "out"),
    (0.460, 0.970, "smooth"),
    (0.540, 1.015, "smooth"),
    (0.620, 1.000, "smooth"),
    (0.660, 1.000, "smooth"),
    (0.800, 0.000, "in"),
    (1.000, 0.000, "smooth"),
]
# Graded fold: the smallest scale of each compressible joint, hub side first.
# Linear-blend skinning across a joint from scale a to scale b squeezes the
# skin down to about min(a, b) - 0.207*|a - b| (smoothstep blend), so neighbouring
# joints must not differ too much or the skin creases into a ring (worst next
# to the rigid tip and the hub). These keep the squeeze above ~0.2 everywhere
# while the arm still folds to about 35 % of its free length.
MIN_SCALE = [0.45, 0.25, 0.25, 0.50]
MIN_SCALE_FACE_ARM = [0.45, 0.30, 0.50]   # below the face x2, above the face
JOINT_DELAY = 0.040        # per joint (1.2 frames): the "마디마디" wave
ARM_DELAY = [0.000, 0.021, 0.009, 0.030, 0.004, 0.026, 0.013, 0.017]


def ease(kind, t):
    if kind == "out":
        return 1 - (1 - t) ** 3
    if kind == "in":
        return t ** 3 * 0.6 + t * t * 0.4
    return t * t * (3 - 2 * t)


def extension(tau):
    tau %= 1.0
    for (t0, v0, _), (t1, v1, kind) in zip(KEYS, KEYS[1:]):
        if t0 <= tau <= t1:
            return v0 + (v1 - v0) * ease(kind, (tau - t0) / (t1 - t0))
    return 0.0


def bump(tau, centre, width):
    d = (tau - centre + 0.5) % 1.0 - 0.5
    return 0.5 * (1 + math.cos(math.pi * min(abs(d) / width, 1.0)))


pose = rig.pose
for pb in pose.bones:
    pb.rotation_mode = "XYZ"

for k, spec in enumerate(arm_specs):
    names = bone_names[k]
    comp = [j for j, r in enumerate(spec["rigid"]) if not r]
    if spec["face"]:
        # by role: joints below the face get the first values, the joint
        # above the face (if any) the last one
        below = [j for j in comp if j < spec["face_joint"]]
        above = [j for j in comp if j > spec["face_joint"]]
        mins = dict(zip(below, MIN_SCALE_FACE_ARM[:-1]))
        mins.update({j: MIN_SCALE_FACE_ARM[-1] for j in above})
    else:
        mins = dict(zip(comp, MIN_SCALE))
    for f in range(1, LOOP + 2):
        tau = (f - 1) / LOOP
        for j, bn in enumerate(names):
            pb = pose.bones[bn]
            # joints unfold one after another, hub side first ("마디마디")
            delay = ARM_DELAY[k] + JOINT_DELAY * j
            if spec["rigid"][j]:
                sy = 1.0
            else:
                n = extension(tau - delay)
                sy = mins[j] + n * (1.0 - mins[j])
            pb.scale = (1.0, sy, 1.0)
            # joint wobble: a travelling wave along the arm, strongest while
            # the arm is out (like the jitter in the GIF's held frames)
            env = 0.30 + 0.70 * bump(tau, 0.58, 0.30)
            phase = 2 * math.pi * (k * 0.137 + 0.11 * j)
            rz = math.radians(3.2) * env * math.sin(2 * math.pi * 3 * tau - phase)
            rx = math.radians(1.6) * env * math.sin(2 * math.pi * 2 * tau - phase * 1.3 + 0.7)
            if j == 0:
                rz *= 0.5
                rx *= 0.5
            pb.rotation_euler = (rx, 0.0, rz)
            pb.keyframe_insert("scale", frame=f)
            pb.keyframe_insert("rotation_euler", frame=f)

def action_fcurves(ob):
    """F-curves of an object's action (Blender 4.x and the 4.4+/5.x slotted API)."""
    act_ = ob.animation_data.action
    if hasattr(act_, "fcurves"):
        return list(act_.fcurves)
    from bpy_extras import anim_utils
    bag = anim_utils.action_get_channelbag_for_slot(act_, ob.animation_data.action_slot)
    return list(bag.fcurves) if bag else []


rig.animation_data.action.name = "Iris_Burst_Loop"
for fc in action_fcurves(rig):
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"
    fc.modifiers.new("CYCLES")

# --------------------------------------------------------------------------
# world, lights, cyclorama (shadow catcher)
# --------------------------------------------------------------------------
# Diffuse light comes from a uniform white sky, so the yellow keeps its exact
# colour. Reflections (glossy rays) instead see a dark studio with one soft
# overhead softbox: highlights read as real light reflections without washing
# the whole surface out to white.
world = bpy.data.worlds.new("Studio")
world.use_nodes = True
wn = world.node_tree.nodes
wl = world.node_tree.links
bg = wn["Background"]
lp = wn.new("ShaderNodeLightPath")
tex = wn.new("ShaderNodeTexCoord")
sep = wn.new("ShaderNodeSeparateXYZ")
ramp = wn.new("ShaderNodeValToRGB")          # softbox: bright cap overhead
ramp.color_ramp.elements[0].position = 0.55
ramp.color_ramp.elements[0].color = (0.012, 0.012, 0.012, 1)
ramp.color_ramp.elements[1].position = 0.80
ramp.color_ramp.elements[1].color = (1.6, 1.6, 1.6, 1)
mix = wn.new("ShaderNodeMix")
mix.data_type = "RGBA"
mix.inputs["A"].default_value = (1, 1, 1, 1)
wl.new(tex.outputs["Generated"], sep.inputs["Vector"])   # world: ray direction
wl.new(sep.outputs["Z"], ramp.inputs["Fac"])
wl.new(lp.outputs["Is Glossy Ray"], mix.inputs["Factor"])
wl.new(ramp.outputs["Color"], mix.inputs["B"])
wl.new(mix.outputs["Result"], bg.inputs["Color"])
bg.inputs["Strength"].default_value = WORLD_DIFFUSE
scene.world = world


def add_sun(name, strength, angle_deg, rot_deg):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy = strength
    ld.angle = math.radians(angle_deg)
    ob = bpy.data.objects.new(name, ld)
    ob.rotation_euler = Euler([math.radians(v) for v in rot_deg], "XYZ")
    scene.collection.objects.link(ob)
    return ob


# key light from upper-front-left, rim light from upper-back-left
key = add_sun("Key_Sun", KEY_STRENGTH, 6.0, (52.0, 0.0, -28.0))
rim = add_sun("Rim_Sun", 0.9, 3.0, (-55.0, 0.0, 35.0))
rim.data.use_shadow = False

# cyclorama: floor + rounded back wall, shadow catcher
bm = bmesh.new()
WALL_Y, RAD, ZTOP, XW, YFRONT = 0.95, 0.9, 6.0, 9.0, -8.0
prof = [(YFRONT, 0.0), (WALL_Y - RAD, 0.0)]
for i in range(1, 17):
    a = (math.pi / 2) * i / 16
    prof.append((WALL_Y - RAD + RAD * math.sin(a), RAD - RAD * math.cos(a)))
prof.append((WALL_Y, ZTOP))
cols = []
for x in (-XW, XW):
    cols.append([bm.verts.new((x, y, z)) for (y, z) in prof])
for i in range(len(prof) - 1):
    bm.faces.new((cols[0][i], cols[1][i], cols[1][i + 1], cols[0][i + 1]))
cyc_mesh = bpy.data.meshes.new("Cyclorama")
bm.to_mesh(cyc_mesh)
bm.free()
cyc_mesh.polygons.foreach_set("use_smooth", np.ones(len(cyc_mesh.polygons), bool))
cyc = bpy.data.objects.new("Cyclorama", cyc_mesh)
scene.collection.objects.link(cyc)
cyc.is_shadow_catcher = True
# the backdrop only catches shadows; it must not bounce light or show up in
# reflections (that would tint the mascot)
cyc.visible_diffuse = False
cyc.visible_glossy = False
cyc.visible_transmission = False
cyc.visible_volume_scatter = False
mat_cyc = bpy.data.materials.new("Cyc_White")
mat_cyc.use_nodes = True
mat_cyc.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1)
mat_cyc.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 1.0
cyc_mesh.materials.append(mat_cyc)

# --------------------------------------------------------------------------
# cameras
# --------------------------------------------------------------------------
xmin, xmax = verts[:, 0].min(), verts[:, 0].max()
zmin, zmax = verts[:, 2].min(), verts[:, 2].max()
cx, cz = 0.5 * (xmin + xmax), 0.5 * (zmin + zmax)
W_obj, H_obj = xmax - xmin, zmax - zmin
ASPECT = 16 / 9
FRAME_H = max(H_obj * 1.42, W_obj * 1.30 / ASPECT)   # room for the overshoot
target = Vector((cx, 0.0, cz - 0.04))


def look_at(ob, tgt):
    d = tgt - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def add_persp_cam(name, elev_deg, lens=85.0):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36.0
    ob = bpy.data.objects.new(name, cd)
    scene.collection.objects.link(ob)
    vfov = 2 * math.atan((36.0 / ASPECT) / 2 / lens)
    dist = (FRAME_H / 2) / math.tan(vfov / 2) + 0.15
    e = math.radians(elev_deg)
    ob.location = target + Vector((0, -dist * math.cos(e), dist * math.sin(e)))
    look_at(ob, target)
    cd.clip_end = 100
    return ob, dist


cam_tilt, _ = add_persp_cam("CAM_front_tilt", 15.0)

cd = bpy.data.cameras.new("CAM_front")
cd.type = "ORTHO"
cd.ortho_scale = FRAME_H * ASPECT
cam_front = bpy.data.objects.new("CAM_front", cd)
scene.collection.objects.link(cam_front)
cam_front.location = Vector((cx, -10.0, cz - 0.04))
cam_front.rotation_euler = Euler((math.radians(90), 0, 0), "XYZ")
cd.clip_end = 100

# orbit: camera on a pivot that sways +-20 degrees over the 3-second clip
pivot = bpy.data.objects.new("CAM_orbit_pivot", None)
scene.collection.objects.link(pivot)
pivot.location = target
cam_orbit, dist = add_persp_cam("CAM_orbit", 9.0)
# matrix_world is only refreshed by a depsgraph update; without this the
# camera would be parented with identity matrices and end up inside the hub
bpy.context.view_layer.update()
mw = cam_orbit.matrix_world.copy()
cam_orbit.parent = pivot
cam_orbit.matrix_parent_inverse = pivot.matrix_world.inverted()
cam_orbit.matrix_world = mw
bpy.context.view_layer.update()
assert (cam_orbit.matrix_world.translation - target).length > 1.0, "orbit camera misplaced"
pivot.rotation_mode = "XYZ"
TOTAL = LOOP * LOOPS
for f in range(1, TOTAL + 2):
    yaw = math.radians(20.0) * math.sin(2 * math.pi * (f - 1) / TOTAL)
    pivot.rotation_euler = (0.0, 0.0, yaw)
    pivot.keyframe_insert("rotation_euler", frame=f)
for fc in action_fcurves(pivot):
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"

scene.camera = cam_tilt

# --------------------------------------------------------------------------
# render settings (Cycles CPU; this build has no OIDN, so no denoiser)
# --------------------------------------------------------------------------
scene.render.engine = "CYCLES"
cy = scene.cycles
cy.device = "CPU"
cy.use_denoising = False
cy.use_adaptive_sampling = True
cy.adaptive_threshold = 0.02
cy.adaptive_min_samples = 16
cy.samples = 128
cy.max_bounces = 4
cy.diffuse_bounces = 2
cy.glossy_bounces = 2
cy.transmission_bounces = 0
cy.transparent_max_bounces = 4
cy.caustics_reflective = False
cy.caustics_refractive = False
cy.sample_clamp_indirect = 2.0
cy.filter_width = 1.5
scene.render.use_persistent_data = True
scene.render.film_transparent = True
scene.render.use_motion_blur = True
scene.render.motion_blur_shutter = 0.35
scene.render.resolution_x = 3840
scene.render.resolution_y = 2160
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "8"
scene.render.image_settings.compression = 15
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.display_settings.display_device = "sRGB"

# compositor: put the transparent render (with caught shadows) over white
scene.use_nodes = True
nt = scene.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
rl = nt.nodes.new("CompositorNodeRLayers")
over = nt.nodes.new("CompositorNodeAlphaOver")
over.inputs[1].default_value = (1, 1, 1, 1)
comp = nt.nodes.new("CompositorNodeComposite")
comp.use_alpha = False
nt.links.new(rl.outputs["Image"], over.inputs[2])
nt.links.new(over.outputs["Image"], comp.inputs["Image"])

# --------------------------------------------------------------------------
# save + export the rest pose as OBJ/MTL
# --------------------------------------------------------------------------
scene.frame_set(1)
os.makedirs(os.path.dirname(os.path.abspath(args.blend)), exist_ok=True)
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.blend), compress=True)

arm_data.pose_position = "REST"
# OBJ placement: sit the model on y = 0 (the scene lifts it FLOOR_GAP above
# the floor); x = 0 stays at the hub, the centre of the star
rig.location.z -= FLOOR_GAP
bpy.context.view_layer.update()
bpy.ops.object.select_all(action="DESELECT")
body.select_set(True)
bpy.context.view_layer.objects.active = body
obj_path = os.path.abspath(args.obj)
bpy.ops.wm.obj_export(
    filepath=obj_path,
    export_selected_objects=True,
    apply_modifiers=False,
    export_uv=False,
    export_normals=True,
    export_materials=True,
    export_triangulated_mesh=True,
    export_vertex_groups=False,
    path_mode="RELATIVE",
    forward_axis="NEGATIVE_Z",
    up_axis="Y",
)
rig.location.z += FLOOR_GAP
arm_data.pose_position = "POSE"

# Blender writes linear Kd; most OBJ viewers (three.js, Maya, 3ds Max, Windows
# 3D Viewer, Unity) read Kd as an sRGB colour, so write the MTL ourselves with
# the drawing's exact colours and the same shading values as the .blend.
mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
with open(mtl_path, "w") as f:
    f.write("# Iris mascot materials\n")
    f.write("# Kd holds the exact sRGB colours of the original drawing:\n")
    f.write("#   Iris_Yellow = #FEDB02 (254, 219, 2)   Iris_Black = #000000\n")
    f.write("# Blender's OBJ importer reads Kd as linear, so in Blender open\n")
    f.write("# model/iris.blend instead (or set Base Color hex to FEDB02 after import).\n")
    f.write("# Pr/Pm/Pc/Pcr: PBR extension (roughness, metallic, clearcoat, clearcoat roughness).\n\n")
    for name, srgb, p in (("Iris_Yellow", YELLOW_SRGB, MAT_PARAMS["Iris_Yellow"]),
                          ("Iris_Black", BLACK_SRGB, MAT_PARAMS["Iris_Black"])):
        ks = 0.08 * p["spec_level"] / 0.5 + 0.25 * p["coat"]   # base F0 + coat, roughly
        ns = 2.0 / max(p["coat_rough"], 0.02) ** 2 - 2.0        # sharp coat highlight
        f.write(f"newmtl {name}\n")
        f.write("Ka 0.000000 0.000000 0.000000\n")
        f.write("Kd %.6f %.6f %.6f\n" % tuple(srgb))
        f.write("Ks %.6f %.6f %.6f\n" % (ks, ks, ks))
        f.write("Ke 0.000000 0.000000 0.000000\n")
        f.write(f"Ns {min(ns, 1000.0):.6f}\n")
        f.write("Ni 1.500000\n")
        f.write("d 1.000000\n")
        f.write("illum 2\n")
        f.write(f"Pr {p['rough']:.6f}\nPm 0.000000\nPc {p['coat']:.6f}\nPcr {p['coat_rough']:.6f}\n\n")
print("saved", args.blend, obj_path, mtl_path)
print("verts", len(verts), "faces", len(faces), "Z0", Z0,
      "bbox x", xmin, xmax, "z", zmin, zmax)
