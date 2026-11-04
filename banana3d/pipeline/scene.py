"""Step 3 - assemble the Blender scene, export OBJ/MTL, render the turnaround.

Run:  python3 pipeline/scene.py <name> [--samples N] [--res PX] [--views 0,45,...]
                                        [--no-export] [--ortho-check]
"""
import argparse
import json
import math
import os

import bpy
import numpy as np
from mathutils import Vector
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIGHT = float(os.environ.get("BANANA_LIGHT", "0.2"))
PX = 0.002  # metres per drawing pixel  (character ~1.5-1.7 m tall)


def srgb_to_lin(c):
    c = np.asarray(c, float) / 255.0
    return tuple(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


YELLOW = srgb_to_lin((254, 241, 10))
BLACK = (0.006, 0.006, 0.006)
BACKDROP = srgb_to_lin((222, 222, 222))


# ----------------------------------------------------------------------------- materials
def principled(name, color, rough, coat=0.0, coat_rough=0.05, spec=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*color, 1.0)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Specular IOR Level"].default_value = spec
    p.inputs["Coat Weight"].default_value = coat
    p.inputs["Coat Roughness"].default_value = coat_rough
    m.diffuse_color = (*color, 1.0)
    return m


def outline_material():
    """Inverted-hull ink: the shell's normals point inward, so the side facing the
    camera is a back face (made transparent) and only the rim behind the body
    shows - a solid black line around every silhouette, from any angle."""
    m = bpy.data.materials.new("Outline_Ink")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    mix = nt.nodes.new("ShaderNodeMixShader")
    ink = nt.nodes.new("ShaderNodeEmission")
    ink.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    ink.inputs["Strength"].default_value = 1.0
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(geo.outputs["Backfacing"], mix.inputs["Fac"])
    nt.links.new(ink.outputs["Emission"], mix.inputs[1])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    m.diffuse_color = (0, 0, 0, 1)
    m.use_backface_culling = True  # viewport / EEVEE use plain culling
    return m


# ----------------------------------------------------------------------------- meshes
def make_mesh(name, v, f, mat, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(v.tolist(), [], f.tolist())
    me.validate()
    me.update()
    if smooth:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.data.materials.append(mat)
    return ob


def signed_volume(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def decimate(ob, ratio):
    if ratio >= 0.999:
        return ob
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio
    dg = bpy.context.evaluated_depsgraph_get()
    new = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    old = ob.data
    ob.modifiers.remove(mod)
    ob.data = new
    bpy.data.meshes.remove(old)
    new.shade_smooth()
    return ob


def corner_normals(ob):
    me = ob.data
    cn = np.empty(len(me.loops) * 3)
    me.corner_normals.foreach_get("vector", cn)
    return cn.reshape(-1, 3, 3)


def mesh_arrays(ob):
    me = ob.data
    v = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", v)
    f = np.empty(len(me.polygons) * 3, np.int64)
    me.polygons.foreach_get("vertices", f)
    n = np.empty(len(me.vertices) * 3)
    me.vertex_normals.foreach_get("vector", n)
    return v.reshape(-1, 3), f.reshape(-1, 3), n.reshape(-1, 3)


def value_noise(p, seed):
    """smooth 3D value noise in [0, 1]"""
    pi = np.floor(p).astype(np.int64)
    t = p - pi
    t = t * t * (3 - 2 * t)

    def h(ix, iy, iz):
        x = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ seed
        x = (x ^ (x >> 13)) * 1274126177
        x = x ^ (x >> 16)
        return (x & 0xFFFF) / 65535.0

    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) * (t[:, 2] if dz else 1 - t[:, 2])
                out = out + w * h(pi[:, 0] + dx, pi[:, 1] + dy, pi[:, 2] + dz)
    return out


# ----------------------------------------------------------------------------- studio
def cyclorama(mat):
    """floor that curves up into a back wall (no visible horizon)"""
    prof = []
    for y in np.linspace(-9, 1.2, 12):
        prof.append((y, 0.0))
    R = 2.2
    for a in np.linspace(0, math.pi / 2, 24)[1:]:
        prof.append((1.2 + R * math.sin(a), R - R * math.cos(a)))
    for z in np.linspace(R, 9, 8)[1:]:
        prof.append((1.2 + R, z))
    xs = np.linspace(-12, 12, 13)
    verts, faces = [], []
    for x in xs:
        for (y, z) in prof:
            verts.append((x, y, z))
    n = len(prof)
    for i in range(len(xs) - 1):
        for j in range(n - 1):
            a = i * n + j
            faces.append((a, a + n, a + n + 1, a + 1))
    me = bpy.data.meshes.new("Cyclorama")
    me.from_pydata(verts, [], faces)
    me.shade_smooth()
    ob = bpy.data.objects.new("Cyclorama", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.data.materials.append(mat)
    return ob


def area_light(name, loc, target, size, energy, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = "DISK"
    ld.size = size
    ld.energy = energy
    ld.color = color
    ob = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    d = Vector(target) - Vector(loc)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return ob


# ----------------------------------------------------------------------------- OBJ / MTL
def write_obj(path, groups, mats):
    """groups: list of (object_name, material_name, verts(Blender coords), faces, corner normals (F,3,3))
    OBJ convention: +Y up, front of the character faces +Z."""
    base = os.path.splitext(os.path.basename(path))[0]
    mtl = base + ".mtl"
    with open(path, "w") as fh:
        fh.write(f"# {base} - 3D model built from the original drawing\n")
        fh.write("# units: metres, +Y up, character faces +Z\n")
        fh.write("# objects 'Outline_Ink_*' = toon ink shell (inverted hull, inside-out on purpose).\n")
        fh.write("# They need back-face culling ON (default in game engines / most web viewers);\n")
        fh.write("# never give them a collider. Delete them for a plain model.\n")
        fh.write(f"mtllib {mtl}\n")
        voff, noff = 1, 1
        for name, mat, v, f, cn in groups:
            vo = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)
            # drop faces that collapse once written (zero area)
            q = np.round(vo, 6)
            a, b, c = q[f[:, 0]], q[f[:, 1]], q[f[:, 2]]
            keep = np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-11
            f, cn = f[keep], cn[keep]
            no = np.stack([cn[..., 0], cn[..., 2], -cn[..., 1]], -1).reshape(-1, 3)
            no = np.round(no / np.maximum(np.linalg.norm(no, axis=1, keepdims=True), 1e-9), 4)
            un, inv = np.unique(no, axis=0, return_inverse=True)
            inv = inv.reshape(-1, 3)
            fh.write(f"o {name}\ng {name}\nusemtl {mat}\ns 1\n")
            fh.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in vo))
            fh.write("".join(f"vn {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in un))
            fv, fn = f + voff, inv + noff
            fh.write("".join(f"f {a}//{d} {b}//{e} {c}//{g}\n" for (a, b, c), (d, e, g) in zip(fv, fn)))
            voff += len(v)
            noff += len(un)
    with open(os.path.join(os.path.dirname(path), mtl), "w") as fh:
        fh.write(f"# materials for {base}.obj\n")
        for name, d in mats.items():
            fh.write(f"\nnewmtl {name}\n")
            for k, val in d.items():
                fh.write(f"{k} {val}\n")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--samples", type=int, default=160)
    ap.add_argument("--res", type=int, default=1600)
    ap.add_argument("--views", default="0,45,90,135,180,225,270,315")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--ortho-check", action="store_true")
    ap.add_argument("--elev", type=float, default=6.0)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    name = a.name
    out = a.outdir or os.path.join(ROOT, "output", name)
    os.makedirs(os.path.join(out, "renders"), exist_ok=True)
    seg = np.load(os.path.join(ROOT, "work", f"{name}.seg.npz"))
    data = np.load(os.path.join(ROOT, "work", f"{name}.mesh.npz"))
    meta = json.loads(str(data["meta"]))
    T1 = data["T"]
    S = seg["S"]
    U = int(seg["U"])
    ys, xs = np.nonzero(S)
    cx = (xs.min() + xs.max() + 1) / 2 / U
    y_floor = (ys.max() + 1) / U
    height_m = (ys.max() - ys.min() + 1) / U * PX

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene

    def to_world(v):
        return np.stack([(v[:, 0] - cx) * PX, -v[:, 2] * PX, (y_floor - v[:, 1]) * PX], 1)

    m_yel = principled("Banana_Yellow", YELLOW, rough=0.38, coat=0.25, coat_rough=0.18, spec=0.4)
    m_blk = principled("Ink_Black", (0.002, 0.002, 0.002), rough=0.42, coat=0.3, coat_rough=0.07, spec=0.25)
    m_line = principled("Ink_Line_Matte", (0.003, 0.003, 0.003), rough=0.8, coat=0.0, spec=0.2)
    m_out = outline_material()

    root = bpy.data.objects.new("Banana", None)
    sc.collection.objects.link(root)
    groups = []
    hull_src = []
    # name the face decals: the biggest is the mouth, the others are eyes (left->right in the drawing)
    fp = [p for p in meta["pieces"] if p.startswith("face")]
    fsz = {p: len(data[p + "_f"]) for p in fp}
    mouth = max(fp, key=lambda p: fsz[p]) if fp else None
    eyes = sorted([p for p in fp if p != mouth], key=lambda p: float(data[p + "_v"][:, 0].mean()))
    face_names = {mouth: "Face_Mouth"}
    for i, p in enumerate(eyes):
        face_names[p] = f"Face_Eye_{i + 1}"

    for piece in meta["pieces"]:
        v = to_world(data[piece + "_v"].astype(np.float64))
        f = data[piece + "_f"].astype(np.int64)
        if signed_volume(v, f) < 0:
            f = f[:, ::-1].copy()
        if piece.endswith("_hull"):
            hull_src.append((piece, v, f))
            continue
        main = piece.startswith("part1")
        if "yellow" in piece:
            mat, label = m_yel, "Banana_Body" if main else "Banana_BackPeelFlap"
            target = 170000 if main else 30000
        elif "cap" in piece:
            mat, label = m_blk, "Banana_Tips_Black" if main else "BackPeelFlap_Tip_Black"
            target = 16000
        elif piece.startswith("face"):
            mat, label = m_blk, face_names.get(piece, "Face_" + piece)
            target = 6000
        elif piece.startswith("web"):
            mat, label = m_line, "InkLine_ArmSeam_" + piece[3:]
            target = 8000
        else:
            mat, label = m_line, "InkLine_FlapSeam_" + piece[8:]
            target = 8000
        ob = make_mesh(label, v, f, mat)
        ob["sharp_split"] = True
        decimate(ob, min(1.0, target / max(len(f), 1)))
        # split normals at real creases (cap cut, decal walls) - no shading bleed
        ob.data.set_sharp_from_angle(angle=math.radians(50))
        ob.parent = root
        groups.append(ob)

    # ---- outline shell (inverted hull) with hand-drawn thickness wobble
    for piece, v, f in hull_src:
        # the shell is built (build_mesh.hull_shell) as the offset surface at the
        # drawn line width; here it is only thinned and turned inside-out
        main = piece.startswith("part1")
        hull = make_mesh("Outline_Ink_" + ("Body" if main else "BackPeelFlap"), v, f[:, ::-1].copy(), m_out)
        decimate(hull, min(1.0, (130000 if main else 30000) / len(f)))
        hull.parent = root
        for attr in ("visible_shadow", "visible_diffuse", "visible_glossy", "visible_transmission",
                     "visible_volume_scatter"):
            setattr(hull, attr, False)
        groups.append(hull)

    # ---- export OBJ + MTL
    if not a.no_export:
        mats = {
            "Banana_Yellow": {"Ka": "0 0 0", "Kd": "0.9961 0.9451 0.0392", "Ks": "0.35 0.35 0.35",
                              "Ns": "180", "d": "1", "illum": "2", "Pr": "0.38", "Pc": "0.25", "Pcr": "0.18"},
            "Ink_Black": {"Ka": "0 0 0", "Kd": "0.0020 0.0020 0.0020", "Ks": "0.4 0.4 0.4",
                          "Ns": "220", "d": "1", "illum": "2", "Pr": "0.42", "Pc": "0.30", "Pcr": "0.07"},
            "Ink_Line_Matte": {"Ka": "0 0 0", "Kd": "0.0030 0.0030 0.0030", "Ks": "0.05 0.05 0.05",
                               "Ns": "20", "d": "1", "illum": "2", "Pr": "0.8"},
            "Outline_Ink": {"Ka": "0 0 0", "Kd": "0 0 0", "Ks": "0 0 0", "Ns": "1", "Ke": "0 0 0",
                            "d": "1", "illum": "0"},
        }
        rows = []
        for ob in groups:
            v, f, n = mesh_arrays(ob)
            rows.append((ob.name, ob.data.materials[0].name, v, f, corner_normals(ob)))
        # main body first, outline last
        rows.sort(key=lambda r: (r[0].startswith("Outline"), not r[0].startswith("Banana_Body"), r[0]))
        write_obj(os.path.join(out, f"{name}.obj"), rows, mats)
        stats = {r[0]: dict(verts=len(r[2]), tris=len(r[3])) for r in rows}
        json.dump(dict(height_m=round(height_m, 4), objects=stats, meta=meta),
                  open(os.path.join(out, f"{name}_model_info.json"), "w"), indent=1)
        print("exported", sum(len(r[3]) for r in rows), "triangles")

    # ---- studio
    m_bd = principled("Backdrop", BACKDROP, rough=0.85, spec=0.3)
    cyclorama(m_bd)
    tgt = (0, 0, height_m * 0.47)
    area_light("Key", (-3.2, -4.0, 4.6), tgt, 1.2, LIGHT * 1500)
    area_light("Fill", (4.2, -3.0, 1.8), tgt, 3.5, LIGHT * 175, (0.95, 0.97, 1.0))
    area_light("Rim", (1.8, 3.4, 3.8), tgt, 2.0, LIGHT * 900, (1.0, 0.98, 0.94))
    area_light("Top", (0.0, -0.5, 6.0), tgt, 3.0, LIGHT * 180)
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bgn = world.node_tree.nodes["Background"]
    bgn.inputs["Color"].default_value = (0.75, 0.76, 0.78, 1)
    bgn.inputs["Strength"].default_value = 0.28 * LIGHT
    sc.world = world

    cam_d = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    frame_h = max(height_m, 0.0) * 1.3
    # widest extent (any turntable angle) -> keep the whole character in frame
    width_m = (xs.max() - xs.min() + 1) / U * PX
    frame = max(frame_h, width_m * 1.22)
    cam_d.lens = 85
    cam_d.sensor_fit = "VERTICAL"
    cam_d.sensor_height = 24
    fov = 2 * math.atan(12 / 85)
    dist = (frame / 2) / math.tan(fov / 2)
    el = math.radians(a.elev)
    look = Vector((0, 0, height_m * 0.5))
    cam.location = look + Vector((0, -dist * math.cos(el), dist * math.sin(el)))
    cam.rotation_euler = (look - cam.location).to_track_quat("-Z", "Y").to_euler()

    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = a.samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.015
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
    sc.cycles.diffuse_bounces = 3
    sc.cycles.glossy_bounces = 3
    sc.cycles.transparent_max_bounces = 24
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.resolution_x = a.res
    sc.render.resolution_y = a.res
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.threads_mode = "AUTO"

    root.location = (0, 0, 0.0015)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, f"{name}.blend"), compress=True)

    labels = {0: "front", 45: "front-right-diagonal", 90: "right-side", 135: "back-right-diagonal",
              180: "back", 225: "back-left-diagonal", 270: "left-side", 315: "front-left-diagonal"}
    if not a.no_render:
        for ang in [int(x) for x in a.views.split(",") if x != ""]:
            # turntable: the character turns, camera + lights stay put
            # ang = how far the camera has orbited to the character's right
            root.rotation_euler = (0, 0, math.radians(ang))
            fn = os.path.join(out, "renders", f"{name}_{ang:03d}_{labels.get(ang, 'view')}.png")
            sc.render.filepath = fn
            bpy.ops.render.render(write_still=True)
            print("rendered", fn, flush=True)
        root.rotation_euler = (0, 0, 0)

    if a.ortho_check:
        # exact orthographic front view at eye level, framed 1:1 with the drawing
        W0, H0 = (int(x) for x in seg["size"])
        cam_d.type = "ORTHO"
        cam_d.sensor_fit = "HORIZONTAL"
        cam_d.ortho_scale = W0 * PX
        sc.render.resolution_x, sc.render.resolution_y = W0, H0
        cx_img = (W0 / 2 - cx) * PX
        cz_img = (y_floor - H0 / 2) * PX + root.location.z
        cam.location = (cx_img, -20, cz_img)
        cam.rotation_euler = (math.radians(90), 0, 0)
        sc.cycles.samples = max(32, a.samples // 2)
        fn = os.path.join(out, "renders", f"{name}_front_ortho_match.png")
        sc.render.filepath = fn
        bpy.ops.render.render(write_still=True)
        print("rendered", fn, flush=True)


if __name__ == "__main__":
    main()
