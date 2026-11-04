"""Step 5 - Unity physics: compound capsule colliders fitted to each banana.

Why capsules: the bananas are concave (legs / arms branch off the body), and PhysX
silently ignores a non-convex MeshCollider on a dynamic Rigidbody, while a convex
hull would fill the gap between the legs.  So each limb gets its own capsule,
fitted along the medial axis (skeleton) of the drawn body region; the radius is
the local half-width of the drawing plus most of the ink outline, so things touch
the visible black line instead of sinking into it.  The small back peel flap of the
legs banana gets a box from its mesh bounds at runtime (see BananaPhysics.cs).

Writes output/<name>/<name>_colliders.json and unity/BananaPhysics.cs (both bananas).
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PX = 0.002
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def neighbours(sk, p):
    y, x = p
    return [(y + dy, x + dx) for dy, dx in NB if sk[y + dy, x + dx]]


def trace_branches(sk):
    """split a 1-px skeleton into branches between end/junction pixels"""
    pts = set(zip(*np.nonzero(sk)))
    deg = {p: len(neighbours(sk, p)) for p in pts}
    nodes = {p for p, d in deg.items() if d != 2}
    seen, branches = set(), []
    for n in nodes:
        for nb in neighbours(sk, n):
            if (n, nb) in seen:
                continue
            path, prev, cur = [n], n, nb
            while True:
                path.append(cur)
                seen.add((prev, cur))
                seen.add((cur, prev))
                if cur in nodes:
                    break
                nxt = [q for q in neighbours(sk, cur) if q != prev and (cur, q) not in seen]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            branches.append(path)
    return branches, nodes, deg


def prune(mask, dt, rounds=4):
    sk = skeletonize(mask)
    for _ in range(rounds):
        br, nodes, deg = trace_branches(sk)
        changed = False
        for b in br:
            a, z = b[0], b[-1]
            ends = [p for p in (a, z) if deg.get(p, 0) == 1]
            juncs = [p for p in (a, z) if deg.get(p, 0) >= 3]
            if ends and juncs and len(b) < 1.6 * dt[juncs[0]]:
                for p in b[:-1] if b[-1] in juncs else b[1:]:
                    sk[p] = False
                changed = True
        sk = skeletonize(sk)
        if not changed:
            break
    return sk


def rdp(pts, eps):
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-9:
        d = np.linalg.norm(pts - a, axis=1)
    else:
        d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n
    i = int(np.argmax(d))
    if d[i] > eps:
        return np.vstack([rdp(pts[: i + 1], eps)[:-1], rdp(pts[i:], eps)])
    return np.vstack([a, b])


def obj_extent(path, obj_name):
    """bounds (min, max) of one object's vertices in the exported OBJ"""
    lo, hi, inside = None, None, False
    with open(path) as fh:
        for line in fh:
            if line.startswith("o "):
                inside = line[2:].strip() == obj_name
            elif inside and line.startswith("v "):
                p = np.array(line.split()[1:4], float)
                lo = p if lo is None else np.minimum(lo, p)
                hi = p if hi is None else np.maximum(hi, p)
    return lo, hi


def capsules_for(name):
    seg = np.load(os.path.join(ROOT, "work", f"{name}.seg.npz"))
    mesh = np.load(os.path.join(ROOT, "work", f"{name}.mesh.npz"))
    meta = json.loads(str(mesh["meta"]))
    U = int(seg["U"])
    S, parts = seg["S"], seg["parts"]
    ys, xs = np.nonzero(S)
    cx = (xs.min() + xs.max() + 1) / 2 / U
    y_floor = (ys.max() + 1) / U
    ink = 0.95 * meta["outline_median_px"]  # (almost) the whole outline width, in drawing px
    objp = os.path.join(ROOT, "output", name, f"{name}.obj")
    lo, hi = obj_extent(objp, "Banana_Body")
    body_h = float(hi[1] - lo[1])  # height of the Banana_Body mesh exactly as exported
    mlo, mhi = obj_extent(objp, "Face_Mouth")
    mouth_x = float((mlo[0] + mhi[0]) / 2)
    caps = []
    for pinfo in meta["parts"]:
        p = pinfo["part"]
        if p != 1:
            continue  # back flap: box collider from its mesh bounds (BananaPhysics.cs)
        m2 = (parts == p) | mesh["fill_mask"]  # strokes painted on the body are solid
        m = m2[::U, ::U]  # work at drawing resolution
        dt = ndi.distance_transform_edt(m)
        sk = prune(m, dt)
        br, _, deg = trace_branches(sk)

        def reach(e, d):
            """distance from e along unit direction d to the body edge (drawing px)"""
            t = 0.0
            while True:
                y, x = int(round(e[0] + d[0] * (t + 1))), int(round(e[1] + d[1] * (t + 1)))
                if not (0 <= y < m.shape[0] and 0 <= x < m.shape[1]) or not m[y, x]:
                    return t
                t += 1.0

        for b in br:
            pts = np.array(b, float)
            if len(pts) < 3:
                continue
            simp = rdp(pts, 5.0)
            # skeleton ends sit inside the limb tip: extend them to the drawn tip so
            # the feet / stem / hands touch things where the outline is
            for idx, nb in ((0, 1), (-1, -2)):
                if deg.get(tuple(int(v) for v in pts[idx]), 0) == 1:
                    e = simp[idx]
                    dvec = e - simp[nb]
                    dvec /= np.linalg.norm(dvec) + 1e-9
                    rt = 0.7 * float(ndi.map_coordinates(dt, e[:, None], order=1)[0])
                    simp[idx] = e + dvec * max(0.0, reach(e, dvec) - rt)
            for a, c in zip(simp[:-1], simp[1:]):
                seglen = np.linalg.norm(c - a)
                k = max(2, int(seglen))
                line = a[None] + (c - a)[None] * np.linspace(0, 1, k)[:, None]
                r = np.median(ndi.map_coordinates(dt, line.T, order=1)) + ink
                if seglen < 0.25 * r and len(simp) > 2:
                    continue
                # where the limb narrows (tips, neck) the end sphere would poke out of the
                # outline: pull that end in, and cover the rest with a thinner tip capsule
                ends = [a.copy(), c.copy()]
                tips = []
                for i, (e, o) in enumerate(((a, c), (c, a))):
                    fit = float(ndi.map_coordinates(dt, e[:, None], order=1)[0]) + ink
                    if r > fit + 0.5:
                        ends[i] = e + (o - e) / seglen * min(r - fit, 0.45 * seglen)
                        tips.append((ends[i], e, fit))
                caps.append(dict(part=p, a=(ends[0][1], ends[0][0]), b=(ends[1][1], ends[1][0]), r=float(r), z=0.0))
                for q0, q1, rt in tips:
                    if np.linalg.norm(q1 - q0) > 0.5:
                        caps.append(dict(part=p, a=(q0[1], q0[0]), b=(q1[1], q1[0]), r=float(rt), z=0.0))
    # to OBJ coordinates (metres, +Y up, front +Z)
    out = []
    for c in caps:
        to = lambda q: [round((q[0] + 0.5 - cx) * PX, 5), round((y_floor - q[1] - 0.5) * PX, 5), round(c["z"] * PX, 5)]
        out.append(dict(part=c["part"], a=to(c["a"]), b=to(c["b"]), r=round(c["r"] * PX, 5)))
    # debug overlay on the drawing
    im = Image.open(os.path.join(ROOT, "src", f"{name}.png")).convert("RGB")
    dr = ImageDraw.Draw(im, "RGBA")
    for c in caps:
        (ax, ay), (bx, by), r = c["a"], c["b"], c["r"]
        dr.line([(ax, ay), (bx, by)], fill=(255, 0, 80, 255), width=3)
        for (x, y) in ((ax, ay), (bx, by)):
            dr.ellipse([x - r, y - r, x + r, y + r], outline=(0, 120, 255, 255), width=2)
        d = np.array([bx - ax, by - ay])
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9) * r
        dr.line([(ax + n[0], ay + n[1]), (bx + n[0], by + n[1])], fill=(0, 120, 255, 255), width=2)
        dr.line([(ax - n[0], ay - n[1]), (bx - n[0], by - n[1])], fill=(0, 120, 255, 255), width=2)
    im.save(os.path.join(ROOT, "output", name, f"{name}_colliders_preview.png"))
    lowest = min(min(c["a"][1], c["b"][1]) - c["r"] for c in out)
    json.dump(dict(units="metres, OBJ space (+Y up, character faces +Z)",
                   banana_body_mesh_height=round(body_h, 5), face_mouth_center_x=round(mouth_x, 5),
                   outline_width=round(ink * PX, 5), lowest_collider_point_y=round(lowest, 5),
                   capsules=out), open(os.path.join(ROOT, "output", name, f"{name}_colliders.json"), "w"), indent=1)
    print(name, len(out), "capsules, Banana_Body height", round(body_h, 4), "lowest collider y", round(lowest, 4))
    return out, body_h, mouth_x, ink * PX


CS = r'''// BananaPhysics.cs - Unity 3D physics (PhysX) for the banana OBJ models.
// Generated by banana3d/pipeline/colliders.py - capsule data fitted to the drawings.
//
// Usage: drag banana_legs.obj / banana_arms.obj into a scene, add this component to the
// model ROOT, done. It recognises legs/arms from the model's own child objects.
// Right-click the component header > "Build Colliders" to rebuild.
//
// What it sets up, and why (PhysX rules that bite with these particular models):
//  * Compound CapsuleColliders, one per limb. A non-convex MeshCollider on a dynamic
//    Rigidbody is silently ignored, and a convex one would fill the gap between the legs.
//    Radii include most of the thick ink outline, so contacts happen at the visible line.
//    The legs banana's back peel flap gets a BoxCollider from its mesh bounds.
//  * Every other collider under the model is removed (or disabled, inside a prefab) - in
//    particular never put a MeshCollider on "Outline_Ink_*": that toon-outline shell is
//    intentionally inside-out (inverted normals), so contacts would push objects INTO it.
//    Keep the OBJ importer's "Generate Colliders" OFF.
//  * Exactly one Rigidbody, on the root. A Rigidbody on a child splits the compound body.
//  * Collision Detection = Continuous Speculative so a thrown banana does not tunnel
//    through thin floors/walls.
//  * OnCollisionEnter / OnTriggerEnter for the capsules arrive on the ROOT object (the one
//    with the Rigidbody) - put your gameplay script there.
//  * Works with any import scale (measured from the Banana_Body mesh) and with Unity's
//    X-mirroring of OBJ files (read from the Face_Mouth mesh, which is off-centre).
//    Mesh Read/Write is NOT required. Keep the model's scale uniform (x = y = z).
using UnityEngine;

[DisallowMultipleComponent]
public class BananaPhysics : MonoBehaviour
{
    public enum Banana { Auto, Legs, Arms }

    public Banana banana = Banana.Auto;
    public bool addRigidbody = true;
    public float mass = 1f;
    public CollisionDetectionMode collisionDetection = CollisionDetectionMode.ContinuousSpeculative;

    struct Cap
    {
        public Vector3 a, b; public float r;
        public Cap(float ax, float ay, float az, float bx, float by, float bz, float r)
        { a = new Vector3(ax, ay, az); b = new Vector3(bx, by, bz); this.r = r; }
    }

    // capsule end points + radius in OBJ space (metres, +Y up, character faces +Z)
    static readonly Cap[] Legs = {
__LEGS__
    };
    static readonly Cap[] Arms = {
__ARMS__
    };
    // measured on the exported OBJ files
    const float LegsBodyMeshHeight = __LEGS_H__f;   // y-size of the Banana_Body mesh
    const float ArmsBodyMeshHeight = __ARMS_H__f;
    const float LegsMouthCenterX = __LEGS_MX__f;     // x of the Face_Mouth mesh centre
    const float ArmsMouthCenterX = __ARMS_MX__f;
    const float OutlineWidth = __INK__f;            // ink line counted as solid (metres)

    void Reset() { Build(); }

    void Awake()
    {
        if (transform.Find("Colliders") == null) Build();
    }

    MeshFilter FindMesh(string part)
    {
        foreach (MeshFilter mf in GetComponentsInChildren<MeshFilter>(true))
            if (mf.name.Contains(part) && mf.sharedMesh != null) return mf;
        return null;
    }

    Banana Detect()
    {
        if (banana != Banana.Auto) return banana;
        if (FindMesh("BackPeelFlap") != null || FindMesh("LegSplit") != null) return Banana.Legs;
        if (FindMesh("ArmSeam") != null) return Banana.Arms;
        bool arms = name.ToLowerInvariant().Contains("arm");
        Debug.LogWarning("BananaPhysics: could not recognise the model from its parts, guessing " +
                         (arms ? "Arms" : "Legs") + " from the name. Set 'banana' explicitly.", this);
        return arms ? Banana.Arms : Banana.Legs;
    }

    [ContextMenu("Build Colliders")]
    public void Build()
    {
        Banana id = Detect();
        Cap[] caps = id == Banana.Arms ? Arms : Legs;
        float bodyH = id == Banana.Arms ? ArmsBodyMeshHeight : LegsBodyMeshHeight;
        float mouthX = id == Banana.Arms ? ArmsMouthCenterX : LegsMouthCenterX;

        // 1) generated colliders live under "Colliders" (reused on rebuild, so it also works
        //    on prefab instances); anything else - e.g. importer MeshColliders on the inverted
        //    outline shell - is removed
        Transform holder = transform.Find("Colliders");
        if (holder == null)
        {
            holder = new GameObject("Colliders").transform;
            holder.SetParent(transform, false);
        }
        holder.localPosition = Vector3.zero; holder.localRotation = Quaternion.identity; holder.localScale = Vector3.one;
        foreach (Collider c in GetComponentsInChildren<Collider>(true))
            if (!c.transform.IsChildOf(holder)) Remove(c);
        foreach (Rigidbody crb in GetComponentsInChildren<Rigidbody>(true))
            if (crb.gameObject != gameObject) Remove(crb);

        // 2) OBJ space -> mesh space (import scale + X mirroring) -> this object's space
        float s = 1f, mirror = -1f;   // Unity negates X when importing OBJ
        Matrix4x4 bodyToRoot = Matrix4x4.identity;
        MeshFilter body = FindMesh("Banana_Body");
        if (body != null)
        {
            Bounds bb = body.sharedMesh.bounds;   // available without Read/Write
            if (bb.size.y > 1e-6f) s = bb.size.y / bodyH;
            bodyToRoot = transform.worldToLocalMatrix * body.transform.localToWorldMatrix;
        }
        else Debug.LogWarning("BananaPhysics: Banana_Body mesh not found, assuming OBJ units.", this);
        MeshFilter mouth = FindMesh("Face_Mouth");
        if (mouth != null && Mathf.Abs(mouth.sharedMesh.bounds.center.x) > 1e-4f * s)
            mirror = Mathf.Sign(mouth.sharedMesh.bounds.center.x) * Mathf.Sign(mouthX);
        float k = (bodyToRoot.GetColumn(0).magnitude + bodyToRoot.GetColumn(1).magnitude +
                   bodyToRoot.GetColumn(2).magnitude) / 3f;   // mesh -> root scale

        Vector3 ls = transform.lossyScale;
        if (Mathf.Abs(ls.x - ls.y) > 1e-3f || Mathf.Abs(ls.y - ls.z) > 1e-3f)
            Debug.LogWarning("BananaPhysics: non-uniform scale distorts the capsules - keep x=y=z.", this);

        // 3) one CapsuleCollider per limb segment
        int n = 0;
        for (int i = 0; i < caps.Length; i++, n++)
        {
            Vector3 a = bodyToRoot.MultiplyPoint3x4(Map(caps[i].a, s, mirror));
            Vector3 b = bodyToRoot.MultiplyPoint3x4(Map(caps[i].b, s, mirror));
            Vector3 d = b - a;
            float len = d.magnitude;
            Transform t = Child(holder, "Capsule_" + i);
            t.localPosition = (a + b) * 0.5f;
            t.localRotation = len > 1e-6f ? Quaternion.FromToRotation(Vector3.up, d / len) : Quaternion.identity;
            t.localScale = Vector3.one;
            CapsuleCollider cc = t.GetComponent<CapsuleCollider>();
            if (cc == null) cc = t.gameObject.AddComponent<CapsuleCollider>();
            cc.enabled = true;
            cc.center = Vector3.zero;
            cc.direction = 1;
            cc.radius = caps[i].r * s * k;
            cc.height = len + 2f * cc.radius;
        }

        // the legs banana's back peel flap: a box around its mesh (+ its ink line)
        MeshFilter flap = id == Banana.Legs ? FindMesh("Banana_BackPeelFlap") : null;
        Transform fb = holder.Find("Flap_Box");
        if (flap != null)
        {
            Bounds fbnd = flap.sharedMesh.bounds;
            Matrix4x4 m = transform.worldToLocalMatrix * flap.transform.localToWorldMatrix;
            if (fb == null) fb = Child(holder, "Flap_Box");
            fb.gameObject.SetActive(true);
            fb.localPosition = m.MultiplyPoint3x4(fbnd.center);
            fb.localRotation = m.rotation;
            fb.localScale = Vector3.one;
            BoxCollider bc = fb.GetComponent<BoxCollider>();
            if (bc == null) bc = fb.gameObject.AddComponent<BoxCollider>();
            bc.center = Vector3.zero;
            bc.size = Vector3.Scale(fbnd.size, m.lossyScale) + Vector3.one * (2f * OutlineWidth * s * k);
        }
        else if (fb != null) fb.gameObject.SetActive(false);

        // leftovers from an older build: switch off instead of deleting (prefab-safe)
        for (Transform t; (t = holder.Find("Capsule_" + n)) != null; n++) t.gameObject.SetActive(false);

        // 4) single dynamic Rigidbody on the root
        if (addRigidbody)
        {
            Rigidbody rb = GetComponent<Rigidbody>();
            if (rb == null) rb = gameObject.AddComponent<Rigidbody>();
            rb.mass = mass;
            rb.isKinematic = false;
            rb.interpolation = RigidbodyInterpolation.Interpolate;
            rb.collisionDetectionMode = collisionDetection;
        }
    }

    static Vector3 Map(Vector3 p, float s, float mirror) => new Vector3(mirror * p.x * s, p.y * s, p.z * s);

    static Transform Child(Transform parent, string childName)
    {
        Transform t = parent.Find(childName);
        if (t == null)
        {
            t = new GameObject(childName).transform;
            t.SetParent(parent, false);
        }
        t.gameObject.SetActive(true);
        return t;
    }

    static void Remove(Component c)
    {
        try
        {
            if (Application.isPlaying) Destroy(c); else DestroyImmediate(c);
        }
        catch (System.InvalidOperationException)
        {
            // components that belong to a prefab asset cannot be destroyed on the instance
            if (c is Collider col) col.enabled = false;
            else Debug.LogWarning("BananaPhysics: remove " + c.GetType().Name + " on '" + c.name +
                                  "' in the prefab - a child Rigidbody splits the compound body.", c);
        }
    }
}
'''


def cs_rows(caps):
    return "\n".join(
        f"        new Cap({c['a'][0]}f, {c['a'][1]}f, {c['a'][2]}f, {c['b'][0]}f, {c['b'][1]}f, {c['b'][2]}f, {c['r']}f),"
        for c in caps)


if __name__ == "__main__":
    legs, hl, mxl, inkl = capsules_for("banana_legs")
    arms, ha, mxa, inka = capsules_for("banana_arms")
    os.makedirs(os.path.join(ROOT, "unity"), exist_ok=True)
    cs = (CS.replace("__LEGS__", cs_rows(legs)).replace("__ARMS__", cs_rows(arms))
          .replace("__LEGS_H__", str(round(hl, 5))).replace("__ARMS_H__", str(round(ha, 5)))
          .replace("__LEGS_MX__", str(round(mxl, 5))).replace("__ARMS_MX__", str(round(mxa, 5)))
          .replace("__INK__", str(round(min(inkl, inka), 5))))
    open(os.path.join(ROOT, "unity", "BananaPhysics.cs"), "w").write(cs)
    print("wrote unity/BananaPhysics.cs")
