"""Step 5 - Unity physics: compound capsule colliders fitted to each banana.

Why capsules: the bananas are concave (legs / arms branch off the body), and PhysX
silently ignores a non-convex MeshCollider on a dynamic Rigidbody, while a convex
hull would fill the gap between the legs.  So each limb gets its own capsule,
fitted along the medial axis (skeleton) of the drawn body region; the radius is
the local half-width of the drawing.

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


def capsules_for(name):
    seg = np.load(os.path.join(ROOT, "work", f"{name}.seg.npz"))
    meta = json.loads(str(np.load(os.path.join(ROOT, "work", f"{name}.mesh.npz"))["meta"]))
    U = int(seg["U"])
    S, parts = seg["S"], seg["parts"]
    ys, xs = np.nonzero(S)
    cx = (xs.min() + xs.max() + 1) / 2 / U
    y_floor = (ys.max() + 1) / U
    caps = []
    body_h = None
    for pinfo in meta["parts"]:
        p = pinfo["part"]
        m2 = parts == p
        m = m2[::U, ::U]  # work at drawing resolution
        dt = ndi.distance_transform_edt(m)
        if p == 1:
            yy = np.nonzero(m2)[0]
            body_h = (yy.max() - yy.min() + 1) / U * PX
        sk = prune(m, dt)
        br, _, _ = trace_branches(sk)
        for b in br:
            pts = np.array(b, float)
            if len(pts) < 3:
                continue
            simp = rdp(pts, 5.0)
            for a, c in zip(simp[:-1], simp[1:]):
                seglen = np.linalg.norm(c - a)
                k = max(2, int(seglen))
                line = a[None] + (c - a)[None] * np.linspace(0, 1, k)[:, None]
                r = np.median(ndi.map_coordinates(dt, line.T, order=1)) * 0.92
                if seglen < 0.25 * r and len(simp) > 2:
                    continue
                caps.append(dict(part=p, a=(a[1], a[0]), b=(c[1], c[0]), r=float(r), z=pinfo["z_off"]))
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
    json.dump(dict(units="metres, OBJ space (+Y up, character faces +Z)", body_height=round(body_h, 5),
                   capsules=out), open(os.path.join(ROOT, "output", name, f"{name}_colliders.json"), "w"), indent=1)
    print(name, len(out), "capsules, body height", round(body_h, 4))
    return out, body_h


CS = r'''// BananaPhysics.cs - Unity 3D physics (PhysX) for the banana OBJ models.
// Generated by banana3d/pipeline/colliders.py - capsule data fitted to the drawings.
//
// Usage: drag banana_legs.obj / banana_arms.obj into a scene, add this component to the
// model ROOT (it auto-detects legs/arms from the object name), done. Right-click the
// component header > "Build Colliders" to rebuild.
//
// What it sets up, and why (PhysX rules that bite with these particular models):
//  * Compound CapsuleColliders, one per limb. A non-convex MeshCollider on a dynamic
//    Rigidbody is silently ignored, and a convex one would fill the gap between the legs.
//  * Every other collider under the model is removed - in particular never put a
//    MeshCollider on "Outline_Ink_*": that toon-outline shell is intentionally inside-out
//    (inverted normals), so contacts would push objects INTO it. Keep the OBJ importer's
//    "Generate Colliders" OFF.
//  * Exactly one Rigidbody, on the root. A Rigidbody on a child splits the compound body.
//  * Collision Detection = Continuous Speculative so a thrown banana does not tunnel
//    through thin floors/walls.
//  * OnCollisionEnter / OnTriggerEnter for the capsules arrive on the ROOT object (the one
//    with the Rigidbody) - put your gameplay script there.
//  * Keep the model's scale uniform (x = y = z); non-uniform parent scale distorts capsules.
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
    const float LegsBodyHeight = __LEGS_H__f;   // height of Banana_Body in the OBJ
    const float ArmsBodyHeight = __ARMS_H__f;

    void Reset() { Build(); }

    void Awake()
    {
        if (transform.Find("Colliders") == null) Build();
    }

    [ContextMenu("Build Colliders")]
    public void Build()
    {
        Banana id = banana != Banana.Auto ? banana
                  : (name.ToLowerInvariant().Contains("arm") ? Banana.Arms : Banana.Legs);
        Cap[] caps = id == Banana.Arms ? Arms : Legs;
        float expectedH = id == Banana.Arms ? ArmsBodyHeight : LegsBodyHeight;

        // 1) clear old colliders (incl. importer-generated MeshColliders on the outline shell)
        Transform old = transform.Find("Colliders");
        if (old != null) Kill(old.gameObject);
        foreach (Collider c in GetComponentsInChildren<Collider>(true)) Kill(c);
        foreach (Rigidbody crb in GetComponentsInChildren<Rigidbody>(true))
            if (crb.gameObject != gameObject) Kill(crb);

        // 2) OBJ space -> this object's local space (handles importer scale + X mirroring)
        float s = 1f, mirror = 1f;
        Matrix4x4 bodyToRoot = Matrix4x4.identity;
        MeshFilter body = null;
        foreach (MeshFilter mf in GetComponentsInChildren<MeshFilter>(true))
            if (mf.name.Contains("Banana_Body")) { body = mf; break; }
        if (body != null && body.sharedMesh != null)
        {
            Mesh mesh = body.sharedMesh;
            if (mesh.bounds.size.y > 1e-6f) s = mesh.bounds.size.y / expectedH;
            bodyToRoot = transform.worldToLocalMatrix * body.transform.localToWorldMatrix;
            if (Application.isEditor || mesh.isReadable) mirror = PickMirror(caps, mesh.vertices, s);
        }
        else Debug.LogWarning("BananaPhysics: Banana_Body mesh not found, assuming OBJ units.", this);

        Vector3 ls = transform.lossyScale;
        if (Mathf.Abs(ls.x - ls.y) > 1e-3f || Mathf.Abs(ls.y - ls.z) > 1e-3f)
            Debug.LogWarning("BananaPhysics: non-uniform scale distorts the capsules - keep x=y=z.", this);

        // 3) one CapsuleCollider per limb segment
        Transform root = new GameObject("Colliders").transform;
        root.SetParent(transform, false);
        for (int i = 0; i < caps.Length; i++)
        {
            Vector3 a = bodyToRoot.MultiplyPoint3x4(Map(caps[i].a, s, mirror));
            Vector3 b = bodyToRoot.MultiplyPoint3x4(Map(caps[i].b, s, mirror));
            Vector3 d = b - a;
            float len = d.magnitude;
            GameObject go = new GameObject("Capsule_" + i);
            go.transform.SetParent(root, false);
            go.transform.localPosition = (a + b) * 0.5f;
            go.transform.localRotation = len > 1e-6f ? Quaternion.FromToRotation(Vector3.up, d / len) : Quaternion.identity;
            CapsuleCollider cc = go.AddComponent<CapsuleCollider>();
            cc.direction = 1;
            cc.radius = caps[i].r * s;
            cc.height = len + 2f * cc.radius;
        }

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

    // Unity may mirror X when converting OBJ (right-handed) to its left-handed space:
    // pick the orientation whose capsule end points sit closest to the actual body mesh.
    static float PickMirror(Cap[] caps, Vector3[] verts, float s)
    {
        int step = Mathf.Max(1, verts.Length / 3000);
        float best = 1f, bestErr = float.MaxValue;
        foreach (float m in new[] { 1f, -1f })
        {
            float err = 0f;
            foreach (Cap c in caps)
                foreach (Vector3 p in new[] { c.a, c.b })
                {
                    Vector3 q = Map(p, s, m);
                    float dmin = float.MaxValue;
                    for (int i = 0; i < verts.Length; i += step)
                        dmin = Mathf.Min(dmin, (verts[i] - q).sqrMagnitude);
                    err += dmin;
                }
            if (err < bestErr) { bestErr = err; best = m; }
        }
        return best;
    }

    static void Kill(Object o)
    {
        if (Application.isPlaying) Destroy(o); else DestroyImmediate(o);
    }
}
'''


def cs_rows(caps):
    return "\n".join(
        f"        new Cap({c['a'][0]}f, {c['a'][1]}f, {c['a'][2]}f, {c['b'][0]}f, {c['b'][1]}f, {c['b'][2]}f, {c['r']}f),"
        for c in caps)


if __name__ == "__main__":
    legs, hl = capsules_for("banana_legs")
    arms, ha = capsules_for("banana_arms")
    os.makedirs(os.path.join(ROOT, "unity"), exist_ok=True)
    cs = (CS.replace("__LEGS__", cs_rows(legs)).replace("__ARMS__", cs_rows(arms))
          .replace("__LEGS_H__", str(round(hl, 5))).replace("__ARMS_H__", str(round(ha, 5))))
    open(os.path.join(ROOT, "unity", "BananaPhysics.cs"), "w").write(cs)
    print("wrote unity/BananaPhysics.cs")
