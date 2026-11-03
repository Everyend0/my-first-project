"""Trace the Iris mascot drawing into 3D-ready geometry data.

Usage:
    python3 trace_iris.py <iris.png> <out_dir>

Reads the flat mascot drawing and writes, in image pixel coordinates
(x right, y down):
    iris_trace.npz   front-sheet mesh of the whole silhouette with per-vertex
                     height (inflation) and per-triangle material ids
    iris_trace.json  metadata: colours, stroke width, hub centre, arm
                     centre lines (used for rigging)
    trace_debug.png / height_debug.png   visual checks

Nothing here invents shape or colour: the outer silhouette, the black
outline band, the eyes and the smile are traced from the drawing and become
constrained edges of the mesh, so the front view reproduces the drawing.
"""
import json
import os
import sys

import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

UP = 4  # super-sampling factor for sub-pixel contour tracing

MAT_YELLOW, MAT_OUTLINE, MAT_FACE = 0, 1, 2


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def largest_component(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def outer_contour(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float64)


def smooth_closed(pts, sigma):
    return np.stack([ndi.gaussian_filter1d(pts[:, i], sigma, mode="wrap")
                     for i in range(2)], 1)


def resample_closed(pts, spacing):
    p = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(12, int(round(s[-1] / spacing)))
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def signed_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def bilinear(img, xy):
    return ndi.map_coordinates(img, [xy[:, 1], xy[:, 0]], order=1, mode="nearest")


def poly_mask(shape, pts, shift=4):
    m = np.zeros(shape, np.uint8)
    p = np.round(pts * (1 << shift)).astype(np.int32)
    cv2.fillPoly(m, [p], 1, lineType=cv2.LINE_8, shift=shift)
    return m.astype(bool)


def subpixel_loop(field, level, seed_mask, sigma=1.5, spacing=2.0):
    """Iso-contour of a scalar field at `level`, traced on a UPx super-sampled
    grid (bilinear), restricted to the component overlapping seed_mask."""
    H, W = field.shape
    up = cv2.resize(field.astype(np.float32), (W * UP, H * UP), interpolation=cv2.INTER_LINEAR)
    m = up > level
    seed_up = cv2.resize(seed_mask.astype(np.uint8), (W * UP, H * UP),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    ids, counts = np.unique(lab[seed_up & m], return_counts=True)
    ids, counts = ids[ids > 0], counts[ids > 0]
    m = lab == ids[np.argmax(counts)]
    c = outer_contour(m)
    # cv2.resize maps pixel centres: up-index u -> (u + 0.5)/UP - 0.5
    c = (c + 0.5) / UP - 0.5
    c = smooth_closed(c, sigma * UP / 2)
    c = resample_closed(c, spacing)
    if signed_area(c) < 0:
        c = c[::-1]
    return c


def hex_grid(mask_ok, spacing):
    h, w = mask_ok.shape
    out = []
    dy = spacing * np.sqrt(3) / 2
    y, row = 0.0, 0
    while y < h:
        xs = np.arange((spacing / 2) if row % 2 else 0.0, w, spacing)
        out.append(np.stack([xs, np.full_like(xs, y)], 1))
        y += dy
        row += 1
    pts = np.vstack(out)
    return pts[bilinear(mask_ok.astype(np.float32), pts) > 0.999]


def point_in_poly(pts, poly):
    """Even-odd test, vectorised over pts."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x0, y0 = poly[:, 0][None], poly[:, 1][None]
    x1, y1 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    cond = (y0 > y) != (y1 > y)
    xin = (x1 - x0) * (y - y0) / np.where(y1 - y0 == 0, 1e-12, y1 - y0) + x0
    return (cond & (x < xin)).sum(1) % 2 == 1


def solve_poisson(mask):
    """u with  -lap(u) = 1  inside mask, u = 0 outside (pixel units)."""
    H, W = mask.shape
    idx = -np.ones((H, W), np.int64)
    ys, xs = np.nonzero(mask)
    n = len(ys)
    idx[ys, xs] = np.arange(n)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)]
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yy, xx = ys + dy, xs + dx
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        nb = np.full(n, -1)
        nb[ok] = idx[yy[ok], xx[ok]]
        good = nb >= 0
        rows.append(np.arange(n)[good])
        cols.append(nb[good])
        vals.append(np.full(good.sum(), -1.0))
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n))
    b = np.ones(n)
    d = 1.0 / A.diagonal()
    M = spla.LinearOperator((n, n), matvec=lambda v: d * v)
    u, info = spla.cg(A, b, rtol=1e-9, maxiter=20000, M=M)
    if info != 0:
        print("  poisson cg info", info)
    out = np.zeros((H, W))
    out[ys, xs] = u
    return out


def triangulate(loops, inner_pts):
    """Constrained Delaunay (Shewchuk Triangle) of several closed loops plus
    interior points. Loop edges are preserved exactly ('p') with no Steiner
    points on them ('Y'). Returns points (loops first, in order) and tris."""
    import triangle as tr
    loop_pts = np.vstack(loops)
    segs, start = [], 0
    for lp in loops:
        n = len(lp)
        i = np.arange(n) + start
        segs.append(np.stack([i, np.roll(i, -1)], 1))
        start += n
    segs = np.vstack(segs)
    inner = np.asarray(inner_pts, float).reshape(-1, 2)
    if len(inner):
        d, _ = cKDTree(loop_pts).query(inner)
        inner = inner[d > 0.9]
        # thin out inner points that collide with each other
        tree = cKDTree(inner)
        drop = set()
        for i, j in sorted(tree.query_pairs(0.9)):
            if i not in drop:
                drop.add(j)
        inner = np.delete(inner, sorted(drop), axis=0)
    pts = np.vstack([loop_pts, inner])
    res = __import__("triangle").triangulate(dict(vertices=pts, segments=segs), "pY")
    if len(res["vertices"]) != len(pts):
        print(f"  note: triangle changed vertex count {len(pts)} -> {len(res['vertices'])}")
    return res["vertices"], res["triangles"]


def mesh_check(tri, nb):
    edges = {}
    for t in tri:
        for u, v in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = (min(u, v), max(u, v))
            edges[k] = edges.get(k, 0) + 1
    border = [k for k, c in edges.items() if c == 1]
    nonloop = [k for k in border
               if not (k[0] < nb and k[1] < nb and (k[1] - k[0]) % nb in (1, nb - 1))]
    over = sum(1 for c in edges.values() if c > 2)
    euler = len(np.unique(tri)) - len(edges) + len(tri)
    return dict(border=len(border), nonloop_border=len(nonloop), overused=over, euler=euler)


# --------------------------------------------------------------------------
def main(src, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    bgr = cv2.imread(src, cv2.IMREAD_COLOR)
    rgb = bgr[:, :, ::-1].astype(np.float32)
    H, W = rgb.shape[:2]
    G_ = rgb[..., 1]
    minc = rgb.min(2)

    # ---- masks ----------------------------------------------------------
    sil = ndi.binary_fill_holes(largest_component(minc < 128))
    yellow_raw = (G_ > 110) & (rgb[..., 0] > 128) & (rgb[..., 2] < 128) & sil
    yellow = ndi.binary_fill_holes(largest_component(yellow_raw))
    face = yellow & (G_ < 110)
    solid = ndi.binary_erosion(yellow_raw, iterations=3)
    yellow_srgb = np.median(rgb[solid], 0)
    black_srgb = np.median(rgb[ndi.binary_erosion(sil & ~yellow_raw & (G_ < 40), iterations=2)], 0)

    # ---- sub-pixel loops --------------------------------------------------
    # outer silhouette: half-way between white (~254) and black/yellow (min ch.)
    outer = subpixel_loop(255.0 - minc, 127.0, sil, sigma=1.5, spacing=2.2)
    # yellow region edge: green channel half-way between black (0) and yellow (219)
    yel_loop = subpixel_loop(G_ * sil, 110.0, yellow, sigma=1.5, spacing=2.2)
    face_loops = []
    n, lab, stats, cen = cv2.connectedComponentsWithStats(face.astype(np.uint8), 8)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 60:
            continue
        fm = lab == k
        grow = ndi.binary_dilation(fm, iterations=4)
        c = subpixel_loop((219.0 - G_) * grow, 109.5, fm, sigma=1.2, spacing=1.2)
        face_loops.append(dict(loop=c, area=int(stats[k, cv2.CC_STAT_AREA]),
                               centroid=cen[k].tolist()))
    face_loops.sort(key=lambda d: (d["area"] > 1500, d["centroid"][0]))
    names = ["eye_L", "eye_R", "smile"] if len(face_loops) == 3 else \
        [f"face{i}" for i in range(len(face_loops))]
    for d, nm in zip(face_loops, names):
        d["name"] = nm

    Rmask = poly_mask((H, W), outer)
    dt = cv2.distanceTransform(Rmask.astype(np.uint8), cv2.DIST_L2, 5)
    dt = np.maximum(dt - 0.5, 0)
    # stroke width measured along the yellow loop
    stroke_w = float(np.median(bilinear(dt, yel_loop)))

    # ---- inflation: Poisson  -lap u = 1, round cross-sections -----------
    u = solve_poisson(Rmask)
    s = np.sqrt(np.clip(2.0 * u, 0, None))  # = sqrt(R^2 - x^2) for a tube
    hub_xy = np.array(np.unravel_index(np.argmax(dt), dt.shape)[::-1], float)
    ridge = (dt >= ndi.maximum_filter(dt, size=3)) & (dt > 3)
    hub_r_in = float(dt.max())
    R_arm = float(np.median(dt[ridge & (dt < 0.6 * hub_r_in)]))  # outer half width
    S1 = 1.3 * R_arm
    k = 0.80 * R_arm / (R_arm / (1 + (R_arm / S1) ** 4) ** 0.25)
    capped = lambda v: k * v / (1 + (v / S1) ** 4) ** 0.25  # noqa: E731
    hmap = capped(s)

    # ---- mesh -----------------------------------------------------------
    rings = []
    for d, spc in ((1.2, 2.3), (3.2, 2.6), (6.5, 3.0), (11.0, 3.6), (26.0, 4.2)):
        lev = largest_component(dt >= d)
        c = resample_closed(smooth_closed(outer_contour(lev), 2.0), spc)
        rings.append(c)
    grid = hex_grid(dt >= 15.0, 5.0)
    # denser points inside and around the face strokes
    face_zone = np.zeros((H, W), bool)
    for d in face_loops:
        face_zone |= poly_mask((H, W), d["loop"])
    face_zone_d = ndi.binary_dilation(face_zone, iterations=6)
    gface = hex_grid(face_zone_d, 2.2)
    grid = grid[bilinear(face_zone_d.astype(np.float32), grid) < 0.5]
    loops = [outer, yel_loop] + [d["loop"] for d in face_loops]
    pts, tri = triangulate(loops, np.vstack(rings + [grid, gface]))
    nb = len(outer)
    chk = mesh_check(tri, nb)
    print(f"mesh: pts={len(pts)} tris={len(tri)} {chk}")
    assert chk["nonloop_border"] == 0 and chk["overused"] == 0 and chk["euler"] == 1

    # per-vertex height.
    # The black outline is a thin, almost flat rim (depth <= H_RIM) and the
    # yellow body is inflated from the yellow edge with a finite edge slope.
    # If the outline band were painted on the steep side of one big inflated
    # shape, it would look thick on one side and hairline-thin on the other as
    # soon as the camera tilts; a flat rim keeps the drawn width from every
    # angle the videos use. The front view (xy) is unchanged either way.
    H_RIM, C_EDGE = 4.0, 30.0
    u_y = solve_poisson(poly_mask((H, W), yel_loop))
    prof = lambda q: np.sqrt(2.0 * q + C_EDGE * C_EDGE) - C_EDGE  # noqa: E731
    R_y = R_arm - stroke_w                       # yellow half-width of an arm
    xs = np.linspace(0.0, S1, 4000)
    # keep the old arm-centre height: capped(R_arm) == H_RIM + capped(k2 * prof(R_y^2/2))
    k2 = np.interp(capped(R_arm) - H_RIM, capped(xs), xs) / prof(R_y ** 2 / 2)
    in_y = bilinear((u_y > 0).astype(np.float32), pts) > 0.5
    in_y[nb:nb + len(yel_loop)] = True           # yellow-loop vertices (loops come first)
    h_y = H_RIM + capped(k2 * prof(np.clip(bilinear(u_y, pts), 0, None)))
    dv = np.clip(bilinear(dt, pts) / stroke_w, 0.0, 1.0)
    h_b = H_RIM * np.sqrt(1.0 - (1.0 - dv) ** 2)  # rounded rim; front/back meet on the outer loop
    h = np.where(in_y, h_y, h_b)
    h[:nb] = 0.0
    hmap = H_RIM + capped(k2 * prof(np.clip(u_y, 0, None)))   # for height_debug.png
    print(f"height: rim {H_RIM}px, edge slope {k2 * R_y / C_EDGE:.2f}, max {h.max():.1f}px")

    # per-triangle material by centroid
    cen_t = pts[tri].mean(1)
    mat = np.full(len(tri), MAT_OUTLINE, np.int32)
    mat[point_in_poly(cen_t, yel_loop)] = MAT_YELLOW
    for d in face_loops:
        mat[point_in_poly(cen_t, d["loop"])] = MAT_FACE
    print("materials:", np.bincount(mat, minlength=3))

    # ---- arms (for rigging) ---------------------------------------------
    rel = outer - hub_xy
    rho = np.hypot(rel[:, 0], rel[:, 1])
    rho_s = ndi.gaussian_filter1d(rho, 8, mode="wrap")
    N = len(outer)
    win = N // 24
    cand = [i for i in range(N) if rho_s[i] == rho_s[np.arange(i - win, i + win + 1) % N].max()]
    cand.sort(key=lambda i: -rho_s[i])
    tips = []
    for i in cand:
        a = np.arctan2(rel[i, 1], rel[i, 0])
        if all(abs((a - np.arctan2(rel[j, 1], rel[j, 0]) + np.pi) % (2 * np.pi) - np.pi) > 0.35
               for j in tips):
            tips.append(i)
        if len(tips) == 8:
            break
    tips.sort()
    notch = []
    for a, b in zip(tips, tips[1:] + tips[:1]):
        idx = np.arange(a, b if b > a else b + N) % N
        notch.append(int(idx[np.argmin(rho_s[idx])]))
    r_hub = float(np.mean(rho[notch]))
    mf = Rmask.astype(np.float32)

    def cross(p, nrm):
        """Mid-point of the arm cross-section through p, or p itself when the
        section runs into a neighbouring arm (too wide to be one arm)."""
        ext = []
        lim = 2.2 * R_arm
        for sgn in (1, -1):
            s_ = 0.0
            while s_ < lim and bilinear(mf, (p + sgn * nrm * (s_ + 0.5))[None])[0] > 0.5:
                s_ += 0.5
            ext.append(s_)
        if max(ext) >= lim:
            return p, None
        return p + nrm * (ext[0] - ext[1]) / 2.0, (ext[0] + ext[1]) / 2.0

    arms = []
    for ti in tips:
        tip_edge = outer[ti]
        cl = hub_xy[None] + np.linspace(0, 1, 100)[:, None] * (tip_edge - hub_xy)[None]
        for _ in range(4):
            tang = np.gradient(cl, axis=0)
            tang /= np.linalg.norm(tang, axis=1, keepdims=True)
            nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
            new = cl.copy()
            rr = np.linalg.norm(cl - hub_xy, axis=1)
            Ltot = rr[-1]
            valid = (rr > 1.3 * r_hub) & (rr < Ltot - 1.1 * R_arm)
            for j in np.nonzero(valid)[0]:
                new[j] = cross(cl[j], nrm[j])[0]
            # outside the valid band: blend linearly between hub and the band
            vj = np.nonzero(valid)[0]
            j0, j1 = vj[0], vj[-1]
            for j in range(j0):
                t = j / j0
                new[j] = hub_xy * (1 - t) + new[j0] * t
            # extend the last valid tangent to the tip region
            for j in range(j1 + 1, len(cl)):
                t = (j - j1) / (len(cl) - 1 - j1)
                new[j] = new[j1] * (1 - t) + tip_edge * t
            new = np.stack([ndi.gaussian_filter1d(new[:, i], 3.0, mode="nearest")
                            for i in range(2)], 1)
            new[0], new[-1] = hub_xy, tip_edge
            cl = new
        seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
        sacc = np.concatenate([[0], np.cumsum(seg)])
        tt = np.linspace(0, sacc[-1], 64)
        cl = np.stack([np.interp(tt, sacc, cl[:, 0]), np.interp(tt, sacc, cl[:, 1])], 1)
        v = tip_edge - hub_xy
        arms.append(dict(tip=tip_edge.tolist(), length=float(sacc[-1]),
                         angle_deg=float(np.degrees(np.arctan2(-v[1], v[0]))),
                         centerline=cl.round(3).tolist()))
        print(f"arm angle={arms[-1]['angle_deg']:7.1f} len={sacc[-1]:6.1f}")

    face_pts = np.vstack([d["loop"] for d in face_loops])
    meta = dict(
        image_size=[W, H],
        yellow_srgb=[float(v) for v in yellow_srgb],
        black_srgb=[float(v) for v in black_srgb],
        stroke_width=stroke_w,
        hub=hub_xy.tolist(), hub_radius=r_hub, hub_inscribed=hub_r_in,
        arm_halfwidth=R_arm, height_k=k, height_S1=S1,
        max_height=float(h.max()),
        arms=arms,
        face_bbox=[float(face_pts[:, 0].min()), float(face_pts[:, 1].min()),
                   float(face_pts[:, 0].max()), float(face_pts[:, 1].max())],
        face=[dict(name=d["name"], area=d["area"], centroid=d["centroid"]) for d in face_loops],
        mesh=dict(points=int(len(pts)), tris=int(len(tri)), outer_loop=int(nb), **chk),
    )
    with open(os.path.join(out_dir, "iris_trace.json"), "w") as f:
        json.dump(meta, f, indent=1)
    np.savez_compressed(os.path.join(out_dir, "iris_trace.npz"),
                        xy=pts, h=h, tris=tri, mat=mat, nloop=np.array(nb),
                        outer=outer, yellow_loop=yel_loop)
    print(f"stroke={stroke_w:.2f}px hub=({hub_xy[0]:.0f},{hub_xy[1]:.0f}) r_hub={r_hub:.1f} "
          f"R_arm={R_arm:.1f} max_h={h.max():.1f} yellow={yellow_srgb} black={black_srgb}")

    # ---- debug images ---------------------------------------------------
    dbg = bgr.copy()
    for lp, col in [(outer, (255, 0, 255)), (yel_loop, (0, 160, 0))] + \
            [(d["loop"], (0, 0, 255)) for d in face_loops]:
        cv2.polylines(dbg, [np.round(lp * 4).astype(np.int32)], True, col, 1, cv2.LINE_AA, shift=2)
    for a in arms:
        cv2.polylines(dbg, [np.round(np.array(a["centerline"])).astype(np.int32)], False,
                      (255, 0, 0), 2)
    cv2.circle(dbg, tuple(np.round(hub_xy).astype(int)), int(r_hub), (0, 128, 255), 2)
    cv2.imwrite(os.path.join(out_dir, "trace_debug.png"), dbg)
    cv2.imwrite(os.path.join(out_dir, "height_debug.png"),
                (hmap / hmap.max() * 255).astype(np.uint8))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
