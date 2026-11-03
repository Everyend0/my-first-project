"""Compare the 1:1 front render of the 3D model with the original drawing.

Usage:
    python3 compare_iris.py <iris.png> <verify_dir>

Reads <verify_dir>/blender_checks.json and front_1to1.png (written by
verify_iris.py), aligns the render to drawing pixels exactly (no fitting),
and reports:
  * silhouette IoU and boundary distance (px)
  * black outline band IoU and median band width (drawing vs render)
  * face strokes (eyes + smile) IoU
  * yellow colour statistics of the render
Writes compare.json and compare_overlay.png (green = drawing only,
magenta = render only).
"""
import json
import os
import sys

import cv2
import numpy as np
from scipy import ndimage as ndi


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


# --------------------------------------------------------------------------
# geometry checks that do not need a render
# --------------------------------------------------------------------------
def loop_crossings(Q):
    """Number of self-intersections of a closed polyline (outline loop)."""
    from scipy.spatial import cKDTree
    A, B = Q, np.roll(Q, -1, 0)
    n = len(Q)
    seg = np.linalg.norm(B - A, axis=1).max()
    pairs = cKDTree((A + B) / 2).query_pairs(seg * 1.01, output_type="ndarray")
    if len(pairs) == 0:
        return 0
    i, j = pairs[:, 0], pairs[:, 1]
    keep = ~((np.abs(i - j) <= 1) | (np.abs(i - j) == n - 1))
    i, j = i[keep], j[keep]

    def orient(p, q, r):
        return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])

    a, b, c, d = A[i], B[i], A[j], B[j]
    hit = (orient(a, b, c) * orient(a, b, d) < 0) & (orient(c, d, a) * orient(c, d, b) < 0)
    return int(hit.sum())


def project_raster(trace, hub, elev_deg, yaw_deg, S=4):
    """Orthographic painter's-algorithm raster of the rest-pose mesh (front and
    back sheets) seen from a camera raised by elev and turned by yaw, in
    drawing pixels x S. Black = outline/face, yellow = body."""
    xy, h, tris, mat = trace["xy"], trace["h"], trace["tris"], trace["mat"]
    e, f = np.radians(elev_deg), np.radians(yaw_deg)
    parts = []
    for sgn in (1, -1):
        X = xy[:, 0] - hub[0]
        Y = -sgn * h
        Z = -(xy[:, 1] - hub[1])
        Xr = X * np.cos(f) - Y * np.sin(f)
        Yr = X * np.sin(f) + Y * np.cos(f)
        up = Yr * np.sin(e) + Z * np.cos(e)
        depth = -Yr * np.cos(e) + Z * np.sin(e)
        m = mat if sgn == 1 else np.where(mat == 2, 0, mat)
        parts.append((Xr, -up, depth, m))
    allx = np.concatenate([p[0] for p in parts])
    ally = np.concatenate([p[1] for p in parts])
    x0, y0 = allx.min() - 5, ally.min() - 5
    img = np.full((int((ally.max() - y0 + 5) * S), int((allx.max() - x0 + 5) * S)), 255, np.uint8)
    T, D, M = [], [], []
    for sx, sy, dep, m in parts:
        P = np.stack([(sx - x0) * S, (sy - y0) * S], 1)
        T.append(P[tris])
        D.append(dep[tris].mean(1))
        M.append(m)
    T, D, M = np.concatenate(T), np.concatenate(D), np.concatenate(M)
    for i in np.argsort(D):
        cv2.fillConvexPoly(img, np.round(T[i] * 16).astype(np.int32),
                           0 if M[i] else 180, lineType=cv2.LINE_8, shift=4)
    return img, x0, y0, S


def outline_bands(img, x0, y0, S):
    """Outline widths (drawing px) at fixed probes across the horizontal arms
    (W, E: vertical cuts) and the vertical arms (S, N: horizontal cuts)."""
    dark = img == 0

    def runs(line):
        out, i, n = [], 0, len(line)
        while i < n:
            if line[i]:
                j = i
                while j < n and line[j]:
                    j += 1
                out.append((i, j - i))
                i = j
            else:
                i += 1
        return out

    def band(mode, v, lo, hi):
        if mode == "col":
            line = dark[:, int((v - x0) * S)]
            r = [(p / S + y0, l / S) for p, l in runs(line)]
        else:
            line = dark[int((v - y0) * S), :]
            r = [(p / S + x0, l / S) for p, l in runs(line)]
        r = [l for p, l in r if lo <= p <= hi]
        return (r[0], r[-1]) if len(r) >= 2 else None

    probes = {"W": [("col", x, -150, 150) for x in (-520, -420, -320)],
              "E": [("col", x, -150, 150) for x in (380, 530, 630)],
              "S": [("row", y, -150, 150) for y in (250, 330, 410)]}
    out = {}
    for arm, ps in probes.items():
        out[arm] = [band(*p) for p in ps]
    return out


def blender_checks_ok(rep):
    cams = rep.get("cameras", {})
    return dict(
        loop_seamless=rep["loop_seam_max_err_m"] < 1e-5,
        face_rigid=rep["face_rigidity_max_err_px"] < 0.05,
        rest_equals_drawing=rep["rest_vs_base_max_err_m"] < 1e-5,
        cameras_see_mascot=bool(cams) and all(c["ok"] for c in cams.values()),
        few_flipped_tris=rep.get("flipped_front_tris_max", 10 ** 9) <= 150,
    )


def main(src, vdir):
    rep = json.load(open(os.path.join(vdir, "blender_checks.json")))
    m = rep["mapping"]
    cam = rep["cam_front"]
    S = m["scale"]
    hx, hy = m["hub_px"]
    Z0 = m["Z0"]
    rgba = cv2.imread(os.path.join(vdir, "front_1to1.png"), cv2.IMREAD_UNCHANGED)
    rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA).astype(np.float32)
    rh, rw = rgba.shape[:2]
    camX, _, camZ = cam["location"]
    # render pixel centre (u+0.5, v+0.5) -> drawing coordinate (x, y)
    ox = hx + S * camX - rw / 2 + 0.5
    oy = hy - S * (camZ - Z0) - rh / 2 + 0.5
    src_rgb = cv2.imread(src)[:, :, ::-1].astype(np.float32)
    H, W = src_rgb.shape[:2]
    # warp render into the drawing's pixel grid: drawing x = render u + ox
    M = np.float32([[1, 0, ox], [0, 1, oy]])
    warped = cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = warped[..., 3] / 255.0
    rgb = warped[..., :3]
    # un-premultiply is not needed: PNG from Blender is straight alpha

    # drawing masks
    sil_d = ndi.binary_fill_holes(src_rgb.min(2) < 128)
    G = src_rgb[..., 1]
    yellow_d = ndi.binary_fill_holes(sil_d & (G > 110))
    black_d = sil_d & ~yellow_d
    face_d = yellow_d & (G < 110)

    # render masks
    sil_r = alpha > 0.5
    # same criterion as the drawing: green channel half-way between black and yellow
    Gr = rgb[..., 1]
    dark_r = sil_r & (Gr < 110)
    # yellow region (with the face strokes filled in); face = dark inside it
    yel_fill_r = ndi.binary_fill_holes(sil_r & ~dark_r)
    yel_fill_r = cv2.morphologyEx(yel_fill_r.astype(np.uint8), cv2.MORPH_OPEN,
                                  np.ones((3, 3), np.uint8)).astype(bool)
    yel_fill_r = ndi.binary_fill_holes(yel_fill_r)
    n, lab = cv2.connectedComponents(yel_fill_r.astype(np.uint8))
    if n > 1:
        sizes = np.bincount(lab.ravel())[1:]
        yel_fill_r = lab == (1 + int(np.argmax(sizes)))
        yel_fill_r = ndi.binary_fill_holes(yel_fill_r)
    face_r = yel_fill_r & dark_r
    band_r = sil_r & ~yel_fill_r

    # boundary distance
    def boundary(mask):
        return mask & ~ndi.binary_erosion(mask)

    dt_r = ndi.distance_transform_edt(~boundary(sil_r))
    dt_d = ndi.distance_transform_edt(~boundary(sil_d))
    bd_d2r = dt_r[boundary(sil_d)]
    bd_r2d = dt_d[boundary(sil_r)]

    # outline width: distance from yellow edge to outside, along yellow edge
    def band_width(sil, yel):
        dt = ndi.distance_transform_edt(sil)
        return float(np.median(dt[boundary(yel)]))

    res = dict(
        silhouette_iou=iou(sil_d, sil_r),
        silhouette_boundary_px=dict(mean_d2r=float(bd_d2r.mean()), p95_d2r=float(np.percentile(bd_d2r, 95)),
                                    mean_r2d=float(bd_r2d.mean()), p95_r2d=float(np.percentile(bd_r2d, 95)),
                                    max_r2d=float(bd_r2d.max())),
        outline_band_iou=iou(black_d, band_r),
        outline_width_px=dict(drawing=band_width(sil_d, yellow_d), render=band_width(sil_r, yel_fill_r)),
        face_iou=iou(face_d, face_r),
        face_area_px=dict(drawing=int(face_d.sum()), render=int(face_r.sum())),
        yellow_render_median=np.median(rgb[yel_fill_r & ~face_r & (alpha > 0.99)], 0).tolist(),
        yellow_drawing_median=np.median(src_rgb[yellow_d & ~face_d], 0).tolist(),
    )
    # ---- outline width from the angles the videos use (geometry, rest pose) ----
    trace_path = os.path.join(os.path.dirname(os.path.abspath(vdir)), "iris_trace.npz")
    trace = np.load(trace_path)
    views = {"front": (0, 0), "tilt15": (15, 0), "orbit+20": (9, 20), "orbit-20": (9, -20)}
    bands, band_ok = {}, True
    for name, (el, yw) in views.items():
        b = outline_bands(*project_raster(trace, (hx, hy), el, yw))
        bands[name] = b
        for arm, probes in b.items():
            for pr in probes:
                if pr is None:
                    band_ok = False
                    continue
                lo, hi = min(pr), max(pr)
                if lo < 11.0 or hi > 26.0 or lo / hi < 0.6:
                    band_ok = False
    res["outline_bands_px"] = bands
    res["outline_bands_ok"] = band_ok

    # ---- folding: the outline loop must never cross itself ---------------------
    lf_path = os.path.join(vdir, "loop_frames.npz")
    if os.path.exists(lf_path):
        lf = np.load(lf_path)
        cross = {k: loop_crossings(lf[k]) for k in lf.files}
        res["loop_crossings"] = {k: v for k, v in cross.items() if v}
        res["loop_crossings_total"] = int(sum(cross.values()))

    checks = blender_checks_ok(rep)
    res["pass"] = dict(
        silhouette=res["silhouette_iou"] > 0.99,
        outline_front=abs(res["outline_width_px"]["render"] - res["outline_width_px"]["drawing"]) < 1.5,
        outline_angles=band_ok,
        face=res["face_iou"] > 0.9,
        no_fold_crossings=res.get("loop_crossings_total", 1) == 0,
        **checks,
    )
    res["all_pass"] = all(res["pass"].values())
    with open(os.path.join(vdir, "compare.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))

    ov = (src_rgb * 0.35 + 255 * 0.65).astype(np.uint8)
    ov[sil_d & ~sil_r] = (0, 200, 0)
    ov[sil_r & ~sil_d] = (255, 0, 255)
    ov[face_d & ~face_r] = (0, 140, 0)
    ov[face_r & ~face_d] = (200, 0, 200)
    cv2.imwrite(os.path.join(vdir, "compare_overlay.png"), ov[:, :, ::-1])
    side = np.concatenate([src_rgb, rgb * alpha[..., None] + 255 * (1 - alpha[..., None])], 1)
    cv2.imwrite(os.path.join(vdir, "compare_side_by_side.png"),
                np.clip(side, 0, 255).astype(np.uint8)[:, :, ::-1])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
