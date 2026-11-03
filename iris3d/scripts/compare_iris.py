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
