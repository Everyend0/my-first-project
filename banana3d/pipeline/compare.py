"""Fidelity check: compare the exact orthographic front render with the drawing.

Prints silhouette / ink IoU and boundary-distance stats and writes an overlay:
  black = ink in both, red = ink only in drawing, blue = ink only in render,
  yellow = body in both.
"""
import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def masks_drawing(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    ink = np.maximum(r, g) < 119
    yel = (r - b > 100) & ~ink
    S = ndi.binary_fill_holes(ink | yel)
    return S, ink & S


def masks_render(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    ink = np.maximum(r, g) < 90
    yel = (r - b > 60) & ~ink
    # the floor seen edge-on below the feet renders as a dark full-width band
    band = ink.mean(axis=1) > 0.8
    ink[band] = False
    yel[band] = False
    S = ndi.binary_fill_holes(ink | yel)
    S = ndi.binary_opening(S, iterations=1)
    lab, n = ndi.label(S)
    # drop anything touching the frame (the floor seen edge-on below the feet)
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    S &= ~np.isin(lab, edge[edge > 0])
    lab, n = ndi.label(S)
    if n > 1:
        sizes = ndi.sum(S, lab, range(1, n + 1))
        S = lab == (1 + int(np.argmax(sizes)))
    return S, ink & S


def boundary_dist(A, B):
    eA = A & ~ndi.binary_erosion(A)
    eB = B & ~ndi.binary_erosion(B)
    dB = ndi.distance_transform_edt(~eB)
    d = dB[eA]
    return float(np.mean(d)), float(np.percentile(d, 95)), float(d.max())


def main(drawing, render, overlay):
    S0, K0 = masks_drawing(drawing)
    S1, K1 = masks_render(render)
    iou = lambda a, b: float((a & b).sum() / max((a | b).sum(), 1))
    res = dict(silhouette_iou=round(iou(S0, S1), 4), ink_iou=round(iou(K0, K1), 4))
    m, p95, mx = boundary_dist(S0, S1)
    res.update(silhouette_edge_mean_px=round(m, 2), silhouette_edge_p95_px=round(p95, 2),
               silhouette_edge_max_px=round(mx, 2))
    # inner edge of the ink outline = the body edge
    B0, B1 = S0 & ~K0, S1 & ~K1
    m, p95, mx = boundary_dist(ndi.binary_fill_holes(B0) & B0, ndi.binary_fill_holes(B1) & B1)
    res.update(body_edge_mean_px=round(m, 2), body_edge_p95_px=round(p95, 2))
    ov = np.full(S0.shape + (3,), 235, np.uint8)
    ov[S0 & S1] = (255, 236, 90)
    ov[K0 & K1] = (0, 0, 0)
    ov[K0 & ~K1] = (230, 0, 0)
    ov[K1 & ~K0] = (0, 90, 255)
    ov[S0 & ~S1 & ~K0] = (255, 150, 150)
    Image.fromarray(ov).save(overlay)
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:4])
