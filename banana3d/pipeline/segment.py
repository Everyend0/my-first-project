"""Step 1 - segment a flat banana drawing into the pieces the 3D build needs.

Everything is done on a 2x bilinear-upsampled copy of the drawing so the traced
edges keep sub-pixel accuracy (the hand-drawn wobble must survive untouched).

Outputs (npz, all arrays at 2x resolution):
  S        filled silhouette (outer edge of the black outline)
  I        inner body region (silhouette minus the outline band and inner strokes)
  W        per-pixel outline width (orig px) taken from the nearest outer contour
  parts    int label image of the body parts (connected components of I)
  caps     black "tip" regions (stem cap, feet/arm tips)
  face     face decals (eyes, mouth)
  strokes  inner ink strokes (part separation lines)
"""
import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure

U = 2  # working upsample factor


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def segment(src, out_npz, debug_png):
    im = Image.open(src).convert("RGB")
    W0, H0 = im.size
    big = im.resize((W0 * U, H0 * U), Image.BILINEAR)
    a = np.asarray(big).astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]

    # continuous ink / background measures (anti-aliasing mixes linearly)
    K = np.clip(1.0 - np.maximum(r, g) / 238.0, 0, 1)      # blackness
    Gb = np.clip(b / 222.0, 0, 1)                            # gray-background-ness

    # background = gray pixels connected to the border
    nb = Gb > 0.5
    lab, n = ndi.label(nb)
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    border = border[border > 0]
    bg = np.isin(lab, border)
    S = ndi.binary_fill_holes(~bg)
    black = (K > 0.5) & S
    yellow = S & ~black

    # ---- outline width along the outer contour --------------------------------
    D_bg = ndi.distance_transform_edt(S)                    # dist to background
    C = S & ndi.binary_dilation(bg)                         # contour pixels
    Dy = ndi.distance_transform_edt(~yellow)                # dist to yellow
    med = np.median(Dy[C])
    # walk the outer contour in order and measure the line width along it
    conts = measure.find_contours(S.astype(np.float32), 0.5)
    pts_all, w_all = [], []
    for ct in conts:
        if len(ct) < 50:
            continue
        w = ndi.map_coordinates(Dy, [ct[:, 0], ct[:, 1]], order=1) - 0.5
        # black tips: there the ink is the cap, not the line -> the width reading is
        # too big.  Reject anything well above the *local* line width and bridge
        # the gap from both sides along the contour, then smooth along arc length.
        ok = w < 1.15 * med
        i = np.arange(len(w))
        w = np.interp(i, i[ok], w[ok], period=len(w))
        w = ndi.gaussian_filter1d(w, 8.0, mode="wrap")
        pts_all.append(ct)
        w_all.append(w)
    pts_all = np.concatenate(pts_all)
    w_all = np.concatenate(w_all)
    tree = cKDTree(pts_all)
    yy, xx = np.nonzero(ndi.binary_dilation(S, iterations=4))
    _, nn = tree.query(np.c_[yy, xx])
    wC = np.full(S.shape, med)
    wC[yy, xx] = w_all[nn]                                   # width of nearest contour

    # outer outline band: ink that lies within one line-width of the background
    B = black & (D_bg <= wC + 1.0)
    I0 = S & ~B

    # ---- classify the remaining ink ------------------------------------------
    ink = I0 & black
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    edt_black = ndi.distance_transform_edt(black)
    Bd = ndi.binary_dilation(B, structure=np.ones((3, 3)), iterations=2)
    caps = np.zeros_like(S)
    face = np.zeros_like(S)
    strokes = np.zeros_like(S)
    info = []
    for i in range(1, n + 1):
        m = lab == i
        area = int(m.sum())
        if area < 40:
            continue
        mx = float(edt_black[m].max())
        touches = bool((m & Bd).any())
        ratio = area / (mx * mx)
        ys, xs = np.nonzero(m)
        if not touches:
            kind = "face"
            face |= m
        elif mx < 24 and ratio > 6:
            kind = "stroke"
            strokes |= m
        else:
            kind = "cap"
            caps |= m
        info.append(dict(kind=kind, area=area, maxedt=round(mx, 1), ratio=round(ratio, 1),
                         bbox=[int(xs.min()) // U, int(ys.min()) // U, int(xs.max()) // U, int(ys.max()) // U]))

    # caps: extend the colour region slightly past the body edge so the 3D cut
    # reaches the rim cleanly (the outline hull covers the extra)
    caps_ext = ndi.binary_dilation(caps, structure=disk(6)) & black & S
    caps_ext |= caps

    I = I0 & ~strokes
    # tiny islands/spurs left by anti-aliasing
    I = ndi.binary_opening(I, structure=disk(1))
    lab_p, npart = ndi.label(I)
    sizes = ndi.sum(I, lab_p, range(1, npart + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s > 2000]
    parts = np.zeros(I.shape, np.int32)
    order = sorted(keep, key=lambda k: -sizes[k - 1])
    for j, k in enumerate(order):
        parts[lab_p == k] = j + 1
    I = parts > 0

    np.savez_compressed(out_npz, S=S, I=I, W=(wC / U).astype(np.float32), parts=parts,
                        caps=caps_ext, face=face, strokes=strokes, black=black,
                        K=K.astype(np.float16), U=U, size=(W0, H0))

    # debug overlay
    dbg = np.zeros(S.shape + (3,), np.uint8) + 230
    dbg[S] = (90, 90, 90)
    pal = [(255, 230, 60), (120, 200, 255), (255, 140, 200), (140, 255, 140)]
    for j in range(1, parts.max() + 1):
        dbg[parts == j] = pal[(j - 1) % len(pal)]
    dbg[caps_ext & ~I] = (40, 40, 160)
    dbg[caps & I] = (0, 0, 220)
    dbg[face] = (220, 0, 0)
    dbg[strokes] = (0, 160, 0)
    Image.fromarray(dbg).resize((W0, H0), Image.NEAREST).save(debug_png)
    summary = dict(size=[W0, H0], stroke_width_median=round(float(med) / U, 2),
                   parts=[int((parts == j).sum()) // (U * U) for j in range(1, parts.max() + 1)],
                   ink=info)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    segment(sys.argv[1], sys.argv[2], sys.argv[3])
