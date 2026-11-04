"""Step 2 - turn the segmented drawing into closed 3D meshes.

Shape rule (keeps the drawing undeformed): every body part is "inflated" from its
exact 2D region.  We solve  -lap(u) = 2  inside the region (u = 0 on the traced
edge) and take the surface  u(x, y) = ((z - z_off) / k)^2 .  For a strip of
half-width r this is exactly a round tube of radius r (scaled by k in depth), so
seen from the front the silhouette is *identical* to the drawing, wobbles and all.

Meshes are extracted with marching cubes; colour regions (black tips, face, inner
ink lines) are cut out with min() of signed fields so their borders follow the
drawn ink edges instead of the triangle grid.

Output: work/<name>.mesh.npz with one entry per piece:
  <piece>_v (float32 Nx3, orig-pixel coords: x right, y down, z toward viewer)
  <piece>_f (int32 Mx3)
plus the outline-width field T (orig px) used for the toon outline shell.
"""
import json
import sys
import time

import numpy as np
import pyamg
from scipy import ndimage as ndi
from scipy import sparse
from skimage import measure

# fill_strokes: inner ink lines that do NOT separate two body parts are painted on
#   the surface instead of being cut into it ("all" = every stroke, "spike" = the part
#   of a stroke farther than spike_d (2x px) from the back flap).  Cutting them made
#   deep grooves that looked like cracks and torn flakes from the diagonal views.
CFG = {
    "banana_legs": dict(k=0.9, flap_k=0.42, flap_ext=56, fill_strokes="spike", spike_d=40,
                        seam_name="legsplit", seam_back=False, paint_strokes_on_flap=True),
    "banana_arms": dict(k=0.9, flap_k=0.42, flap_ext=56, fill_strokes="all", spike_d=0,
                        seam_name="armseam", seam_back=True, paint_strokes_on_flap=False),
}


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def poisson(mask):
    """Solve -lap(u) = 2 on mask (unit grid), u = 0 outside. Returns u (same grid)."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    N = len(ys)
    idx = -np.ones(mask.shape, np.int64)
    idx[ys, xs] = np.arange(N)
    rows, cols = [np.arange(N)], [np.arange(N)]
    vals = [np.full(N, 4.0)]
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        nid = np.full(N, -1)
        nid[ok] = idx[ny[ok], nx[ok]]
        ok = nid >= 0
        rows.append(np.arange(N)[ok])
        cols.append(nid[ok])
        vals.append(-np.ones(ok.sum()))
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric")
    x = ml.solve(np.full(N, 2.0), tol=1e-9, accel="cg", maxiter=400)
    u = np.zeros(mask.shape)
    u[ys, xs] = x
    return u


def inflate_field(mask, U, blur=2.0):
    """Poisson solution extended smoothly outside, zero level centred on the mask
    edge, returned in ORIGINAL-pixel^2 units on the 2x grid.

    The edge slope is averaged ALONG the edge before it is used: taking it per edge
    pixel gave every pixel-staircase step its own offset, which showed up as fine
    horizontal ribbing in glossy highlights at grazing angles (side views)."""
    # sub-pixel smoothing of the traced edge: removes raster/scan jitter (period of a
    # few px) but keeps the hand-drawn wobble (tens of px) - so the front outline
    # still matches the drawing while side views get clean highlights
    mask = ndi.gaussian_filter(mask.astype(np.float32), 3.0) > 0.5
    u = poisson(mask)
    d_out = ndi.distance_transform_edt(~mask)
    edge = mask & ~ndi.binary_erosion(mask)
    num = ndi.gaussian_filter(np.where(edge, u, 0.0), 3.0)
    den = ndi.gaussian_filter(edge.astype(float), 3.0)
    Gs = num / np.maximum(den, 1e-9)
    _, ind_e = ndi.distance_transform_edt(~edge, return_indices=True)
    Gf = Gs[ind_e[0], ind_e[1]]
    uf = np.where(mask, u - 0.5 * Gf, -Gf * (d_out - 0.5))
    uf = ndi.gaussian_filter(uf, blur)
    return uf / (U * U)


def signed_dist(mask, U, blur=0.8):
    """Signed distance in ORIGINAL px (positive inside) on the 2x grid."""
    d_in = ndi.distance_transform_edt(mask)
    d_out = ndi.distance_transform_edt(~mask)
    s = np.where(mask, d_in - 0.5, -(d_out - 0.5))
    return ndi.gaussian_filter(s, blur) / U


def down2(a):
    H, W = a.shape
    return a[: H // 2 * 2, : W // 2 * 2].reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))


def mc(vol, origin, spacing):
    """marching cubes; vol axes = (y, x, z). Returns verts (x, y, z) and faces."""
    t = time.time()
    v, f, _, _ = measure.marching_cubes(vol, level=0.0, spacing=spacing, allow_degenerate=False)
    out = np.empty_like(v)
    out[:, 0] = v[:, 1] + origin[0]
    out[:, 1] = v[:, 0] + origin[1]
    out[:, 2] = v[:, 2] + origin[2]
    print(f"    mc {vol.shape} -> {len(v)} v / {len(f)} f  ({time.time() - t:.1f}s)", flush=True)
    return out.astype(np.float32), f.astype(np.int32)


def taubin(v, f, fixed=None, iters=40, lam=0.5, mu=-0.53):
    """volume-preserving smoothing; removes marching-cubes terracing (scale ~2 px)
    while keeping the drawn wobble (20+ px) - vertices in `fixed` do not move"""
    n = len(v)
    i = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    A = sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    w = np.ones(n) if fixed is None else (~fixed).astype(float)
    v = v.astype(np.float64)
    for _ in range(iters):
        v += lam * w[:, None] * (A @ v / deg[:, None] - v)
        v += mu * w[:, None] * (A @ v / deg[:, None] - v)
    return v.astype(np.float32)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def noise_vol(shape, wl, seed):
    """smooth random field in [-0.5, 0.5] with feature size ~wl voxels"""
    rng = np.random.default_rng(seed)
    g = rng.random(tuple(s // wl + 3 for s in shape)).astype(np.float32) - 0.5
    z = ndi.zoom(g, wl, order=1)
    return z[: shape[0], : shape[1], : shape[2]]


def hull_shell(u2, k, z_off, hmax, region, T1, U, seed, A=0.5):
    """Ink outline shell: the surface at distance T from the body, where T is the
    line width measured in the drawing.  On the drawn silhouette rim the width is
    exact; on faces that only become silhouettes from other angles the width
    wobbles like a hand-drawn line."""
    M = int(np.ceil(T1.max() * (1 + A * 0.8))) + 4
    y0, y1, x0, x1 = bbox(region, 2 * M + 8)
    y0, x0 = y0 // 2 * 2, x0 // 2 * 2
    y1, x1 = min(u2.shape[0], (y1 + 1) // 2 * 2), min(u2.shape[1], (x1 + 1) // 2 * 2)
    u1 = down2(u2[y0:y1, x0:x1])
    zc = down2(z_off[y0:y1, x0:x1])[:, :, None] if np.ndim(z_off) == 2 else z_off
    zs = np.arange(np.floor(np.min(zc) - hmax - M - 2), np.ceil(np.max(zc) + hmax + M + 2) + 1, 1.0)
    solid = (u1[:, :, None] - ((zs[None, None, :] - zc) / k) ** 2) >= 0
    t = time.time()
    dist = ndi.distance_transform_edt(~solid).astype(np.float32)
    del solid
    dist = ndi.gaussian_filter(dist, 1.0) - 0.5
    gz = np.abs(np.gradient(dist, axis=2))
    m = smoothstep(0.3, 0.75, gz)
    del gz
    nz = noise_vol(dist.shape, 60, seed) + 0.35 * noise_vol(dist.shape, 30, seed + 100)
    Tc = T1[y0 // U: y0 // U + u1.shape[0], x0 // U: x0 // U + u1.shape[1]]
    H = Tc[:, :, None] * (1 + A * nz * m) - dist
    del dist, m, nz
    print(f"    hull field {H.shape} ({time.time() - t:.1f}s)", flush=True)
    return mc(H.astype(np.float32), (x0 / U + 0.5, y0 / U + 0.5, zs[0]), (1, 1, 1))


def bbox(mask, pad):
    ys, xs = np.nonzero(mask)
    return max(ys.min() - pad, 0), ys.max() + pad + 1, max(xs.min() - pad, 0), xs.max() + pad + 1


def flap_region(reg, main, strokes, D_bg, medT, U, R):
    """Whole back peel flap (2x grid).  Only `reg` is visible from the front; the
    rest hides behind the legs.  The hidden part is the drawn flap grown by R, cut
    below by the drawn bottom arc continued as a smooth curve, then rounded - so
    from behind it reads as ONE simple rounded flap (no steps or notches)."""
    H2, W2 = reg.shape
    ys, xs = np.nonzero(reg)
    cols = np.unique(xs)
    bot = np.array([ys[xs == c].max() for c in cols])
    on_arc = D_bg[bot, cols] / U <= 1.6 * medT  # bottom pixels that touch the outline arc
    if on_arc.sum() >= 5:
        cf = np.polyfit(cols[on_arc], bot[on_arc] + 0.5, 2)
    else:
        cf = np.array([0.0, 0.0, bot.max() + 0.5])
    X = np.arange(W2)
    yc = np.polyval(cf, X)
    above = np.arange(H2)[:, None] <= yc[None, :]
    allowed = reg | main | (ndi.distance_transform_edt(~strokes) <= 3)
    ext = (ndi.distance_transform_edt(~reg) <= R) & above & allowed
    ext = ndi.distance_transform_edt(ndi.distance_transform_edt(~ext) <= 12) > 12  # close
    ext = ndi.distance_transform_edt(~(ndi.distance_transform_edt(ext) > 14)) <= 14  # open (round corners)
    ext = (ext & above & allowed) | reg
    ext = ndi.distance_transform_edt(ndi.distance_transform_edt(~ext) <= 4) > 4
    return (ext & allowed) | reg, cf


def flap_depth(region, u_f, k_f, u_m, k_m, Tf2, U, pad=100):
    """Per-pixel depth centre (2x grid) for the flap: it follows the back of the
    body, sitting just far enough behind it that its own ink outline (width Tf)
    shows all round from behind and nothing pokes through the body."""
    H2, W2 = region.shape
    y0, y1, x0, x1 = bbox(region, pad)
    sl = (slice(y0, y1), slice(x0, x1))
    zb = -k_m * np.sqrt(np.maximum(u_m[sl], 0))  # body back surface (0 where no body)
    r = int(round(float(Tf2[sl][region[sl]].max()) * U))
    zb_n = ndi.grey_erosion(zb, footprint=disk(r))  # deepest body point within a line width
    hf = k_f * np.sqrt(np.maximum(u_f[sl], 0))
    need = np.maximum(hf + 1.5, 1.15 * Tf2[sl] + 1.5)
    c = zb_n - need
    z = c.copy()
    for _ in range(40):
        z = ndi.gaussian_filter(np.minimum(z, c), 4.0)
    reg_c = ndi.binary_dilation(region[sl], structure=disk(r))
    viol = float(np.max((z - c)[reg_c]))
    if viol > 0:
        z -= viol
    out = np.full((H2, W2), float(np.median(z[region[sl]])))
    out[sl] = z
    return out


def build(name, seg_npz, out_npz):
    cfg = CFG[name]
    d = np.load(seg_npz)
    U = int(d["U"])
    S, I, parts = d["S"], d["I"].copy(), d["parts"].copy()
    caps, face, strokes = d["caps"], d["face"], d["strokes"]
    H2, W2 = S.shape
    pieces = {}
    meta = {"parts": []}
    B = S & ~I & ~strokes  # outer outline band

    # ---------- inner strokes that get painted instead of cut --------------------
    fill = np.zeros_like(strokes)
    if cfg["fill_strokes"] == "all":
        fill = strokes.copy()
    elif cfg["fill_strokes"] == "spike" and parts.max() >= 2:
        cand = strokes & (ndi.distance_transform_edt(parts != 2) > cfg["spike_d"])
        lab, n = ndi.label(cand, structure=np.ones((3, 3)))
        for i in range(1, n + 1):
            if (lab == i).sum() >= 200:
                fill |= lab == i
    parts[fill] = 1
    I |= fill
    strokes_cut = strokes & ~fill

    # ---------- outline width field T (orig px) --------------------------------
    D_bg = ndi.distance_transform_edt(S)
    edgeI = I & ~ndi.binary_erosion(I)
    near_B = ndi.binary_dilation(B, structure=np.ones((3, 3)), iterations=2)
    medT = float(np.median(D_bg[edgeI & near_B])) / U
    # outline thickness = distance from the inner body edge to the outer ink edge;
    # anything implausibly thick (cap interiors, opened-away slivers) -> median
    good = edgeI & near_B & (D_bg / U <= 1.3 * medT)
    # along inner strokes between two parts (legs vs. back flap) the line is only as
    # wide as the gap between them - a wider shell would eat into the other part
    D_other = np.full(S.shape, np.inf)
    for q in range(1, int(parts.max()) + 1):
        dq = ndi.distance_transform_edt(parts != q)
        sel = edgeI & (parts > 0) & (parts != q)
        D_other[sel] = np.minimum(D_other[sel], dq[sel])
    T_inner = np.clip((D_other - 1.0) / U, 0.5 * medT, medT)
    T_edge = np.where(good, D_bg / U, T_inner)
    _, ind = ndi.distance_transform_edt(~edgeI, return_indices=True)
    T2 = T_edge[ind[0], ind[1]]
    T2 = ndi.gaussian_filter(T2, 2.0)
    T1 = down2(T2)
    meta["outline_median_px"] = medT

    # ---------- body parts ----------------------------------------------------
    nparts = int(parts.max())
    main = parts == 1
    fields = {}
    for p in range(1, nparts + 1):
        reg = parts == p
        Th = T1
        if p == 1:
            k, z_off, region = cfg["k"], 0.0, reg
        else:
            # back peel flap seen between the legs: a separate piece hanging BEHIND
            # the body; its hidden part tucks under the legs
            region, cf = flap_region(reg, main, strokes_cut, D_bg, medT, U, cfg["flap_ext"])
            meta["flap_arc_fit"] = [round(float(c), 6) for c in cf]
            k = cfg["flap_k"]
            # its ink line: drawn width on the visible bottom arc, inner-line weight
            # (like the drawn strokes) on the edge that only shows from behind
            arc_vis = reg & (D_bg / U <= 1.6 * medT)
            w = smoothstep(4.0, 14.0, ndi.distance_transform_edt(~arc_vis) / U)
            Tf2 = (1 - w) * T2 + w * np.minimum(T2, 0.72 * medT)
            Th = down2(Tf2)
        t = time.time()
        u2 = inflate_field(region, U)
        print(f"  part {p}: poisson {region.sum()} px ({time.time() - t:.1f}s)", flush=True)
        hmax = k * np.sqrt(max(u2.max(), 0))
        if p != 1:
            u_m, k_m = fields[1][0], fields[1][1]
            z_off = flap_depth(region, u2, k, u_m, k_m, Tf2, U)
            zr = z_off[region]
            meta["parts"].append(dict(part=p, k=k, z_centre_min=round(float(zr.min()), 2),
                                      z_centre_max=round(float(zr.max()), 2), hmax=round(float(hmax), 2),
                                      area_px=int(region.sum()) // (U * U)))
        else:
            meta["parts"].append(dict(part=p, k=k, z_off=0.0, hmax=round(float(hmax), 2),
                                      area_px=int(region.sum()) // (U * U)))
        fields[p] = (u2, k, z_off, region)

        # 1x grid volume
        y0, y1, x0, x1 = bbox(region, 8)
        y0, x0 = y0 // 2 * 2, x0 // 2 * 2
        y1, x1 = min(H2, (y1 + 1) // 2 * 2), min(W2, (x1 + 1) // 2 * 2)
        u1 = down2(u2[y0:y1, x0:x1])
        zc = down2(z_off[y0:y1, x0:x1])[:, :, None] if np.ndim(z_off) == 2 else z_off
        zs = np.arange(np.floor(np.min(zc) - hmax - 3), np.ceil(np.max(zc) + hmax + 3) + 1, 1.0)
        F = u1[:, :, None] - ((zs[None, None, :] - zc) / k) ** 2
        F = F.astype(np.float32)
        origin = (x0 / U + 0.5, y0 / U + 0.5, zs[0])
        # toon outline shell = true 3D offset surface of the part (no folds)
        hv, hf = hull_shell(u2, k, z_off, hmax, region, Th, U, seed=p, A=0.5 if p == 1 else 0.2)
        pieces[f"part{p}_hull_v"], pieces[f"part{p}_hull_f"] = taubin(hv, hf), hf

        cap_here = caps & ndi.binary_dilation(region, structure=disk(4))
        if cap_here.any():
            sc = down2(signed_dist(cap_here, U)[y0:y1, x0:x1])
            Gc = np.sqrt(np.maximum(u1, 0)) * 2.0 + 20.0  # ~|grad u| near the rim
            Sc = (sc * Gc)[:, :, None].astype(np.float32)
            for tag, fld in (("yellow", np.minimum(F, -Sc)), ("cap", np.minimum(F, Sc))):
                v, f = mc(fld, origin, (1, 1, 1))
                # keep the yellow/black border (and the internal cut) exactly in place
                sv = ndi.map_coordinates(sc, [v[:, 1] - origin[1], v[:, 0] - origin[0]], order=1, mode="nearest")
                pieces[f"part{p}_{tag}_v"], pieces[f"part{p}_{tag}_f"] = taubin(v, f, fixed=np.abs(sv) < 1.5), f
        else:
            v, f = mc(F, origin, (1, 1, 1))
            pieces[f"part{p}_yellow_v"], pieces[f"part{p}_yellow_f"] = taubin(v, f), f
        del F

    # ---------- thin black decals on a part's surface ----------------------------
    def decal(mask, p, tag, a=2.0, delta=0.6, side=1):
        """side=+1: front surface, side=-1: back surface (mirror image)"""
        u2, k, z_off, region = fields[p]
        y0, y1, x0, x1 = bbox(mask, 6)
        s = signed_dist(mask, U)[y0:y1, x0:x1]
        zc = z_off[y0:y1, x0:x1] if np.ndim(z_off) == 2 else z_off
        hs = zc + side * k * np.sqrt(np.maximum(u2[y0:y1, x0:x1], 0))
        lo, hi = (hs - a, hs + delta) if side > 0 else (hs - delta, hs + a)
        msk = mask[y0:y1, x0:x1]
        zlo, zhi = np.floor(lo[msk].min() - 2), np.ceil(hi[msk].max() + 2)
        zs = np.arange(zlo, zhi + 0.5, 0.5)
        slab = np.minimum(zs[None, None, :] - lo[:, :, None], hi[:, :, None] - zs[None, None, :])
        Fd = np.minimum(s[:, :, None], slab).astype(np.float32)
        origin = ((x0 + 0.5) / U, (y0 + 0.5) / U, zs[0])
        pieces[f"{tag}_v"], pieces[f"{tag}_f"] = mc(Fd, origin, (0.5, 0.5, 0.5))

    lab, n = ndi.label(face, structure=np.ones((3, 3)))
    for i in range(1, n + 1):
        decal(lab == i, 1, f"face{i}")

    # ---------- inner ink strokes -----------------------------------------------
    # painted strokes (arm seams, top of the leg split): a thin ink decal on the
    # body surface, ending where the body's outline ink starts
    lab, n = ndi.label(fill, structure=np.ones((3, 3)))
    j = 0
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < 400:
            continue
        j += 1
        decal(m, 1, f"{cfg['seam_name']}{j}")
        if cfg["seam_back"]:
            decal(m, 1, f"{cfg['seam_name']}back{j}", side=-1)
    if cfg["paint_strokes_on_flap"] and nparts >= 2:
        # the lines between legs and the back flap: paint them on the flap's front
        near_flap = ndi.distance_transform_edt(parts != 2) <= 30
        lab, n = ndi.label(strokes_cut, structure=np.ones((3, 3)))
        for i in range(1, n + 1):
            m = lab == i
            if m.sum() < 400 or not (m & near_flap).any():
                continue
            m = ndi.binary_dilation(m, structure=disk(2)) & fields[2][3]
            decal(m, 2, f"flapline{i}", a=2.0, delta=0.8)

    meta["pieces"] = sorted({k.rsplit("_", 1)[0] for k in pieces})
    np.savez_compressed(out_npz, T=T1.astype(np.float32), meta=json.dumps(meta), fill_mask=fill, **pieces)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3])
