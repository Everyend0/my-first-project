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
from scipy.sparse import csgraph  # noqa: F401  (sparse.csgraph)
from skimage import measure
from skimage.morphology import skeletonize

# fill_strokes: inner ink lines that do NOT separate two body parts are painted on
#   the surface instead of being cut into it ("all" = every stroke, "spike" = the part
#   of a stroke farther than spike_d (2x px) from the back flap).  Cutting them made
#   deep grooves that looked like cracks and torn flakes from the diagonal views.
# flap_mode (piece between the legs): "fused" = the back peel strip grows out of the body
#   (one surface, smooth fillet, one outline) and droops down like a peel; the drawn
#   piece and its lines become part of the body, the lines are painted.
#   "strip" = the back strip of the peel as a separate piece, spread out
#   backwards like the legs (front: its end shows as the drawn triangle; black tip);
#   "mid" = sits in the middle between the legs, so
#   the back view shows the same split line (Λ + the line going up) as the front;
#   "behind" = a separate peel flap hanging behind the legs (rounded flap from behind).
CFG = {
    "banana_legs": dict(k=0.9, flap_k=0.42, flap_ext=56, flap_mode="fused", strip_len="floor", strip_tip=34, strip_rise=70,
                        strip_start_deg=28, strip_bend_deg=50, strip_curl_deg=0, strip_endk=1.0, strip_fillet=30.0, strip_shape="leg_copy", hollow_piece=True, front_leg=True, strip_flat=0.6, strip_smooth=14.0, strip_w0=50, strip_wm=50, strip_w1=42, strip_t0=45,
                        strip_t1=38,
                        fill_strokes="all", spike_d=40,
                        seam_name="legsplit", seam_back=False, paint_strokes_on_flap=True),
    "banana_arms": dict(k=0.9, flap_k=0.42, flap_ext=56, flap_mode="mid", fill_strokes="all", spike_d=0,
                        seam_name="armseam", seam_back=True, paint_strokes_on_flap=False),
}


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def poisson(mask, phi=None):
    """Solve -lap(u) = 2 on the unit grid, u = 0 on the boundary.  Without `phi` the
    boundary is the pixel mask.  With `phi` (signed distance, + inside) the boundary
    is phi's sub-pixel zero line (Shortley-Weller stencil: an inside node next to the
    boundary uses the true fractional distance to it) - no pixel staircase."""
    if phi is not None:
        mask = phi > 0
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    N = len(ys)
    idx = -np.ones(mask.shape, np.int64)
    idx[ys, xs] = np.arange(N)
    rows, cols, vals = [], [], []
    diag = np.zeros(N)
    for axis in (0, 1):
        h = {}
        nid = {}
        for sgn in (-1, 1):
            ny = ys + (sgn if axis == 0 else 0)
            nx = xs + (sgn if axis == 1 else 0)
            ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            j = np.full(N, -1)
            j[ok] = idx[ny[ok], nx[ok]]
            hh = np.ones(N)
            if phi is not None:
                out = j < 0
                pi = phi[ys[out], xs[out]]
                pj = np.full(out.sum(), -1.0)
                ok2 = ok[out]
                pj[ok2] = phi[ny[out][ok2], nx[out][ok2]]
                hh[out] = np.clip(pi / np.maximum(pi - pj, 1e-9), 0.05, 1.0)
            h[sgn], nid[sgn] = hh, j
        hl, hr = h[-1], h[1]
        for sgn, hs in ((-1, hl), (1, hr)):
            c = 2.0 / ((hl + hr) * hs)
            diag += c
            inn = nid[sgn] >= 0
            rows.append(np.arange(N)[inn])
            cols.append(nid[sgn][inn])
            vals.append(-c[inn])
    rows.append(np.arange(N))
    cols.append(np.arange(N))
    vals.append(diag)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    if phi is None:
        ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric")
        x = ml.solve(np.full(N, 2.0), tol=1e-9, accel="cg", maxiter=400)
    else:
        ml = pyamg.smoothed_aggregation_solver(A, symmetry="nonsymmetric")
        x = ml.solve(np.full(N, 2.0), tol=1e-9, accel="gmres", maxiter=400)
    u = np.zeros(mask.shape)
    u[ys, xs] = x
    return u


def edge_phi(mask, K=None, sigma=5.0, band=6.0):
    """Sub-pixel signed distance (2x px, + inside) to the part's edge.
    Near yellow/black ink edges it follows the drawing's anti-aliased line (K =
    blackness, edge at K = 0.5) instead of the pixel staircase; then the edge is
    low-passed along its length (sigma px): scan jitter with a period of 10-20 px is
    invisible from the front but, on the steep rim of an inflated shape, it tilts
    the surface by several degrees and shows up as glossy beads / ribbing.  The
    hand-drawn wobble (tens of px) is kept."""
    d_in = ndi.distance_transform_edt(mask)
    d_out = ndi.distance_transform_edt(~mask)
    sb = np.where(mask, d_in - 0.5, -(d_out - 0.5))
    phi0 = ndi.gaussian_filter(sb, 0.7)
    if K is not None:
        L = ndi.gaussian_filter(0.5 - K.astype(np.float64), 1.0)
        gy, gx = np.gradient(L)
        gm = np.hypot(gx, gy)
        dK = L / np.maximum(gm, 1e-6)
        valid = (np.abs(sb) < 2.5) & (gm > 0.08) & (np.abs(dK - sb) < 1.5)
        wt = ndi.gaussian_filter(valid.astype(np.float64), 1.0) * (1 - smoothstep(1.5, 4.0, np.abs(sb)))
        phi0 = (1 - wt) * phi0 + wt * np.clip(dK, -4, 4)
    d_s = smooth_edge_distance(phi0, sigma=sigma, band=band + 2)
    if d_s is None:
        return phi0
    wt = 1 - smoothstep(band - 2.0, band, np.abs(sb))
    ok = np.isfinite(d_s)
    return np.where(ok, (1 - wt) * phi0 + wt * np.nan_to_num(d_s), phi0)


def inflate_field(mask, U, K=None, blur=1.0):
    """Poisson inflation field on the 2x grid, in ORIGINAL-pixel^2 units, zero on the
    (sub-pixel, smoothed) edge and extended linearly outside it."""
    phi = edge_phi(mask, K)
    u = poisson(mask, phi=phi)
    inside = phi > 0
    # slope at the edge, averaged along the edge, carried outward linearly
    near = inside & (phi > 0.4) & (phi < 2.5)
    g = np.where(near, u / np.maximum(phi, 1e-6), 0.0)
    num = ndi.gaussian_filter(g, 6.0)
    den = ndi.gaussian_filter(near.astype(float), 6.0)
    Gs = num / np.maximum(den, 1e-9)
    has = den > 1e-3
    _, ind = ndi.distance_transform_edt(~has, return_indices=True)
    Gf = Gs[ind[0], ind[1]]
    uf = np.where(inside, u, Gf * phi)
    # inside, very close to the edge, use the same linear law (Poisson nodes there
    # are few and the stencil is least accurate)
    w = 1 - smoothstep(1.0, 3.0, phi)
    uf = np.where(inside, (1 - w) * uf + w * Gf * phi, uf)
    uf = ndi.gaussian_filter(uf, blur)
    return uf / (U * U)


def smooth_edge_distance(f, sigma, band=7.0):
    """signed distance (+ inside, pixels) to the zero contour of f after smoothing
    that contour along its arc length (gaussian, `sigma` px); NaN beyond `band`."""
    out = np.full(f.shape, np.nan)
    pts, nrm = [], []
    for c in measure.find_contours(f, 0.0):
        if len(c) < 30:
            continue
        closed = np.linalg.norm(c[0] - c[-1]) < 1.5
        seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
        sarc = np.concatenate([[0], np.cumsum(seg)])
        t = np.arange(0, sarc[-1], 0.5)
        y, x = np.interp(t, sarc, c[:, 0]), np.interp(t, sarc, c[:, 1])
        mode = "wrap" if closed else "nearest"
        ys = ndi.gaussian_filter1d(y, sigma / 0.5, mode=mode)
        xs = ndi.gaussian_filter1d(x, sigma / 0.5, mode=mode)
        ty, tx = np.gradient(ys), np.gradient(xs)
        n = np.hypot(ty, tx) + 1e-12
        ny, nx = tx / n, -ty / n
        probe = ndi.map_coordinates(f, [ys + 2 * ny, xs + 2 * nx], order=1)
        if np.mean(probe > 0) < 0.5:  # make the normal point inside (f > 0)
            ny, nx = -ny, -nx
        pts.append(np.stack([ys, xs], 1))
        nrm.append(np.stack([ny, nx], 1))
    if not pts:
        return None
    from scipy.spatial import cKDTree
    P, N = np.concatenate(pts), np.concatenate(nrm)
    # only pixels near the edge
    near = ndi.distance_transform_edt(~(np.abs(ndi.gaussian_filter(np.sign(f), 0.5)) < 0.999)) <= band
    yy, xx = np.nonzero(near)
    dist, j = cKDTree(P).query(np.stack([yy, xx], 1).astype(float))
    sgn = np.sign((yy - P[j, 0]) * N[j, 0] + (xx - P[j, 1]) * N[j, 1])
    out[yy, xx] = dist * np.where(sgn == 0, 1, sgn)
    return out


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


def hull_shell(u2, k, z_off, hmax, region, T1, U, seed, A=0.5, field=None, zrange=None, extra=None, zlo=None,
               zhi=None):
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
    if field is not None:  # custom solid: field(u1, zs) >= 0 inside, z within zrange
        zs = np.arange(np.floor(zrange[0] - M - 2), np.ceil(zrange[1] + M + 2) + 1, 1.0)
        solid = field(u1, zs) >= 0
    else:
        z_lo = np.min(zc) - hmax if zlo is None else min(np.min(zc) - hmax, zlo)
        z_hi = np.max(zc) + hmax if zhi is None else max(np.max(zc) + hmax, zhi)
        zs = np.arange(np.floor(z_lo - M - 2), np.ceil(z_hi + M + 2) + 1, 1.0)
        solid = np.empty(u1.shape + (len(zs),), bool)
        u1f = u1.astype(np.float32)[:, :, None]
        for a in range(0, len(zs), 64):  # in slabs: keeps memory low
            dz = ((zs[a:a + 64] - zc) / k).astype(np.float32) if np.ndim(zc) == 0 else (zs[a:a + 64] - zc) / k
            solid[:, :, a:a + 64] = (u1f - dz ** 2) >= 0
        del u1f
    if extra is not None:  # e.g. the back peel strip that grows out of the body
        extra(solid, x0 / U + 0.5 + np.arange(solid.shape[1]), y0 / U + 0.5 + np.arange(solid.shape[0]), zs)
    t = time.time()
    Tc = T1[y0 // U: y0 // U + u1.shape[0], x0 // U: x0 // U + u1.shape[1]].astype(np.float32)
    H = hull_field(solid, Tc, A, seed, M)
    del solid
    print(f"    hull field {H.shape} ({time.time() - t:.1f}s)", flush=True)
    return mc(H, (x0 / U + 0.5, y0 / U + 0.5, zs[0]), (1, 1, 1))


def hull_field(solid, Tc, A, seed, M, chunk=48):
    """T(x, y) * (1 + wobble) - distance-to-solid, computed in z-slabs: distances are
    only needed up to ~M, so each slab only needs M+ voxels of margin (exact)"""
    ny, nx, nz_ = solid.shape
    dmax = float(M) + 6.0
    mg = int(dmax) + 6
    noise = NoiseField(solid.shape, [(60, seed, 1.0), (30, seed + 100, 0.35)])
    H = np.empty(solid.shape, np.float32)
    for z0 in range(0, nz_, chunk):
        z1 = min(nz_, z0 + chunk)
        a, b = max(0, z0 - mg), min(nz_, z1 + mg)
        sub = solid[:, :, a:b]
        if sub.any():
            d = ndi.distance_transform_edt(~sub).astype(np.float32)
        else:
            d = np.full(sub.shape, dmax, np.float32)
        np.minimum(d, dmax, out=d)
        d = ndi.gaussian_filter(d, 1.0) - 0.5
        m = smoothstep(0.3, 0.75, np.abs(np.gradient(d, axis=2))).astype(np.float32)
        c0, c1 = z0 - a, z0 - a + (z1 - z0)
        H[:, :, z0:z1] = Tc[:, :, None] * (1 + A * noise.slab(z0, z1) * m[:, :, c0:c1]) - d[:, :, c0:c1]
        del d, m
    return H


class NoiseField:
    """noise_vol() evaluated slab by slab (same smooth random field, no full-size
    array): coarse random grids, trilinear interpolation with zoom's corner-aligned
    mapping, done one axis at a time"""

    def __init__(self, shape, octaves):
        self.shape = shape
        self.oct = []
        for wl, seed, amp in octaves:
            rng = np.random.default_rng(seed)
            g = rng.random(tuple(s // wl + 3 for s in shape)).astype(np.float32) - 0.5
            self.oct.append((g, wl, amp))

    @staticmethod
    def _lerp(arr, axis, n_out, idx):
        n_in = arr.shape[axis]
        n_full = n_in * 0 + n_out
        c = idx * (n_in - 1) / max(n_full - 1, 1)
        i0 = np.clip(np.floor(c).astype(int), 0, n_in - 2)
        f = (c - i0).astype(np.float32)
        a = np.take(arr, i0, axis=axis)
        b = np.take(arr, i0 + 1, axis=axis)
        sh = [1] * arr.ndim
        sh[axis] = len(idx)
        f = f.reshape(sh)
        return a * (1 - f) + b * f

    def slab(self, z0, z1):
        ny, nx, nz = self.shape
        out = None
        for g, wl, amp in self.oct:
            full = tuple(s * wl for s in g.shape)  # zoom output size before cropping
            t = self._lerp(g, 2, full[2], np.arange(z0, z1))
            t = self._lerp(t, 1, full[1], np.arange(nx))
            t = self._lerp(t, 0, full[0], np.arange(ny))
            out = amp * t if out is None else out + amp * t
        return out


def strip_curve(L, bend_deg, curl_deg, n=800, start_deg=0.0):
    """centreline of the back peel strip in the (y, z) plane, starting at (0, 0) and
    heading straight back (-z).  It bends down by `bend_deg` (negative = up) over the
    first 70 % of its length, then the end curls back by `curl_deg` - like a peel."""
    s = np.linspace(0.0, L, n)
    th = np.radians(start_deg) + np.radians(bend_deg - start_deg) * smoothstep(0.0, 0.7 * L, s) \
        - np.radians(curl_deg) * smoothstep(0.7 * L, L, s)
    ty, tz = np.sin(th), -np.cos(th)
    ds = s[1] - s[0]
    y = np.concatenate([[0.0], np.cumsum(0.5 * (ty[1:] + ty[:-1]) * ds)])
    z = np.concatenate([[0.0], np.cumsum(0.5 * (tz[1:] + tz[:-1]) * ds)])
    return dict(s=s, y=y, z=z, ty=ty, tz=tz, ny=-tz, nz=ty)


def sweep_coords(Y, Z, cv, yc, zf):
    """for points (Y, Z) (orig px): signed offset v across the strip (along its local
    'down' direction) and arc length s along it (s < 0 in front, s > L past the end)"""
    from scipy.spatial import cKDTree
    P = np.stack([cv["y"] + yc, cv["z"] + zf], 1)
    q = np.stack([np.ravel(Y), np.ravel(Z)], 1)
    _, k = cKDTree(P).query(q)
    dy, dz = q[:, 0] - P[k, 0], q[:, 1] - P[k, 1]
    v = dy * cv["ny"][k] + dz * cv["nz"][k]
    s = cv["s"][k] + dy * cv["ty"][k] + dz * cv["tz"][k]
    return v.reshape(np.shape(Y)), s.reshape(np.shape(Y))


def smax(a, b, r):
    """smooth union of two inside-positive distance fields: a fillet of radius ~r
    where they meet, so a branch grows out of the body instead of being stuck on"""
    h = np.clip(r - np.abs(a - b), 0, None) / r
    return np.maximum(a, b) + h * h * (0.25 * r)


def strip_setup(reg, U, L, tip, bend, curl, w0, w1, t0, t1, wm=None, rise=0.0, start_deg=0.0):
    """back peel strip = a flat peel ribbon (half-width w, half-thickness t, both
    tapering from root to tip) whose root sits inside the drawn piece between the legs"""
    ys, xs = np.nonzero(reg)
    # the root sits `rise` px above the drawn piece, inside the body, so the strip
    # peels off the back of the body and then bends down
    return dict(cv=strip_curve(L, bend, curl, start_deg=start_deg), yc=(ys.mean() + 0.5) / U - rise,
                xc=(xs.mean() + 0.5) / U,
                w0=w0, w1=w1, wm=w0 if wm is None else wm, t0=t0, t1=t1, L=L, tip=tip, U=U, bend=bend,
                curl=curl, ext=max(w0, t0, wm or 0) + 2)


def leg_copy_setup(m1, caps1, k_leg, smooth=6.0, flat=1.0, wsmooth=4.0, zdir=-1.0):
    """Third leg = an exact copy of the drawn leg that stands on the floor: its centre
    line (skeleton, junction -> foot), its width along that line and its black tip,
    turned 90 degrees about the vertical so it points backwards.  Same size, same
    length, same bend, same foot height as the drawn legs.  m1 / caps1: 1x masks."""
    from collections import deque
    sk = skeletonize(m1)
    nb = ndi.convolve(sk.astype(np.int32), np.ones((3, 3), np.int32), mode="constant") - sk
    ey, ex = np.nonzero(sk & (nb == 1))
    if len(ey) < 3:
        raise RuntimeError("leg copy: skeleton has fewer than 3 ends")
    order = np.argsort(ey)
    stem = (ey[order[0]], ex[order[0]])  # top end
    foot = (ey[order[-1]], ex[order[-1]])  # lowest end = the foot on the floor
    other = (ey[order[-2]], ex[order[-2]])

    def bfs(src):
        prev = {src: None}
        q = deque([src])
        while q:
            p = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    n = (p[0] + dy, p[1] + dx)
                    if n not in prev and 0 <= n[0] < sk.shape[0] and 0 <= n[1] < sk.shape[1] and sk[n]:
                        prev[n] = p
                        q.append(n)
        return prev

    prev = bfs(stem)

    def path_to(p):
        out = []
        while p is not None:
            out.append(p)
            p = prev[p]
        return out[::-1]  # stem -> p

    pf, po = path_to(foot), path_to(other)
    j = 0
    while j < min(len(pf), len(po)) and pf[j] == po[j]:
        j += 1
    leg = np.array(pf[j - 1:], float) + 0.5  # junction -> foot, pixel-centre coords (y, x)
    seg = np.linalg.norm(np.diff(leg, axis=0), axis=1)
    sarc = np.concatenate([[0], np.cumsum(seg)])
    s = np.arange(0, sarc[-1], 1.0)
    y = ndi.gaussian_filter1d(np.interp(s, sarc, leg[:, 0]), smooth, mode="nearest")
    x = ndi.gaussian_filter1d(np.interp(s, sarc, leg[:, 1]), smooth, mode="nearest")
    dt = ndi.distance_transform_edt(m1)
    w = ndi.gaussian_filter1d(ndi.map_coordinates(dt, [y - 0.5, x - 0.5], order=1), wsmooth, mode="nearest")
    incap = ndi.map_coordinates(ndi.binary_dilation(caps1, iterations=2).astype(np.float32),
                                [y - 0.5, x - 0.5], order=1) > 0.5
    tip = 0.0
    if incap[-1]:
        k = len(s) - 1
        while k > 0 and incap[k - 1]:
            k -= 1
        tip = float(s[-1] - s[k])
    # turn the leg's sideways run into a backwards (zdir=-1) or forwards (+1) run
    sgn = 1.0 if x[-1] < x[0] else -1.0
    zz = zdir * sgn * (x[0] - x)
    yy = y - y[0]
    ty, tz = np.gradient(yy), np.gradient(zz)
    nn = np.hypot(ty, tz) + 1e-9
    ty, tz = ty / nn, tz / nn
    cv = dict(s=s, y=yy, z=zz, ty=ty, tz=tz, ny=-tz, nz=ty)
    L = float(s[-1])
    return dict(cv=cv, yc=float(y[0]), xc=float(x[0]), L=L, tip=tip, sarr=s, warr=w, tarr=flat * w,
                ext=float(w.max()) + 2, endk=1.0, U=None, x0=float(x[0]), a=-zdir * sgn,
                info=dict(junction=[float(x[0]), float(y[0])], foot=[float(x[-1]), float(y[-1])],
                          direction="front" if zdir > 0 else "back"))


def strip_window(X, Y, c, pad):
    """index ranges of a grid (cell-centre coords X, Y) that cover the strip"""
    cv = c["cv"]
    xa, xb = c["xc"] - c["ext"] - pad, c["xc"] + c["ext"] + pad
    ya, yb = cv["y"].min() + c["yc"] - c["ext"] - pad, cv["y"].max() + c["yc"] + c["ext"] + pad
    xi = np.nonzero((X >= xa) & (X <= xb))[0]
    yi = np.nonzero((Y >= ya) & (Y <= yb))[0]
    if len(xi) == 0 or len(yi) == 0:
        return None
    return yi[0], yi[-1] + 1, xi[0], xi[-1] + 1


def strip_eval(X, Y, zs, c):
    """strip field (inside > 0, ~distance) on the grid X x Y x zs, plus arc length s(Y, Z)"""
    Yg, Zg = np.meshgrid(Y, zs, indexing="ij")
    v, ss = sweep_coords(Yg, Zg, c["cv"], c["yc"], 0.0)
    L = c["L"]
    if "sarr" in c:  # measured profiles (third leg = copy of a drawn leg)
        w = np.interp(ss, c["sarr"], c["warr"])[:, None, :]
        t = np.interp(ss, c["sarr"], c["tarr"])[:, None, :]
    else:
        q = smoothstep(0.0, L, ss)
        # width: slim where it leaves the body, widest a third of the way, slim at the tip
        w = (c["w0"] + (c["wm"] - c["w0"]) * smoothstep(0.0, 0.3 * L, ss)
             + (c["w1"] - c["wm"]) * smoothstep(0.3 * L, L, ss))[:, None, :]
        t = (c["t0"] + (c["t1"] - c["t0"]) * q)[:, None, :]
    e = (np.maximum(-ss, 0) + np.maximum(ss - L, 0))[:, None, :] / (c.get("endk", 1.5) * t)
    x = (np.asarray(X) - c["xc"])[None, :, None]
    r = np.sqrt((x / w) ** 2 + (v[:, None, :] / t) ** 2 + e ** 2)
    return ((1.0 - r) * np.minimum(w, t)).astype(np.float32), ss


def strip_info(c):
    cv, L = c["cv"], c["L"]
    samp = np.linspace(0, L, 8)
    if "sarr" in c:
        pts = [[float(c["xc"]), float(np.interp(q, cv["s"], cv["y"]) + c["yc"]),
                float(np.interp(q, cv["s"], cv["z"]))] for q in samp]
        rad = [float(0.5 * (np.interp(q, c["sarr"], c["warr"]) + np.interp(q, c["sarr"], c["tarr"])))
               for q in samp]
        return dict(points=pts, radius=rad, L=float(L), tip=float(c["tip"]), mode="leg_copy",
                    direction=c.get("info", {}).get("direction", "back"))
    pts = [[float(c["xc"]), float(np.interp(q, cv["s"], cv["y"]) + c["yc"]), float(np.interp(q, cv["s"], cv["z"]))]
           for q in samp]
    def wid(q):
        return c["w0"] + (c["wm"] - c["w0"]) * smoothstep(0.0, 0.3 * L, q) + (c["w1"] - c["wm"]) * smoothstep(0.3 * L, L, q)
    rad = [float(0.5 * (wid(q) + c["t0"] + (c["t1"] - c["t0"]) * smoothstep(0, L, q))) for q in samp]
    return dict(points=pts, radius=rad, L=L, bend=c["bend"], curl=c["curl"], tip=c["tip"])


def strip_pieces(u2, T1, reg, U, k_f, L, tip, bend, curl, seed, taper=0.35, k_b=1.0, A=0.2):
    """Back peel strip: the drawn piece between the legs (its inflation field u2 gives
    the cross-section) swept backwards along a bending centreline.  Seen from the
    front its near end is exactly the drawn shape.  Returns pieces + collider info."""
    cv = strip_curve(L, bend, curl)
    ys, xs = np.nonzero(reg)
    yc = (ys.mean() + 0.5) / U  # strip centre row (orig px) = drawn piece's centroid
    half = (ys.max() - ys.min()) / (2 * U) + 2
    umax = float(max(u2.max(), 1e-6))
    hmax = k_f * np.sqrt(umax)
    Tmax = float(T1.max())
    M = int(np.ceil(min(Tmax, 25.0) * (1 + A * 0.8))) + 4
    X = np.arange(np.floor(xs.min() / U) - M - 4, np.ceil(xs.max() / U) + M + 5) + 0.5
    Y = np.arange(np.floor(cv["y"].min() + yc - half) - M - 4, np.ceil(cv["y"].max() + yc + half) + M + 5) + 0.5
    zs = np.arange(np.floor(cv["z"].min() - half - M - 14), np.ceil(hmax + M + 6) + 1, 1.0)
    Yg, Zg = np.meshgrid(Y, zs, indexing="ij")
    v, ss = sweep_coords(Yg, Zg, cv, yc, 0.0)
    rows = yc + v  # cross-section row of each (Y, Z)
    R = np.broadcast_to((rows * U - 0.5)[:, None, :], (len(Y), len(X), len(zs)))
    C = np.broadcast_to((X * U - 0.5)[None, :, None], R.shape)
    ut = ndi.map_coordinates(u2, [R.ravel(), C.ravel()], order=1, mode="nearest").reshape(R.shape)
    Tt = ndi.map_coordinates(T1, [(R / U).ravel(), (C / U).ravel()], order=1, mode="nearest").reshape(R.shape)
    del R, C
    t = smoothstep(0.2 * L, L, ss)[:, None, :]
    F = ut - taper * umax * t - (np.maximum(-ss, 0) / k_f)[:, None, :] ** 2 \
        - (np.maximum(ss - L, 0) / k_b)[:, None, :] ** 2
    del ut
    origin = (X[0], Y[0], zs[0])
    # outline shell at the drawn line width of the matching cross-section point
    solid = F >= 0
    dist = ndi.distance_transform_edt(~solid).astype(np.float32)
    dist = ndi.gaussian_filter(dist, 1.0) - 0.5
    gz = np.abs(np.gradient(dist, axis=2))
    m = smoothstep(0.3, 0.75, gz)
    nz = noise_vol(dist.shape, 60, seed) + 0.35 * noise_vol(dist.shape, 30, seed + 100)
    H = Tt * (1 + A * nz * m) - dist
    del dist, gz, m, nz, Tt, solid
    out = {}
    hv, hf = mc(H.astype(np.float32), origin, (1, 1, 1))
    out["hull"] = (taubin(hv, hf), hf)
    del H
    gF = np.sqrt(sum(g ** 2 for g in np.gradient(F)))
    F = (F / np.maximum(gF, 1e-3)).astype(np.float32)
    del gF
    Sc = np.broadcast_to(((L - tip) - ss)[:, None, :], F.shape).astype(np.float32)
    for tag, fl in (("yellow", np.minimum(F, Sc)), ("cap", np.minimum(F, -Sc))):
        vv, ff = mc(fl, origin, (1, 1, 1))
        _, sv = sweep_coords(vv[:, 1], vv[:, 2], cv, yc, 0.0)
        out[tag] = (taubin(vv, ff, fixed=np.abs(sv - (L - tip)) < 1.5), ff)
    # centreline + radius for colliders (orig px; x = piece centre)
    xc = (xs.mean() + 0.5) / U
    rr = np.sqrt(2.0 * umax)  # ~ inscribed radius of the drawn piece
    samp = np.linspace(0, L, 7)
    pts = [[float(xc), float(np.interp(q, cv["s"], cv["y"]) + yc), float(np.interp(q, cv["s"], cv["z"]))] for q in samp]
    rad = [float(rr * np.sqrt(max(1 - taper * smoothstep(0.2 * L, L, q), 0.05))) for q in samp]
    return out, dict(points=pts, radius=rad, L=L, bend=bend, curl=curl, tip=tip)


def clean_mesh(v, f, min_faces=200, min_frac=0.02):
    """remove duplicate faces (incl. the same triangle wound both ways) and loose
    islands that are tiny next to the piece's biggest island; drop unused verts"""
    info = []
    key = np.sort(f, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    dup = cnt[inv.ravel()] > 1
    if dup.any():
        # keep one copy of a face only if all copies share the same winding
        keep = ~dup
        seen = {}
        for i in np.nonzero(dup)[0]:
            seen.setdefault(int(inv.ravel()[i]), []).append(i)
        for ids in seen.values():
            w = {tuple(np.roll(f[i], -int(np.argmin(f[i])))) for i in ids}
            if len(w) == 1:
                keep[ids[0]] = True
        info.append(f"{int((~keep).sum())} duplicate faces")
        f = f[keep]
    n = len(v)
    i = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    j = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    A = sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    ncomp, lab = sparse.csgraph.connected_components(A, directed=False)
    if ncomp > 1:
        fl = lab[f[:, 0]]
        sizes = np.bincount(fl, minlength=ncomp)
        big = sizes.max()
        ok = (sizes >= min_faces) | (sizes >= min_frac * big)
        drop = ~ok[fl]
        if drop.any():
            info.append(f"{int((~ok & (sizes > 0)).sum())} crumbs ({int(drop.sum())} faces)")
            f = f[~drop]
    used = np.zeros(n, bool)
    used[f.ravel()] = True
    remap = np.cumsum(used) - 1
    return v[used], remap[f].astype(np.int32), ", ".join(info)


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


def build(name, seg_npz, out_npz, reuse=None):
    """reuse: an earlier mesh npz of the same character - only the back strip (part 2)
    is rebuilt, everything else is copied (for quick previews of strip shapes)"""
    import os
    cfg = dict(CFG[name])
    for key in ("strip_len", "strip_tip", "strip_bend_deg", "strip_curl_deg", "strip_w0", "strip_wm", "strip_w1",
                "strip_rise", "strip_start_deg",
                "strip_t0", "strip_t1"):
        if os.environ.get("BANANA_" + key.upper()):
            val = os.environ["BANANA_" + key.upper()]
            cfg[key] = val if val == "floor" else float(val)
    d = np.load(seg_npz)
    U = int(d["U"])
    S, I, parts = d["S"], d["I"].copy(), d["parts"].copy()
    caps, face, strokes = d["caps"], d["face"], d["strokes"]
    Kd = d["K"].astype(np.float32)
    H2, W2 = S.shape
    pieces = {}
    meta = {"parts": []}
    B = S & ~I & ~strokes  # outer outline band

    D_bg_pre = ndi.distance_transform_edt(S)
    _eI = I & ~ndi.binary_erosion(I)
    _nB = ndi.binary_dilation(B, structure=np.ones((3, 3)), iterations=2)
    medT_pre = float(np.median(D_bg_pre[_eI & _nB])) / U
    del D_bg_pre, _eI, _nB

    # ---------- back peel strip / third leg that grows out of the body ----------
    strip = None
    strips = []
    if cfg["flap_mode"] == "fused" and parts.max() >= 2:
        reg_tri = parts == 2

        def make(L):
            c = strip_setup(reg_tri, U, L, float(cfg["strip_tip"]),
                            float(cfg["strip_bend_deg"]), float(cfg["strip_curl_deg"]),
                            *[float(cfg[k_]) for k_ in ("strip_w0", "strip_w1", "strip_t0", "strip_t1")],
                            wm=float(cfg["strip_wm"]), rise=float(cfg["strip_rise"]),
                            start_deg=float(cfg["strip_start_deg"]))
            c["endk"] = float(cfg.get("strip_endk", 1.5))
            return c

        def lowest(c):
            cv, L = c["cv"], c["L"]
            q = smoothstep(0.0, L, cv["s"])
            t = c["t0"] + (c["t1"] - c["t0"]) * q
            side = cv["y"] + t * np.abs(cv["ny"])
            end = cv["y"][-1] + c["endk"] * t[-1] * max(cv["ty"][-1], 0.0) + t[-1] * abs(cv["ny"][-1])
            return c["yc"] + max(side.max(), end)

        if cfg.get("strip_shape") == "leg_copy":
            m1 = (((parts > 0) | strokes)[::U, ::U]).copy()
            capf = signed_dist(caps, U).astype(np.float32)
            dirs = [-1.0] + ([1.0] if cfg.get("front_leg") else [])
            for zdir in dirs:
                c = leg_copy_setup(m1, caps[::U, ::U], float(cfg["k"]), smooth=float(cfg["strip_smooth"]),
                                   flat=float(cfg["strip_flat"]), wsmooth=float(cfg["strip_smooth"]), zdir=zdir)
                c["U"] = U
                c["capfield"] = capf
                strips.append(c)
                print(f"  {c['info']['direction']} leg = copy of the drawn standing leg: length {c['L']:.0f} px, "
                      f"max half-width {c['warr'].max():.0f} px", flush=True)
            strip = strips[0]
        elif cfg["strip_len"] == "floor":
            # long enough that its (yellow) foot stands on the floor like the drawn
            # feet: the outline then reaches the floor line
            ysS = np.nonzero(S)[0]
            target = (ysS.max() + 1) / U - medT_pre
            lo_, hi_ = 50.0, 1500.0
            for _ in range(40):
                mid = 0.5 * (lo_ + hi_)
                if lowest(make(mid)) < target:
                    lo_ = mid
                else:
                    hi_ = mid
            strip = make(lo_)
        else:
            strip = make(float(cfg["strip_len"]))
        if not strips:
            strips = [strip]
        if cfg.get("hollow_piece"):
            # per the user: the filled yellow face between the legs is removed - the
            # space is empty (background shows); the drawn lines around it stay as ink
            parts[reg_tri] = 0
            I = I & ~reg_tri
            reg_tri_removed = reg_tri
        else:
            parts[reg_tri] = 1  # the drawn piece is now part of the body (one surface)
        meta["strips"] = [strip_info(c) for c in strips]

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
    # the shells of both parts overlap inside the line, but each stays ~2.5 px clear of
    # the other part's surface (touching made shell and yellow interleave into speckles
    # when seen at an angle)
    T_inner = np.clip((D_other - 1.0) / U - 1.5, 0.3 * medT, medT)
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
        if reuse is not None and p == 1:
            continue
        if p == 1:
            k, z_off, region = cfg["k"], 0.0, reg
        elif cfg["flap_mode"] in ("mid", "strip"):
            # the piece between the legs, centred in depth like the legs' inner
            # edges: front and back look the same (drawn split line on both sides)
            k, z_off, region = cfg["flap_k"], 0.0, reg
        else:
            # back peel flap seen between the legs: a separate piece hanging BEHIND
            # the body; its hidden part tucks under the legs
            region, cf = flap_region(reg, main, strokes_cut, D_bg, medT, U, cfg["flap_ext"])
            meta["flap_arc_fit"] = [round(float(c), 6) for c in cf]
            k = cfg["flap_k"]
            # its ink line: drawn width on the visible bottom arc, inner-line weight
            # (like the drawn strokes) on the edge that only shows from behind
            # (the zone below the drawn arc keeps the measured width: the hull is
            # evaluated out there, one line width below the flap's edge)
            xr = np.nonzero(reg)[1]
            Xg = np.arange(W2)[None, :]
            below = (np.arange(H2)[:, None] >= np.polyval(cf, np.arange(W2))[None, :] - 4) \
                & (Xg >= xr.min()) & (Xg <= xr.max())
            w = np.clip(ndi.gaussian_filter((~below).astype(np.float32), 6.0), 0, 1)
            Tf2 = (1 - w) * T2 + w * np.minimum(T2, 0.72 * medT)
            Th = down2(Tf2)
        t = time.time()
        u2 = inflate_field(region, U, K=Kd)
        print(f"  part {p}: poisson {region.sum()} px ({time.time() - t:.1f}s)", flush=True)
        hmax = k * np.sqrt(max(u2.max(), 0))
        if p != 1 and cfg["flap_mode"] not in ("mid", "strip"):
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

        if p != 1 and cfg["flap_mode"] == "strip":
            # the back peel strip, swept backwards along a bending peel-like curve
            sp, info = strip_pieces(u2, T1, reg, U, k, float(cfg["strip_len"]), float(cfg["strip_tip"]),
                                    float(cfg["strip_bend_deg"]), float(cfg["strip_curl_deg"]), seed=p)
            for tag, (vv, ff) in sp.items():
                pieces[f"part{p}_{tag}_v"], pieces[f"part{p}_{tag}_f"] = vv, ff
            meta["parts"][-1].update(strip=info)
            meta["strip"] = info
            continue

        # 1x grid volume
        y0, y1, x0, x1 = bbox(region, 8)
        y0, x0 = y0 // 2 * 2, x0 // 2 * 2
        y1, x1 = min(H2, (y1 + 1) // 2 * 2), min(W2, (x1 + 1) // 2 * 2)
        u1 = down2(u2[y0:y1, x0:x1])
        zc = down2(z_off[y0:y1, x0:x1])[:, :, None] if np.ndim(z_off) == 2 else z_off
        z_lo = np.min(zc) - hmax - 3
        z_hi = np.max(zc) + hmax + 3
        use_strip = bool(strips) and p == 1
        if use_strip:
            z_lo = min([z_lo] + [c["cv"]["z"].min() - c["ext"] - 8 for c in strips])
            z_hi = max([z_hi] + [c["cv"]["z"].max() + c["ext"] + 8 for c in strips])
        zs = np.arange(np.floor(z_lo), np.ceil(z_hi) + 1, 1.0)
        dz = (zs[None, None, :] - zc).astype(np.float32)
        u1 = u1.astype(np.float32)
        F = u1[:, :, None] - (dz / np.float32(k)) ** 2
        # divide by |grad F| -> ~distance to the surface: linear across every voxel,
        # so marching cubes places vertices accurately also where the surface turns
        # steep near the rim (the raw quadratic field left lattice-moire 'beads')
        gy, gx = np.gradient(u1)
        if np.ndim(zc) == 3:
            zcy, zcx = np.gradient(zc[:, :, 0])
            Fx = gx[:, :, None] + 2 * dz / k ** 2 * zcx[:, :, None]
            Fy = gy[:, :, None] + 2 * dz / k ** 2 * zcy[:, :, None]
            gF = np.sqrt(Fx ** 2 + Fy ** 2 + (2 * dz / k ** 2) ** 2)
            del Fx, Fy
        else:
            gF = (gx ** 2 + gy ** 2)[:, :, None] + (2 * dz / np.float32(k ** 2)) ** 2
            np.sqrt(gF, out=gF)
        np.maximum(gF, 1e-3, out=gF)
        F /= gF
        F = F.astype(np.float32, copy=False)
        del gF, dz
        origin = (x0 / U + 0.5, y0 / U + 0.5, zs[0])
        Xm = x0 / U + 0.5 + np.arange(F.shape[1])
        Ym = y0 / U + 0.5 + np.arange(F.shape[0])
        fused = []  # (strip, window, Fs, ss) for every leg grown out of the body
        for c in (strips if use_strip else []):
            w_ = strip_window(Xm, Ym, c, 6)
            if w_ is None:
                continue
            a0, a1, b0, b1 = w_
            Fs, ss = strip_eval(Xm[b0:b1], Ym[a0:a1], zs, c)
            gS = np.sqrt(sum(g ** 2 for g in np.gradient(Fs)))
            Fs = (Fs / np.maximum(gS, 1e-3)).astype(np.float32)
            del gS
            F[a0:a1, b0:b1, :] = smax(F[a0:a1, b0:b1, :], Fs, float(cfg.get("strip_fillet", 12.0)))
            fused.append((c, w_, Fs, ss))
            print(f"  {c.get('info', {}).get('direction', 'strip')} leg fused into the body: window {Fs.shape}",
                  flush=True)
        win = fused[0][1] if fused else None

        def strip_solid(solid, X, Y, zz):
            for c in strips:
                w = strip_window(X, Y, c, 6)
                if w is not None:
                    Fh, _ = strip_eval(X[w[2]:w[3]], Y[w[0]:w[1]], zz, c)
                    solid[w[0]:w[1], w[2]:w[3], :] |= Fh >= 0

        # toon outline shell = true 3D offset surface of the part (no folds)
        hv, hf = hull_shell(u2, k, z_off, hmax, region, Th, U, seed=p, A=0.5 if p == 1 else 0.2,
                            extra=strip_solid if use_strip else None, zlo=z_lo if use_strip else None,
                            zhi=z_hi if use_strip else None)
        pieces[f"part{p}_hull_v"], pieces[f"part{p}_hull_f"] = taubin(hv, hf), hf

        cap_here = caps & ndi.binary_dilation(region, structure=disk(4))
        if win is not None:
            # black caps = drawn caps + the strip's tip (only inside the strip itself)
            sc = down2(signed_dist(cap_here, U)[y0:y1, x0:x1]) if cap_here.any() else \
                np.full(F.shape[:2], -50.0)
            Cc = np.broadcast_to(sc[:, :, None].astype(np.float32), F.shape).copy()
            for c, (a0, a1, b0, b1), Fs, ss in fused:
                if "capfield" in c:
                    # copied leg: its black foot is the drawn foot cap, carried over by
                    # the same turn that made the leg (drawing x -> depth)
                    Yg_, Zg_ = np.meshgrid(Ym[a0:a1], zs, indexing="ij")
                    Xd = c["x0"] + c["a"] * Zg_
                    capd = ndi.map_coordinates(c["capfield"], [Yg_ * U - 0.5, Xd * U - 0.5], order=1,
                                               mode="constant", cval=-50.0).astype(np.float32)
                    cut = np.broadcast_to(capd[:, None, :], Fs.shape)
                else:
                    # past the tip cut
                    cut = np.broadcast_to((ss[:, None, :] - (c["L"] - c["tip"])).astype(np.float32), Fs.shape)
                # ... AND inside that leg itself (the body and other legs stay yellow)
                Cc[a0:a1, b0:b1, :] = np.maximum(Cc[a0:a1, b0:b1, :], np.minimum(cut, Fs + 2.0))
            del fused
            for tag, sign in (("cap", 1.0), ("yellow", -1.0)):
                fld = Cc.copy() if sign > 0 else np.negative(Cc)
                np.minimum(fld, F, out=fld)
                v, f = mc(fld, origin, (1, 1, 1))
                del fld
                cv_ = ndi.map_coordinates(Cc, [v[:, 1] - origin[1], v[:, 0] - origin[0], v[:, 2] - origin[2]],
                                          order=1, mode="nearest")
                pieces[f"part{p}_{tag}_v"], pieces[f"part{p}_{tag}_f"] = taubin(v, f, fixed=np.abs(cv_) < 1.5), f
            del Cc
        elif cap_here.any():
            sc = down2(signed_dist(cap_here, U)[y0:y1, x0:x1])
            Sc = sc[:, :, None].astype(np.float32)  # same (distance) units as F
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
    def decal(mask, p, tag, a=2.0, delta=0.6, side=1, full=False, grow=0.3):
        """side=+1: front surface, side=-1: back surface (mirror image).
        full=True: the ink goes down through the body to just short of the opposite
        surface, so steep walls (e.g. where a painted line meets a cut) stay black."""
        u2, k, z_off, region = fields[p]
        y0, y1, x0, x1 = bbox(mask, 6)
        # +grow px: rendered ink edges lose ~0.3 px to anti-aliasing against yellow
        s = signed_dist(mask, U)[y0:y1, x0:x1] + grow
        zc = z_off[y0:y1, x0:x1] if np.ndim(z_off) == 2 else z_off
        hk = k * np.sqrt(np.maximum(u2[y0:y1, x0:x1], 0))
        hs = zc + side * hk
        if full:
            opp = zc - side * hk  # the opposite surface
            if side > 0:
                lo, hi = np.minimum(np.maximum(hs - 1e3, opp + 0.5), hs - a), hs + delta
            else:
                lo, hi = hs - delta, np.maximum(np.minimum(hs + 1e3, opp - 0.5), hs + a)
        else:
            # never reach through a thin part (e.g. the flap's rim) to its other side
            opp = zc - side * hk
            if side > 0:
                lo, hi = np.minimum(np.maximum(hs - a, opp + 0.3), hs - 0.5), hs + delta
            else:
                lo, hi = hs - delta, np.maximum(np.minimum(hs + a, opp - 0.3), hs + 0.5)
        msk = mask[y0:y1, x0:x1]
        zlo, zhi = np.floor(lo[msk].min() - 2), np.ceil(hi[msk].max() + 2)
        zs = np.arange(zlo, zhi + 0.5, 0.5)
        slab = np.minimum(zs[None, None, :] - lo[:, :, None], hi[:, :, None] - zs[None, None, :])
        Fd = np.minimum(s[:, :, None], slab).astype(np.float32)
        origin = ((x0 + 0.5) / U, (y0 + 0.5) / U, zs[0])
        pieces[f"{tag}_v"], pieces[f"{tag}_f"] = mc(Fd, origin, (0.5, 0.5, 0.5))

    if reuse is not None:
        old = np.load(reuse)
        om = json.loads(str(old["meta"]))
        for key in old.files:
            if key.endswith("_v") or key.endswith("_f"):
                if not key.startswith("part2_"):
                    pieces[key] = old[key]
        meta["parts"] = [q for q in om["parts"] if q["part"] == 1] + [q for q in meta["parts"] if q["part"] != 1]
        meta["pieces"] = sorted({k.rsplit("_", 1)[0] for k in pieces})
        np.savez_compressed(out_npz, T=T1.astype(np.float32), meta=json.dumps(meta), fill_mask=fill, **pieces)
        print(json.dumps(meta, indent=1))
        return

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
        # where a painted stroke continues as a cut line (leg split -> flap lines),
        # reach 6 px over the cut so no sliver of body or flap shows in between
        m = m | (ndi.binary_dilation(m, structure=disk(12)) & strokes_cut)
        decal(m, 1, f"{cfg['seam_name']}{j}", delta=1.4, full=True)
        if cfg["seam_back"]:
            decal(m, 1, f"{cfg['seam_name']}back{j}", delta=1.4, side=-1, full=True)
    if cfg["paint_strokes_on_flap"] and nparts >= 2 and cfg["flap_mode"] == "behind":
        # (only needed when the flap hangs behind the legs: then the flap's front shows
        # through the cut lines; with the flap in the middle the outline fills them)
        # the lines between legs and the back flap: paint them on the flap's front
        near_flap = ndi.distance_transform_edt(parts != 2) <= 30
        lab, n = ndi.label(strokes_cut, structure=np.ones((3, 3)))
        for i in range(1, n + 1):
            m = lab == i
            if m.sum() < 400 or not (m & near_flap).any():
                continue
            m = ndi.binary_dilation(m, structure=disk(2)) & fields[2][3]
            decal(m, 2, f"flapline{i}", a=2.0, delta=0.8)

    # mesh hygiene: drop marching-cubes crumbs (tiny loose islands) and doubled faces
    for name in sorted({k.rsplit("_", 1)[0] for k in pieces}):
        v, f, info = clean_mesh(pieces[name + "_v"], pieces[name + "_f"])
        pieces[name + "_v"], pieces[name + "_f"] = v, f
        if info:
            print(f"  clean {name}: {info}", flush=True)
    meta["pieces"] = sorted({k.rsplit("_", 1)[0] for k in pieces})
    np.savez_compressed(out_npz, T=T1.astype(np.float32), meta=json.dumps(meta), fill_mask=fill, **pieces)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3], reuse=sys.argv[4] if len(sys.argv) > 4 else None)
