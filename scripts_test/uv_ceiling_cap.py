import argparse, numpy as np, cv2
from plyfile import PlyData, PlyElement

# ---------- I/O ----------
def read_ply(p):
    ply = PlyData.read(p); v = ply["vertex"].data
    X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    return ply, v, X

def write_ply_like(v_in, mask, out_path):
    PlyData([PlyElement.describe(v_in[mask], 'vertex')], text=False).write(out_path)

# ---------- Geometric primitives ----------
def fit_plane_auto(X, iters=1500, th=0.04):
    # Try y-up / z-up and choose the better one; fall back to PCA if both fail
    def fit_plane(X, upv, iters, th):
        rng = np.random.default_rng(0)
        best = (-1e9, None, None); N = len(X)
        for _ in range(iters):
            i = rng.choice(N, 3, replace=False)
            p1,p2,p3 = X[i]
            n = np.cross(p2-p1, p3-p1); L = np.linalg.norm(n)
            if L < 1e-8: continue
            n /= L
            if np.dot(n, upv) < 0: n = -n
            dot = float(np.dot(n, upv))
            if dot < 0.85: continue
            d = np.abs((X - p1) @ n)
            score = int((d < th).sum()) + 0.1*N*dot
            if score > best[0]:
                best = (score, n.astype(np.float32), p1.astype(np.float32))
        return best[1], best[2], best[0]

    ny,py,sy = fit_plane(X, np.array([0,1,0], np.float32), iters, th)
    nz,pz,sz = fit_plane(X, np.array([0,0,1], np.float32), iters, th)
    if ny is None and nz is None:
        C = np.cov(X.T); w,v = np.linalg.eigh(C)
        n = v[:, np.argmin(w)].astype(np.float32)
        n /= np.linalg.norm(n) + 1e-12
        p0 = X[np.argmin(X @ n)]
        return n, p0.astype(np.float32)
    if nz is None or (ny is not None and sy >= sz): return ny, py
    return nz, pz

def build_axes(n):
    ref = np.array([0,0,1], np.float32) if abs(n[2]) < 0.9 else np.array([1,0,0], np.float32)
    u = np.cross(ref, n); u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u);  v /= np.linalg.norm(v) + 1e-12
    return u.astype(np.float32), v.astype(np.float32)

# ---------- Core: per-cell “base height” + ceiling stripping ----------
def ceiling_cap(U, V, h, cell=0.22, p_base=0.35, k_local=2.0, global_mult=1.2):
    """
    Return a boolean keep mask:
      - Per-cell “base height” = p_base quantile (default 35% → biased to preserve the main body)
      - Per-cell tolerance = k_local * 1.4826 * MAD(cell heights)
      - Global tolerance = global_mult * IQR(h) (protects undulating main structures)
      - Threshold = base height + max(per-cell tol, global tol)
      - Only drop points with h > threshold (strictly peeling the “ceiling” above)
    """
    u0, v0 = U.min(), V.min()
    ui = ((U - u0) / cell).astype(np.int32)
    vi = ((V - v0) / cell).astype(np.int32)
    H, W = vi.max() + 1, ui.max() + 1

    # 1) Per-cell quantile (base)
    qmap = np.full((H, W), np.nan, np.float32)
    # 2) Per-cell MAD
    madmap = np.full((H, W), np.nan, np.float32)

    # Bucket points into cells
    buckets = {}
    for idx, (y, x) in enumerate(zip(vi, ui)):
        buckets.setdefault((y, x), []).append(idx)

    for (y, x), idxs in buckets.items():
        hh = h[idxs]
        qmap[y, x]   = np.quantile(hh, p_base)
        madmap[y, x] = np.median(np.abs(hh - np.median(hh)))

    # Fill empty cells (3x3 neighborhood average)
    def fill_nan(m):
        mask = np.isnan(m).astype(np.uint8)
        if not mask.any(): return m
        mf = m.copy(); mf[mask == 1] = 0
        w  = (~mask.astype(bool)).astype(np.float32)
        ker = np.ones((3,3), np.float32)
        num = cv2.filter2D(mf, -1, ker, borderType=cv2.BORDER_REPLICATE)
        den = cv2.filter2D(w,  -1, ker, borderType=cv2.BORDER_REPLICATE) + 1e-6
        return num / den

    qmap   = fill_nan(qmap)
    madmap = fill_nan(madmap)

    # Slight smoothing to reduce threshold jitter
    qmap   = cv2.GaussianBlur(qmap,   (0,0), sigmaX=1.0, sigmaY=1.0)
    madmap = cv2.GaussianBlur(madmap, (0,0), sigmaX=1.0, sigmaY=1.0)

    # Global tolerance (protect main structures): IQR * multiplier
    iqr = float(np.quantile(h, 0.75) - np.quantile(h, 0.25))
    tol_global = max(1e-3, iqr * float(global_mult))

    # Threshold = base + max(local, global)
    tol_local = k_local * 1.4826 * madmap
    tol = np.maximum(tol_local, tol_global).astype(np.float32)

    thr = qmap[vi, ui] + tol[vi, ui]
    keep = h <= thr

    # Peel only “large overhead clouds”: require a clear proportion within a 3x3 neighborhood (avoid killing isolated points)
    # Rasterize dropped points per cell
    drop_mask = np.zeros((H, W), np.uint16)
    for (y, x), idxs in buckets.items():
        d = np.count_nonzero(~keep[idxs])
        drop_mask[y, x] = d

    drop_blur = cv2.GaussianBlur(drop_mask.astype(np.float32), (0,0), 1.2)
    # If a cell’s drop is very isolated (still small after blur), undo the drop for that cell
    weak = drop_blur[vi, ui] < np.quantile(drop_blur, 0.60)
    keep[~weak & (~keep)] = keep[~weak & (~keep)]  
    keep[weak == True] = True 

    return keep

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out_ply", default="ceiling_capped.ply")
    ap.add_argument("--cell", type=float, default=0.22)
    ap.add_argument("--p_base", type=float, default=0.35)
    ap.add_argument("--k_local", type=float, default=2.0)
    ap.add_argument("--global_mult", type=float, default=1.2)
    args = ap.parse_args()

    ply, v, X = read_ply(args.ply)

    # Auto-align ground
    n, p0 = fit_plane_auto(X)
    # Build top-down coordinates
    ref = np.array([0,0,1], np.float32) if abs(n[2]) < 0.9 else np.array([1,0,0], np.float32)
    u = np.cross(ref, n); u /= np.linalg.norm(u) + 1e-12
    vv = np.cross(n, u);  vv /= np.linalg.norm(vv) + 1e-12
    U = (X - p0) @ u; V = (X - p0) @ vv; h = (X - p0) @ n

    keep = ceiling_cap(U, V, h, cell=args.cell, p_base=args.p_base,
                       k_local=args.k_local, global_mult=args.global_mult)

    write_ply_like(ply["vertex"].data, keep, args.out_ply)
    print(f"[done] kept {keep.sum()}/{len(keep)} → {args.out_ply}")

if __name__ == "__main__":
    import numpy as np
    main()
