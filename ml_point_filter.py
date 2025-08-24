# ml_point_filter.py
import argparse, math, numpy as np, torch, torch.nn as nn, cv2
from plyfile import PlyData, PlyElement

# ---------- I/O ----------
def read_ply(p):
    ply = PlyData.read(p); v = ply["vertex"].data
    X   = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)  # 仅为接口一致，后面不用颜色
    opa = np.asarray(v["opacity"], np.float32) if "opacity" in v.dtype.names else None
    scales = None
    if all(n in v.dtype.names for n in ["scale_0","scale_1","scale_2"]):
        scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32)
    return ply, v, X, fdc, opa, scales

def write_filtered_ply(v_in, mask, out_path):
    PlyData([PlyElement.describe(v_in[mask], 'vertex')], text=False).write(out_path)

# ---------- ground & axes ----------
def fit_plane(X, upv, iters=1500, th=0.04, upright_min_dot=0.85, seed=0):
    rng = np.random.default_rng(seed); best = (-1e9, None, None); N = len(X)
    for _ in range(iters):
        i = rng.choice(N, 3, replace=False); p1,p2,p3 = X[i]
        n = np.cross(p2-p1, p3-p1); L = np.linalg.norm(n)
        if L < 1e-8: continue
        n /= L
        if np.dot(n, upv) < 0: n = -n
        dot = float(np.dot(n, upv))
        if dot < upright_min_dot: continue
        d = np.abs((X - p1) @ n)
        score = int((d < th).sum()) + 0.1*N*dot
        if score > best[0]: best = (score, n.astype(np.float32), p1.astype(np.float32))
    return best[1], best[2], best[0]

def auto_ground(X, iters=1500, th=0.04):
    ny,py,sy = fit_plane(X, np.array([0,1,0],np.float32), iters, th)
    nz,pz,sz = fit_plane(X, np.array([0,0,1],np.float32), iters, th)
    if ny is None and nz is None:
        C = np.cov(X.T); w,v = np.linalg.eigh(C)
        n = v[:, np.argmin(w)].astype(np.float32); n /= (np.linalg.norm(n)+1e-12)
        p0 = X[np.argmin(X @ n)]
        return n, p0.astype(np.float32)
    if nz is None or (ny is not None and sy >= sz): return ny, py
    return nz, pz

def build_axes(n):
    ref = np.array([0,0,1],np.float32) if abs(n[2])<0.9 else np.array([1,0,0],np.float32)
    u = np.cross(ref, n); u /= np.linalg.norm(u)+1e-12
    v = np.cross(n, u);  v /= np.linalg.norm(v)+1e-12
    return u.astype(np.float32), v.astype(np.float32)

# ---------- grid stats ----------
def grid_indices(U,V, cell):
    u0, v0 = U.min(), V.min()
    ui = ((U-u0)/cell).astype(np.int32); vi = ((V-v0)/cell).astype(np.int32)
    H, W = vi.max()+1, ui.max()+1
    return ui,vi,H,W,u0,v0

def local_density(U,V, cell=0.12, smooth_px=1.5):
    ui,vi,H,W,u0,v0 = grid_indices(U,V,cell)
    cnt = np.zeros((H,W), np.float32); np.add.at(cnt, (vi,ui), 1.0)
    if smooth_px>0:
        cnt = cv2.GaussianBlur(cnt,(0,0),sigmaX=smooth_px,sigmaY=smooth_px)
    den = cnt[vi,ui]
    return den, (ui,vi,H,W)

def local_height_stats(h, grid, ksize=3):
    ui,vi,H,W = grid[0],grid[1],grid[2],grid[3]
    # Step 1: per-cell median
    med = np.full((H,W), np.nan, np.float32)
    # Efficiency: bucket points per cell, then compute medians
    buckets = {}
    for idx,(y,x) in enumerate(zip(vi,ui)):
        buckets.setdefault((y,x), []).append(idx)
    for (y,x), idxs in buckets.items():
        hh = h[idxs]
        med[y,x] = np.median(hh)
    # Fill empty cells using neighborhood averages
    mask = np.isnan(med).astype(np.uint8)
    if mask.any():
        med_filled = med.copy()
        med_filled[mask==1] = 0
        w = (~mask.astype(bool)).astype(np.float32)
        ker = np.ones((ksize,ksize), np.float32)
        med_num = cv2.filter2D(med_filled, -1, ker, borderType=cv2.BORDER_REPLICATE)
        med_den = cv2.filter2D(w, -1, ker, borderType=cv2.BORDER_REPLICATE) + 1e-6
        med = med_num/med_den
    # MAD
    # Simplify: approximate via |h - med_cell| quantile (no re-bucketing）
    med_cell = med[vi,ui]
    abs_dev = np.abs(h - med_cell)
    # Per-cell MAD
    MAD = np.full((H,W), np.nan, np.float32)
    for (y,x), idxs in buckets.items():
        MAD[y,x] = np.median(np.abs(h[idxs] - med[y,x]))
    # Fill missing MAD values
    mask2 = np.isnan(MAD).astype(np.uint8)
    if mask2.any():
        MAD_f = MAD.copy(); MAD_f[mask2==1]=0
        w2 = (~mask2.astype(bool)).astype(np.float32)
        ker = np.ones((ksize,ksize), np.float32)
        MAD_num = cv2.filter2D(MAD_f, -1, ker, borderType=cv2.BORDER_REPLICATE)
        MAD_den = cv2.filter2D(w2, -1, ker, borderType=cv2.BORDER_REPLICATE) + 1e-6
        MAD = MAD_num / MAD_den
    mad_cell = MAD[vi,ui]
    dev = abs_dev / (1.4826*mad_cell + 1e-3)
    return med_cell, mad_cell, dev

def scale_stats(scales):
    if scales is None:
        return np.zeros(1,np.float32),np.zeros(1,np.float32),np.ones(1,np.float32)
    s_mean = scales.mean(1)
    s_max  = scales.max(1)
    s_min  = scales.min(1)
    aniso  = (s_max+1e-6)/(s_min+1e-6)
    return s_mean, s_max, aniso

# ---------- features ----------
def build_features(X, fdc, opa, scales, n, p0):
    u,v = build_axes(n)
    U = (X - p0) @ u; V = (X - p0) @ v; h = (X - p0) @ n
    r = np.sqrt(U*U + V*V)
    den, grid = local_density(U,V, cell=0.12, smooth_px=1.5)
    med_h, mad_h, dev = local_height_stats(h, grid, ksize=3)
    s_mean, s_max, aniso = scale_stats(scales)
    feats = [h[:,None], r[:,None], den[:,None], dev[:,None], s_mean[:,None], s_max[:,None], aniso[:,None]]
    if opa is not None: feats.append(opa[:,None])
    F = np.concatenate(feats,1).astype(np.float32)
    mu = F.mean(0, keepdims=True); sd = F.std(0, keepdims=True)+1e-6
    Fz = (F-mu)/sd
    aux = {"h":h, "den":den, "dev":dev, "opa":opa, "U":U, "V":V}
    return Fz, aux

# ---------- weak labels ----------
def weak_labels(aux, q_pos_dev=1.5, q_neg_dev=3.5, q_hi_h=0.98, q_den_pos=0.60, q_den_neg=0.10, opa_min=0.03):
    h, den, dev, opa = aux["h"], aux["den"], aux["dev"], aux["opa"]
    N = len(h); y = np.full(N, -1, np.int64)
    # Positive: locally consistent height & high density & sufficiently opaque
    pos = (dev <= q_pos_dev) & (den >= np.quantile(den, q_den_pos))
    if opa is not None: pos &= (opa >= max(opa_min, np.quantile(opa,0.20)))
    # Negative: strong local outliers in height OR (very high height + very low density) OR very low opacity
    neg = (dev >= q_neg_dev) | ((h >= np.quantile(h, q_hi_h)) & (den <= np.quantile(den, q_den_neg)))
    if opa is not None: neg |= (opa < opa_min)
    y[pos] = 1; y[neg] = 0
    return y

# ---------- model ----------
class MLP(nn.Module):
    def __init__(self, d, h=96):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,1))
    def forward(self,x): return self.f(x).squeeze(-1)

def train(F, y, epochs=12, device="cpu"):
    idx = np.where(y!=-1)[0]
    if len(idx) < 2000:
        return np.ones(len(F), np.float32)*0.9
    pos = idx[y[idx]==1]; neg = idx[y[idx]==0]; m = min(len(pos), len(neg))
    rng = np.random.default_rng(0); pos=rng.choice(pos,m,False); neg=rng.choice(neg,m,False)
    tr=np.concatenate([pos,neg]); rng.shuffle(tr)
    X=torch.from_numpy(F[tr]).float().to(device); T=torch.from_numpy(y[tr]).float().to(device)
    w_pos=len(tr)/(2*max(1,len(pos))); w_neg=len(tr)/(2*max(1,len(neg)))
    W=torch.where(T>0.5, torch.full_like(T,w_pos), torch.full_like(T,w_neg))
    net=MLP(F.shape[1]).to(device); opt=torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bce=nn.BCEWithLogitsLoss(reduction="none")
    net.train()
    for ep in range(epochs):
        opt.zero_grad(); out=net(X); loss=(bce(out,T)*W).mean(); loss.backward(); opt.step()
        if (ep+1)%3==0:
            with torch.no_grad(): pr=torch.sigmoid(out)
            hard=((T<0.5)&(pr>0.7)).nonzero().squeeze(-1).cpu().numpy()
            if hard.size>0:
                extra=rng.choice(hard, min(2048,hard.size), replace=False)
                X=torch.cat([X,X[extra]],0); T=torch.cat([T,T[extra]],0); W=torch.cat([W,W[extra]],0)
    net.eval()
    with torch.no_grad(): P=torch.sigmoid(net(torch.from_numpy(F).float().to(device))).cpu().numpy().astype(np.float32)
    return P

# ---------- ROI lock ----------
def roi_lock(U,V,dev,P,thr=0.55, cell=0.14):
    keep = P >= thr
    ui,vi,H,W,u0,v0 = grid_indices(U,V,cell)
    mask = np.zeros((H,W), np.uint8); mask[vi[keep], ui[keep]] = 255
    num, lab = cv2.connectedComponents(mask,4)
    if num>1:
        areas=[(lab==i).sum() for i in range(1,num)]
        k=1+int(np.argmax(areas))
        core=(lab[vi,ui]==k)
    else:
        core=keep
    # Also include points with "consistent height" near the core to avoid fragmenting the main body
    near = (dev <= 2.5)
    final = core | near
    # Exclude clearly isolated sparse regions
    mask2 = np.zeros((H,W), np.uint8); mask2[vi[final], ui[final]] = 255
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), 1)
    num2, lab2 = cv2.connectedComponents(mask2,4)
    if num2>1:
        areas=[(lab2==i).sum() for i in range(1,num2)]
        k=1+int(np.argmax(areas))
        final = (lab2[vi,ui]==k)
    return final

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out_ply", default="filtered_v3.ply")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--thr", type=float, default=0.55)
    ap.add_argument("--opa_min", type=float, default=0.03)
    args=ap.parse_args()

    ply,v,X,fdc,opa,scales = read_ply(args.ply)
    n,p0 = auto_ground(X)
    u,vv = build_axes(n); U=(X-p0)@u; V=(X-p0)@vv

    F, aux = build_features(X, fdc, opa, scales, n, p0)
    y = weak_labels(aux, opa_min=args.opa_min)
    P = train(F, y, epochs=args.epochs, device=args.device)

    keep = roi_lock(U,V, aux["dev"], P, thr=args.thr, cell=0.14)
    write_filtered_ply(ply["vertex"].data, keep, args.out_ply)
    print(f"[done] keep {keep.sum()}/{len(keep)} → {args.out_ply}")

if __name__ == "__main__":
    main()
