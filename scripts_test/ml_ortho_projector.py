# ml_ortho_projector.py
import argparse, math
import numpy as np
import cv2
import torch
import torch.nn as nn
from plyfile import PlyData

Y00 = 0.2820947918

# ---------------- I/O: read 3DGS PLY ----------------
def read_3dgs_ply(p):
    ply = PlyData.read(p); v = ply["vertex"].data
    X   = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)

    rest = [n for n in v.dtype.names if n.startswith("f_rest_")]
    f_rest, deg = None, 0
    if rest:
        rest = sorted(rest, key=lambda s: int(s.split("_")[-1]))
        fr   = np.stack([v[n] for n in rest], 1).astype(np.float32)
        C    = fr.shape[1] // 3
        L    = int(round(math.sqrt(C + 1) - 1))
        deg  = max(0, L)
        f_rest = fr.reshape(-1, 3, C)

    rfs = [f for f in v.dtype.names if f.startswith("rot_")] \
       or [f for f in v.dtype.names if f.startswith("rotation")]
    Rquat = None
    if len(rfs) >= 4:
        Rquat = np.stack([v[rfs[i]] for i in range(4)], 1).astype(np.float32)

    opa = np.asarray(v["opacity"], np.float32) if "opacity" in v.dtype.names else None
    return X, fdc, f_rest, deg, Rquat, opa

def quat_to_R(q):
    q=np.asarray(q,np.float32)
    if q.ndim==1: q=q[None,:]
    # 兼容 xyzw / wxyz
    if np.mean(np.abs(q[:,3])) < np.mean(np.abs(q[:,0])):
        q=q[:,[1,2,3,0]]
    q=q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-12)
    x,y,z,w=q[:,0],q[:,1],q[:,2],q[:,3]
    R=np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],1).reshape(-1,3,3).astype(np.float32)
    return R

def eval_sh_basis(deg, d):
    x,y,z = d[...,0], d[...,1], d[...,2]
    out = [np.full_like(x, Y00, np.float32)]
    if deg>=1:
        c1=0.4886025119; out += [-c1*y, c1*z, -c1*x]
    if deg>=2:
        c2=[1.09254843059,0.315391565253,0.546274215296]
        out += [c2[0]*x*y,-c2[0]*y*z,c2[1]*(3*z*z-1.0),-c2[0]*x*z,c2[2]*(x*x-y*y)]
    if deg>=3:
        c3=[0.590043589926,2.89061144264,0.457045799464,0.37317633259,1.44530572132]
        out += [c3[0]*y*(3*x*x-y*y),-c3[1]*x*y*z,c3[2]*y*(5*z*z-1.0),
                -c3[3]*z*(5*z*z-3.0),c3[2]*x*(5*z*z-1.0),-c3[1]*z*(x*x-y*y),
                c3[4]*x*(x*x-3*y*y)]
    return np.stack(out,-1).astype(np.float32)

def sh_color_topdown(fdc, f_rest, deg, Rquat, n):
    if deg<=0 or f_rest is None or Rquat is None:
        return np.clip(1/(1+np.exp(-(fdc*Y00))),0,1)
    Rm=quat_to_R(Rquat); v=(-n).astype(np.float32); v/=np.linalg.norm(v)+1e-12
    dirs=(Rm.transpose(0,2,1)@v[None,:].repeat(Rm.shape[0],0)[...,None]).squeeze(-1)
    dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)+1e-12
    B=eval_sh_basis(deg,dirs); feats=np.concatenate([fdc[:,:,None],f_rest],2)
    rgb=1/(1+np.exp(-(feats*B[:,None,:]).sum(2)))
    return np.clip(rgb,0,1)

# ---------------- Ground plane & axes ----------------
def fit_plane_once(X, upv, iters=1500, th=0.04, upright_min_dot=0.85, seed=0):
    rng=np.random.default_rng(seed)
    best=(-1e9,None,None); N=X.shape[0]
    for _ in range(iters):
        i=rng.choice(N,3,False); p1,p2,p3=X[i]
        n=np.cross(p2-p1,p3-p1); L=np.linalg.norm(n)
        if L<1e-8: continue
        n/=L
        if np.dot(n,upv)<0: n=-n
        dot=float(np.dot(n,upv))
        if dot<upright_min_dot: continue
        d=np.abs((X-p1)@n); score=int((d<th).sum())+0.1*N*dot
        if score>best[0]: best=(score,n.astype(np.float32),p1.astype(np.float32))
    return best[1],best[2],best[0]

def auto_ground(X, iters=1500, th=0.04):
    ny,py,sy = fit_plane_once(X, np.array([0,1,0],np.float32), iters, th)
    nz,pz,sz = fit_plane_once(X, np.array([0,0,1],np.float32), iters, th)
    if ny is None and nz is None:
        C=np.cov(X.T); w,v=np.linalg.eigh(C)
        n=v[:,np.argmin(w)].astype(np.float32); n/=np.linalg.norm(n)+1e-12
        p0=X[np.argmin(X@n)]
        return n,p0.astype(np.float32)
    if nz is None or (ny is not None and sy>=sz): return ny,py
    return nz,pz

def build_axes(n):
    ref=np.array([0,0,1],np.float32) if abs(n[2])<0.9 else np.array([1,0,0],np.float32)
    u=np.cross(ref,n); u/=np.linalg.norm(u)+1e-12
    v=np.cross(n,u);   v/=np.linalg.norm(v)+1e-12
    return u.astype(np.float32), v.astype(np.float32)

# ---------------- ROI ----------------
def density_roi(U,V,cell=0.06,q=0.65,dilate_m=3.0):
    u0,v0=U.min(),V.min()
    ui=((U-u0)/cell).astype(np.int32); vi=((V-v0)/cell).astype(np.int32)
    H,W=vi.max()+1,ui.max()+1
    cnt=np.zeros((H,W),np.int32); np.add.at(cnt,(vi,ui),1)
    if cnt.max()==0: return np.ones_like(U,bool),(u0,v0,u0+W*cell,v0+H*cell)
    th=np.quantile(cnt[cnt>0],q); mask=(cnt>=max(1,th)).astype(np.uint8)
    rad=max(1,int(round(dilate_m/cell)))
    mask=cv2.dilate(mask,np.ones((2*rad+1,2*rad+1),np.uint8),1)
    nlab,lab=cv2.connectedComponents(mask,4)
    if nlab>1:
        areas=[int((lab==i).sum()) for i in range(1,nlab)]
        k=1+int(np.argmax(areas)); mask=(lab==k).astype(np.uint8)
    keep=mask[vi,ui].astype(bool)
    ys,xs=np.where(mask>0)
    umin=u0+xs.min()*cell; umax=u0+(xs.max()+1)*cell
    vmin=v0+ys.min()*cell; vmax=v0+(ys.max()+1)*cell
    return keep,(umin,vmin,umax,vmax)

def robust_bounds(U,V,q_low=0.30,q_high=0.70):
    umin,umax=np.quantile(U,[q_low,q_high]); vmin,vmax=np.quantile(V,[q_low,q_high])
    return float(umin),float(vmin),float(umax),float(vmax)

# ---------------- Soft splat with feats (fixed) ----------------
def softsplat_with_feats(X, rgb, n, p0, u, v, opa=None,
                         pix=0.010, pad=0.002, ss=3, beta=4.5,
                         q_low=0.30, q_high=0.70,
                         shift_u=0.0, shift_v=0.0, frame_scale=1.0):
    U=(X-p0)@u; V=(X-p0)@v; H=(X-p0)@n
    keep,(u0,v0,u1,v1)=density_roi(U,V,0.06,0.65,3.0)
    U,V,H,rgb = U[keep],V[keep],H[keep],rgb[keep]
    if opa is not None: opa=opa[keep]

    # 高度粗裁（去极端远景）
    h_lo,h_hi=np.quantile(H,[0.02,0.90])
    sel=(H>=h_lo)&(H<=h_hi)
    U,V,H,rgb = U[sel],V[sel],H[sel],rgb[sel]
    if opa is not None: opa=opa[sel]

    # 主体分位裁
    uu0,vv0,uu1,vv1=robust_bounds(U,V,q_low,q_high)
    u0,v0,u1,v1 = max(u0,uu0),max(v0,vv0),min(u1,uu1),min(v1,vv1)

    # pad & 平移/缩放
    pu=(u1-u0)*pad; pv=(v1-v0)*pad
    u0-=pu; u1+=pu; v0-=pv; v1+=pv
    base_w=(u1-u0); base_h=(v1-v0)
    cu=(u0+u1)/2;  cv=(v0+v1)/2
    half_u=base_w*0.5*frame_scale; half_v=base_h*0.5*frame_scale
    du=base_w*shift_u; dv=base_h*shift_v
    u0,u1 = cu-half_u+du, cu+half_u+du
    v0,v1 = cv-half_v+dv, cv+half_v+dv

    W=int(np.ceil((u1-u0)/pix))+1
    Ht=int(np.ceil((v1-v0)/pix))+1
    W_hi, H_hi = W*ss, Ht*ss
    pix_hi = pix/ss

    c=(U-u0)/pix_hi; r=(v1-V)/pix_hi
    eps=1e-6
    c=np.clip(c,0,W_hi-1-eps); r=np.clip(r,0,H_hi-1-eps)
    j=np.floor(c).astype(np.int32); i=np.floor(r).astype(np.int32)
    fx=c-j; fy=r-i
    w00=(1-fx)*(1-fy); w10=fx*(1-fy); w01=(1-fx)*fy; w11=fx*fy

    # 深度权重：越贴地权重越大（负号）
    z = (H - h_lo) / max(1e-6, (h_hi - h_lo))
    z = np.clip(z, 0.0, 1.0)
    wdepth = np.exp(-beta * z).astype(np.float32)
    wopa=np.clip(opa,0.0,1.0).astype(np.float32) if opa is not None else 1.0
    w = (wdepth*wopa).astype(np.float32)

    acc = np.zeros((H_hi*W_hi,3),np.float32)
    den = np.zeros((H_hi*W_hi,),np.float32)
    accH = np.zeros((H_hi*W_hi,),np.float32)
    accH2= np.zeros((H_hi*W_hi,),np.float32)
    flat=lambda ii,jj:(ii*W_hi+jj)

    def splat(wij,jj,ii):
        jj=np.clip(jj,0,W_hi-1); ii=np.clip(ii,0,H_hi-1)
        f=flat(ii,jj); ww=(wij*w).astype(np.float32)
        np.add.at(den, f, ww)
        np.add.at(acc, f, (rgb*ww[:,None]).astype(np.float32))
        np.add.at(accH,f, (H*ww).astype(np.float32))
        np.add.at(accH2,f,((H*H)*ww).astype(np.float32))

    splat(w00,j,i); splat(w10,j+1,i); splat(w01,j,i+1); splat(w11,j+1,i+1)

    img_hi  = acc.reshape(Ht*ss, W*ss, 3)
    Wsum_hi = den.reshape(Ht*ss, W*ss).astype(np.float32)
    Hsum_hi = accH.reshape(Ht*ss, W*ss).astype(np.float32)
    H2sum_hi= accH2.reshape(Ht*ss, W*ss).astype(np.float32)

    sigma=0.6*ss
    img_hi  = cv2.GaussianBlur(img_hi,(0,0),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REPLICATE)
    Wsum_hi = cv2.GaussianBlur(Wsum_hi,(0,0),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REPLICATE)
    Hsum_hi = cv2.GaussianBlur(Hsum_hi,(0,0),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REPLICATE)
    H2sum_hi= cv2.GaussianBlur(H2sum_hi,(0,0),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REPLICATE)

    # 广播安全
    img_hi   = img_hi / np.clip(Wsum_hi[...,None], 1e-6, None)
    Hmean_hi = Hsum_hi[...,None] / np.clip(Wsum_hi[...,None], 1e-6, None)
    Hvar_hi  = np.maximum(0.0, H2sum_hi[...,None]/np.clip(Wsum_hi[...,None],1e-6,None) - Hmean_hi*Hmean_hi)

    def down(x):
        C = x.shape[2]
        return x.reshape(Ht,ss,W,ss,C).mean(3).mean(1)

    img  = down(img_hi)
    wsum = down(Wsum_hi[...,None])[...,0]
    hmu  = down(Hmean_hi)[...,0]
    hvar = down(Hvar_hi)[...,0]

    return img.astype(np.float32), wsum.astype(np.float32), hmu.astype(np.float32), hvar.astype(np.float32), (W,Ht)

# ---------------- Teacher jitter (unified size) ----------------
def build_teacher(X, rgb, n, p0, u, v, opa, base_args, jit=6, beta_lo=4.0, beta_hi=5.0):
    imgs=[]; refHW=None
    rng=np.random.default_rng(0)
    for _ in range(jit):
        sh_u = rng.uniform(-0.01, 0.01)
        sh_v = rng.uniform(-0.01, 0.01)
        beta = rng.uniform(beta_lo, beta_hi)
        img, wsum, _, _, _ = softsplat_with_feats(
            X, rgb, n, p0, u, v, opa=opa,
            pix=base_args["pix"], pad=base_args["pad"], ss=max(3, base_args["ss"]),
            beta=beta, q_low=base_args["q_low"], q_high=base_args["q_high"],
            shift_u=base_args["shift_u"]+sh_u, shift_v=base_args["shift_v"]+sh_v,
            frame_scale=base_args["frame_scale"]
        )
        if refHW is None: refHW=(img.shape[0], img.shape[1])
        if (img.shape[0], img.shape[1]) != refHW:
            img  = cv2.resize(img,  (refHW[1], refHW[0]), interpolation=cv2.INTER_AREA)
            wsum = cv2.resize(wsum, (refHW[1], refHW[0]), interpolation=cv2.INTER_AREA)

        if (wsum>0).any():
            m = wsum > np.quantile(wsum[wsum>0], 0.25)
        else:
            m = np.ones_like(wsum,bool)
        im = img.copy(); im[~m] = np.nan
        imgs.append(im)

    stack = np.stack(imgs,0)
    teacher = np.nanmedian(stack, axis=0)

    nanmask = np.isnan(teacher[...,0])
    if nanmask.any():
        fill = np.nan_to_num(teacher, nan=0.0)
        w = (~nanmask).astype(np.float32)
        ker = np.ones((3,3), np.float32)
        num = cv2.filter2D(fill, -1, ker, borderType=cv2.BORDER_REPLICATE)
        den = cv2.filter2D(w,   -1, ker, borderType=cv2.BORDER_REPLICATE)[...,None] + 1e-6
        teacher = num / den
    return np.clip(teacher,0,1)

# ---------------- Light CNN refiner ----------------
class RefineNet(nn.Module):
    def __init__(self, in_ch=6, hid=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hid, 3, 3, padding=1)
        )
    def forward(self, x):
        base = x[:,0:3]
        res  = self.net(x)
        return torch.clamp(base + res, 0.0, 1.0)

def train_refiner(feats, teacher, iters=400, lr=1e-3, device="cpu", patch=256):
    H,W,C = feats.shape
    net = RefineNet(in_ch=C).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    loss_l1 = nn.L1Loss()

    def sample_patch():
        ph = min(patch, H); pw = min(patch, W)
        i = np.random.randint(0, H-ph+1); j = np.random.randint(0, W-pw+1)
        x = feats[i:i+ph, j:j+pw, :]
        y = teacher[i:i+ph, j:j+pw, :]
        return x, y

    net.train()
    for it in range(iters):
        x,y = sample_patch()
        x_t = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).float().to(device)
        y_t = torch.from_numpy(y.transpose(2,0,1)).unsqueeze(0).float().to(device)
        opt.zero_grad()
        out = net(x_t)
        loss = loss_l1(out, y_t)
        loss.backward(); opt.step()
        if (it+1)%100==0:
            print(f"[train] step {it+1}/{iters}  L1={loss.item():.4f}")
    return net

# ---------------- Tiled inference with cosine window ----------------
def infer_refiner(net, feats, device="cpu", tile=768, overlap=96):
    """
    瓦片推理 + 余弦窗加权融合，消除十字缝与块缝。
    """
    H, W, C = feats.shape
    out  = np.zeros((H, W, 3), np.float32)
    wacc = np.zeros((H, W, 1), np.float32)

    def cosine_window(h, w):
        wy = 0.5 * (1 - np.cos(2*np.pi*np.arange(h)/(h-1)))
        wx = 0.5 * (1 - np.cos(2*np.pi*np.arange(w)/(w-1)))
        ww = np.outer(wy, wx).astype(np.float32)
        ww = (ww + 0.5).clip(0.0, 1.0)  # 温和抬升
        return ww[..., None]

    step = tile - overlap
    win_full = cosine_window(min(tile, H), min(tile, W))

    net.eval()
    with torch.no_grad():
        for i0 in range(0, H, step):
            for j0 in range(0, W, step):
                i1 = min(H, i0 + tile)
                j1 = min(W, j0 + tile)
                x  = feats[i0:i1, j0:j1, :]
                x_t = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).float().to(device)
                y_t = net(x_t).cpu().numpy()[0].transpose(1,2,0)

                ww = win_full[:(i1-i0), :(j1-j0), :]
                out[i0:i1, j0:j1, :]  += y_t * ww
                wacc[i0:i1, j0:j1, :] += ww

    out /= np.clip(wacc, 1e-6, None)
    return np.clip(out, 0, 1)

# ---------------- Main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--pix", type=float, default=0.010)
    ap.add_argument("--ss",  type=int,   default=3)
    ap.add_argument("--beta",type=float, default=4.5)
    ap.add_argument("--pad", type=float, default=0.002)
    ap.add_argument("--q_low",  type=float, default=0.30)
    ap.add_argument("--q_high", type=float, default=0.70)
    ap.add_argument("--shift_u", type=float, default=0.0)
    ap.add_argument("--shift_v", type=float, default=0.0)
    ap.add_argument("--frame_scale", type=float, default=1.0)
    ap.add_argument("--teacher_jit", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps_per_epoch", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="ortho_ml.png")
    ap.add_argument("--save_teacher", default="")
    ap.add_argument("--save_baseline", default="")
    args=ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # 读取 + 地面对齐
    X, fdc, f_rest, L, Rquat, opa = read_3dgs_ply(args.ply)
    n,p0 = auto_ground(X)
    u,v = build_axes(n)
    rgb = sh_color_topdown(fdc, f_rest, L, Rquat, n)

    # 基线渲染
    base_args = dict(pix=args.pix, pad=args.pad, ss=args.ss,
                     q_low=args.q_low, q_high=args.q_high,
                     shift_u=args.shift_u, shift_v=args.shift_v,
                     frame_scale=args.frame_scale)
    base_img, wsum, hmu, hvar, (W,H) = softsplat_with_feats(
        X, rgb, n, p0, u, v, opa=opa, beta=args.beta, **base_args
    )
    if args.save_baseline:
        cv2.imwrite(args.save_baseline,
                    cv2.cvtColor((np.clip(base_img,0,1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # 教师图
    teacher = build_teacher(X, rgb, n, p0, u, v, opa, base_args, jit=args.teacher_jit)
    if args.save_teacher:
        cv2.imwrite(args.save_teacher,
                    cv2.cvtColor((teacher*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # CNN 特征
    f_w = wsum
    if (f_w>0).any():
        f_w = (f_w - np.quantile(f_w[f_w>0], 0.05)) / (np.quantile(f_w[f_w>0], 0.95)-np.quantile(f_w[f_w>0], 0.05)+1e-6)
        f_w = np.clip(f_w, 0, 1)
    f_hmu = (hmu - np.quantile(hmu, 0.05)) / (np.quantile(hmu, 0.95)-np.quantile(hmu, 0.05)+1e-6)
    f_hmu = np.clip(f_hmu, 0, 1)
    f_hvar = (hvar - np.quantile(hvar, 0.05)) / (np.quantile(hvar, 0.95)-np.quantile(hvar, 0.05)+1e-6)
    f_hvar = np.clip(f_hvar, 0, 1)
    feats = np.concatenate([base_img, f_w[...,None], f_hmu[...,None], f_hvar[...,None]], 2).astype(np.float32)

    # 训练
    iters = max(1, args.epochs) * max(50, args.steps_per_epoch)
    net = train_refiner(feats, teacher, iters=iters, lr=args.lr, device=args.device, patch=256)

    # 推理（余弦窗融合）
    pred = infer_refiner(net, feats, device=args.device, tile=768, overlap=96)

    # ---------- 置信度引导的教师混合 & 填洞 ----------
    conf = wsum.copy()
    pred = np.clip(pred, 0, 1)

    if (conf > 0).any():
        c_lo = np.quantile(conf[conf > 0], 0.15)
        c_hi = np.quantile(conf[conf > 0], 0.70)
        alpha = (conf - c_lo) / (c_hi - c_lo + 1e-6)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha = cv2.GaussianBlur(alpha, (0,0), 1.2)[..., None]
        pred = alpha * pred + (1.0 - alpha) * np.clip(teacher, 0, 1)

    mask = (conf < np.quantile(conf[conf > 0], 0.10) if (conf > 0).any() else (conf <= 0))
    mask = mask.astype(np.uint8) * 255
    if mask.any():
        pred8 = (pred * 255).astype(np.uint8)
        ker = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=1)
        for c in range(3):
            pred8[..., c] = cv2.inpaint(pred8[..., c], mask, 4, cv2.INPAINT_TELEA)
        pred = pred8.astype(np.float32) / 255.0

    # ---------- 边缘保持去色块 + 轻量锐化 ----------
    pred8 = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
    pred8 = cv2.bilateralFilter(pred8, d=0, sigmaColor=16, sigmaSpace=5)
    blur = cv2.GaussianBlur(pred8, (0,0), 1.1)
    pred8 = cv2.addWeighted(pred8, 1.18, blur, -0.18, 0)
    pred  = pred8.astype(np.float32) / 255.0

    out8 = (np.clip(pred,0,1)*255).astype(np.uint8)
    cv2.imwrite(args.out, cv2.cvtColor(out8, cv2.COLOR_RGB2BGR))
    print("saved:", args.out)

if __name__ == "__main__":
    main()
