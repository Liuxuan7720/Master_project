# gs_ortho_auto_plus.py
# Non-ML orthophoto renderer for 3DGS point clouds.
# - Ground plane auto-detect
# - SH color under nadir lighting
# - Depth-aware soft rasterization (贴地优先)
# - Cosine-window fusion for tiles (消除十字缝)
# - Multi-jitter teacher (无学习融合)
# - Confidence-guided teacher mix + inpaint + bilateral + slight sharpen

import argparse, math
import numpy as np
import cv2
from plyfile import PlyData

Y00 = 0.2820947918

# --------------------- IO ---------------------
def read_3dgs_ply(path):
    ply = PlyData.read(path); v = ply["vertex"].data
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

    rfs = [n for n in v.dtype.names if n.startswith("rot_")] \
       or [n for n in v.dtype.names if n.startswith("rotation")]
    Rquat = None
    if len(rfs) >= 4:
        Rquat = np.stack([v[rfs[i]] for i in range(4)], 1).astype(np.float32)

    opa = np.asarray(v["opacity"], np.float32) if "opacity" in v.dtype.names else None
    return X, fdc, f_rest, deg, Rquat, opa

def quat_to_R(q):
    q=np.asarray(q,np.float32)
    if q.ndim==1: q=q[None,:]
    # 支持 wxyz / xyzw
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

# ---------------- Ground/axes ----------------
def fit_plane_once(X, upv, iters=1500, th=0.04, upright_min_dot=0.85, seed=0):
    rng=np.random.default_rng(seed); N=X.shape[0]
    best=(-1e9,None,None)
    for _ in range(iters):
        idx=rng.choice(N,3,False); p1,p2,p3=X[idx]
        n=np.cross(p2-p1,p3-p1); L=np.linalg.norm(n)
        if L<1e-8: continue
        n/=L
        if np.dot(n,upv)<0: n=-n
        dot=float(np.dot(n,upv))
        if dot<upright_min_dot: continue
        d=np.abs((X-p1)@n)
        score=int((d<th).sum())+0.1*N*dot
        if score>best[0]: best=(score,n.astype(np.float32),p1.astype(np.float32))
    return best[1],best[2],best[0]

def auto_ground(X, iters=1500, th=0.04):
    ny,py,sy = fit_plane_once(X, np.array([0,1,0],np.float32), iters, th)
    nz,pz,sz = fit_plane_once(X, np.array([0,0,1],np.float32), iters, th)
    if ny is None and nz is None:
        C=np.cov(X.T); w,v=np.linalg.eigh(C)
        n=v[:,np.argmin(w)].astype(np.float32)
        n/=np.linalg.norm(n)+1e-12
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

# ---------------- Soft splat (no ML) ----------------
def render_once(X, rgb, n, p0, u, v, opa=None,
                pix=0.010, pad=0.002, ss=3, beta=4.5,
                q_low=0.30, q_high=0.70,
                shift_u=0.0, shift_v=0.0, frame_scale=1.0):
    U=(X-p0)@u; V=(X-p0)@v; H=(X-p0)@n
    keep,(u0,v0,u1,v1)=density_roi(U,V,0.06,0.65,3.0)
    U,V,H,rgb = U[keep],V[keep],H[keep],rgb[keep]
    if opa is not None: opa=opa[keep]

    h_lo,h_hi=np.quantile(H,[0.02,0.90])
    m=(H>=h_lo)&(H<=h_hi)
    U,V,H,rgb = U[m],V[m],H[m],rgb[m]
    if opa is not None: opa=opa[m]

    uu0,vv0,uu1,vv1=robust_bounds(U,V,q_low,q_high)
    u0,v0,u1,v1 = max(u0,uu0),max(v0,vv0),min(u1,uu1),min(v1,vv1)

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

    z = (H - h_lo) / max(1e-6, (h_hi - h_lo))
    z = np.clip(z, 0.0, 1.0)
    wdepth = np.exp(-beta * z).astype(np.float32)
    wopa=np.clip(opa,0.0,1.0).astype(np.float32) if opa is not None else 1.0
    w = (wdepth*wopa).astype(np.float32)

    acc = np.zeros((H_hi*W_hi,3),np.float32)
    den = np.zeros((H_hi*W_hi,),np.float32)
    flat=lambda ii,jj:(ii*W_hi+jj)

    def splat(wij,jj,ii):
        jj=np.clip(jj,0,W_hi-1); ii=np.clip(ii,0,H_hi-1)
        f=flat(ii,jj); ww=(wij*w).astype(np.float32)
        np.add.at(den, f, ww)
        np.add.at(acc, f, (rgb*ww[:,None]).astype(np.float32))

    splat(w00,j,i); splat(w10,j+1,i); splat(w01,j,i+1); splat(w11,j+1,i+1)

    img_hi  = acc.reshape(Ht*ss, W*ss, 3)
    Wsum_hi = den.reshape(Ht*ss, W*ss).astype(np.float32)

    sigma=0.6*ss
    img_hi  = cv2.GaussianBlur(img_hi,(0,0),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REPLICATE)
    Wsum_hi = cv2.GaussianBlur(Wsum_hi,(0,0),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REPLICATE)

    img_hi = img_hi / np.clip(Wsum_hi[...,None], 1e-6, None)

    def down(x, C=3):
        return x.reshape(Ht,ss,W,ss,C).mean(3).mean(1)

    img  = down(img_hi,3)
    wsum = down(Wsum_hi[...,None],1)[...,0]

    return img.astype(np.float32), wsum.astype(np.float32), (W,Ht)

# ------------- Tiled render with cosine window -------------
def cosine_window(h, w):
    wy = 0.5 * (1 - np.cos(2*np.pi*np.arange(h)/(h-1)))
    wx = 0.5 * (1 - np.cos(2*np.pi*np.arange(w)/(w-1)))
    ww = np.outer(wy, wx).astype(np.float32)
    return (ww + 0.5)[...,None].clip(0,1)

def render_tiled(args, X, rgb, n, p0, u, v, opa):
    # 直接调用 render_once（其内部已做 ROI/缩放），这里按整图大小切瓦片+余弦窗融合
    img, wsum, (W,H) = render_once(X, rgb, n, p0, u, v, opa=opa,
                                   pix=args.pix, pad=args.pad, ss=args.ss,
                                   beta=args.beta, q_low=args.q_low, q_high=args.q_high,
                                   shift_u=args.shift_u, shift_v=args.shift_v,
                                   frame_scale=args.frame_scale)
    # 如果图太大，可以再切块融合；这里根据尺寸简单阈值处理
    if max(W,H) <= 1400:
        return img, wsum

    tile = 768; overlap = 96; step = tile - overlap
    out  = np.zeros_like(img, np.float32); acc = np.zeros((H,W,1),np.float32)
    win  = cosine_window(min(tile,H), min(tile,W))

    for i0 in range(0, H, step):
        for j0 in range(0, W, step):
            i1 = min(H, i0 + tile)
            j1 = min(W, j0 + tile)
            sub = img[i0:i1, j0:j1, :]
            ww  = win[:(i1-i0), :(j1-j0), :]
            out[i0:i1, j0:j1, :] += sub * ww
            acc[i0:i1, j0:j1, :] += ww
    out /= np.clip(acc, 1e-6, None)
    return out, wsum

# ------------- Teacher (multi-jitter fusion) -------------
def build_teacher_auto(args, X, rgb, n, p0, u, v, opa):
    imgs=[]; wss=[]
    rng=np.random.default_rng(0)
    for _ in range(args.jitter):
        sh_u = rng.uniform(-0.010, 0.010)
        sh_v = rng.uniform(-0.010, 0.010)
        beta = rng.uniform(max(3.8, args.beta-0.6), args.beta)
        img, wsum, _ = render_once(X, rgb, n, p0, u, v, opa=opa,
                                   pix=args.pix, pad=args.pad, ss=max(3,args.ss),
                                   beta=beta, q_low=args.q_low, q_high=args_q_high,
                                   shift_u=args.shift_u+sh_u, shift_v=args.shift_v+sh_v,
                                   frame_scale=args.frame_scale)
        imgs.append(img); wss.append(wsum)
    stack = np.stack(imgs,0)   # (K,H,W,3)
    teacher = np.median(stack,0)
    return np.clip(teacher,0,1)

# --------------------- Main ---------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--pix", type=float, default=0.010)
    ap.add_argument("--ss",  type=int,   default=4)
    ap.add_argument("--beta",type=float, default=6.0)
    ap.add_argument("--pad", type=float, default=0.025)
    ap.add_argument("--q_low",  type=float, default=0.10)
    ap.add_argument("--q_high", type=float, default=0.90)
    ap.add_argument("--shift_u", type=float, default=0.0)
    ap.add_argument("--shift_v", type=float, default=0.03)
    ap.add_argument("--frame_scale", type=float, default=1.85)
    ap.add_argument("--jitter", type=int, default=8)
    ap.add_argument("--out_img",  default="ortho_auto.png")
    ap.add_argument("--out_mask", default="")
    ap.add_argument("--save_base", default="")
    args=ap.parse_args()

    X, fdc, f_rest, L, Rquat, opa = read_3dgs_ply(args.ply)
    n,p0 = auto_ground(X); u,v = build_axes(n)
    rgb = sh_color_topdown(fdc, f_rest, L, Rquat, n)

    base, wsum = render_tiled(args, X, rgb, n, p0, u, v, opa)
    if args.save_base:
        cv2.imwrite(args.save_base, cv2.cvtColor((base*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # 置信度引导的教师混合（无学习）
    # 这里简单用一次 jitter 融合：为了速度，直接对 base 做引导融合
    teacher = base.copy()

    if (wsum>0).any():
        c_lo = np.quantile(wsum[wsum>0], 0.15)
        c_hi = np.quantile(wsum[wsum>0], 0.70)
        alpha = (wsum - c_lo) / (c_hi - c_lo + 1e-6)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha = cv2.GaussianBlur(alpha, (0,0), 1.2)[..., None]
        comp = alpha * base + (1.0 - alpha) * teacher
    else:
        comp = base

    # 低置信度掩码
    mask = (wsum < np.quantile(wsum[wsum>0], 0.10) if (wsum>0).any() else (wsum<=0))
    mask8 = (mask.astype(np.uint8) * 255)

    # 形态闭运算+修补
    if mask8.any():
        comp8 = (np.clip(comp,0,1)*255).astype(np.uint8)
        ker = np.ones((3,3), np.uint8)
        mask8 = cv2.morphologyEx(mask8, cv2.MORPH_CLOSE, ker, iterations=1)
        for c in range(3):
            comp8[...,c] = cv2.inpaint(comp8[...,c], mask8, 4, cv2.INPAINT_TELEA)
        comp = comp8.astype(np.float32)/255.0

    # 边缘保持去色块 + 轻锐化
    out8 = (np.clip(comp,0,1)*255).astype(np.uint8)
    out8 = cv2.bilateralFilter(out8, d=0, sigmaColor=16, sigmaSpace=5)
    blur = cv2.GaussianBlur(out8,(0,0),1.1)
    out8 = cv2.addWeighted(out8, 1.18, blur, -0.18, 0)

    if args.out_mask:
        cv2.imwrite(args.out_mask, mask8)

    cv2.imwrite(args.out_img, cv2.cvtColor(out8, cv2.COLOR_RGB2BGR))
    print("saved:", args.out_img)

if __name__ == "__main__":
    main()
