import os, argparse, math
import numpy as np
from plyfile import PlyData
from PIL import Image
import cv2

Y00 = 0.2820947918  # SH DC

# ---------- I/O ----------
def read_3dgs_ply(ply_path):
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)

    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
    rest_names = [n for n in v.dtype.names if n.startswith("f_rest_")]
    f_rest, sh_degree = None, 0
    if len(rest_names) > 0:
        rest_names = sorted(rest_names, key=lambda s: int(s.split("_")[-1]))
        f_rest_flat = np.stack([v[n] for n in rest_names], 1).astype(np.float32)
        C = f_rest_flat.shape[1] // 3
        L = int(round(math.sqrt(C + 1) - 1))
        sh_degree = max(0, L)
        f_rest = f_rest_flat.reshape(-1, 3, C)

    rgb_dc = 1.0 / (1.0 + np.exp(-(fdc * Y00)))

    S = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32)
    if np.mean(S) < 0 or np.max(S) < 1e-3:  # log-scale
        S = np.exp(S)

    opacity = np.asarray(v["opacity"], np.float32) if "opacity" in v.dtype.names else None
    if opacity is not None: opacity = np.clip(opacity, 0.0, 1.0)

    rot_fields = [f for f in v.dtype.names if f.startswith("rot_")] \
              or [f for f in v.dtype.names if f.startswith("rotation")]
    Rquat = None
    if len(rot_fields) >= 4:
        Rquat = np.stack([v[rot_fields[i]] for i in range(4)], 1).astype(np.float32)

    return X, rgb_dc, S, opacity, Rquat, fdc, f_rest, sh_degree

# ---------- SH ----------
def eval_sh_basis(deg, dirs):
    x, y, z = dirs[...,0], dirs[...,1], dirs[...,2]
    out = [np.full_like(x, Y00, dtype=np.float32)]
    if deg >= 1:
        c1 = 0.4886025119
        out += [-c1*y, c1*z, -c1*x]
    if deg >= 2:
        c2 = [1.09254843059, 0.315391565253, 0.546274215296]
        out += [c2[0]*x*y, -c2[0]*y*z, c2[1]*(3*z*z-1.0), -c2[0]*x*z, c2[2]*(x*x-y*y)]
    if deg >= 3:
        c3 = [0.590043589926, 2.89061144264, 0.457045799464, 0.37317633259, 1.44530572132]
        out += [
            c3[0]*y*(3*x*x - y*y),
            -c3[1]*x*y*z,
            c3[2]*y*(5*z*z - 1.0),
            -c3[3]*z*(5*z*z - 3.0),
            c3[2]*x*(5*z*z - 1.0),
            -c3[1]*z*(x*x - y*y),
            c3[4]*x*(x*x - 3*y*y)
        ]
    return np.stack(out, -1).astype(np.float32)

def sh_color_local(fdc, f_rest, deg, Rm, viewdir_world):
    if deg <= 0 or f_rest is None or Rm is None:
        return 1/(1+np.exp(-(fdc*Y00)))
    v = viewdir_world / (np.linalg.norm(viewdir_world)+1e-12)
    dirs = (Rm.transpose(0,2,1) @ v.astype(np.float32)).astype(np.float32)
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True)+1e-12)
    B = eval_sh_basis(deg, dirs)
    feats = np.concatenate([fdc[:,:,None], f_rest], axis=2)
    rgb_lin = (feats * B[:,None,:]).sum(axis=2)
    return 1/(1+np.exp(-rgb_lin))

# ---------- utils ----------
def rgb_to_sv(rgb):
    R,G,B = rgb[:,0], rgb[:,1], rgb[:,2]
    Cmax = np.maximum.reduce([R,G,B]); Cmin = np.minimum.reduce([R,G,B])
    V = Cmax
    S = np.where(Cmax < 1e-6, 0.0, (Cmax - Cmin) / (Cmax + 1e-6))
    blue_bias = B - np.maximum(R,G)
    return S, V, blue_bias

def quat_to_R(q):
    q = np.asarray(q, np.float32)
    if q.ndim == 1: q = q[None,:]
    if np.mean(np.abs(q[:,3])) < np.mean(np.abs(q[:,0])):  # (wxyz) -> (xyz w)
        q = q[:,[1,2,3,0]]
    q = q / (np.linalg.norm(q, axis=1, keepdims=True)+1e-12)
    x,y,z,w = q[:,0],q[:,1],q[:,2],q[:,3]
    R = np.stack([
        1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w),
        2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)
    ],1).reshape(-1,3,3).astype(np.float32)
    return R

def fit_plane_constrained(X, iters=3000, thresh=0.03, up_axis="auto",
                          upright_min_dot=0.85, align_weight=0.10, seed=123):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if N < 3:
        return np.array([0,1,0],np.float32), X.mean(0).astype(np.float32), np.ones((N,),bool)
    ups = [np.array([0,1,0],np.float32), np.array([0,0,1],np.float32)]
    if up_axis=="y": ups=[ups[0]]
    if up_axis=="z": ups=[ups[1]]
    best=(-1e9,None,None,None)
    for up in ups:
        for _ in range(iters):
            i = rng.choice(N,3,False); p1,p2,p3=X[i]
            n=np.cross(p2-p1,p3-p1); L=np.linalg.norm(n)
            if L<1e-8: continue
            n/=L
            if np.dot(n,up)<0: n=-n
            dot=float(np.dot(n,up))
            if dot<upright_min_dot: continue
            d=np.abs((X-p1)@n); inl=d<thresh; cnt=int(inl.sum())
            score=cnt+align_weight*N*dot
            if score>best[0]: best=(score,n,p1,inl)
    if best[1] is None:
        best=(-1,None,None,None)
        for _ in range(iters):
            i=rng.choice(N,3,False); p1,p2,p3=X[i]
            n=np.cross(p2-p1,p3-p1); L=np.linalg.norm(n)
            if L<1e-8: continue
            n/=L
            d=np.abs((X-p1)@n); inl=d<thresh; cnt=int(inl.sum())
            if cnt>best[0]: best=(cnt,n,p1,inl)
    return best[1].astype(np.float32), best[2].astype(np.float32), best[3]

def build_plane_axes(n):
    ref = np.array([0,0,1],np.float32) if abs(n[2])<0.9 else np.array([1,0,0],np.float32)
    u=np.cross(ref,n); u/= (np.linalg.norm(u)+1e-12)
    v=np.cross(n,u);   v/= (np.linalg.norm(v)+1e-12)
    return u,v

# ---------- render ----------
def render_ortho(
    X, rgb_dc, S, Rquat, opacity, fdc, f_rest, sh_degree, n, p0, u, v,
    pix=0.01, img_res=4096, downsample_to=2048,
    h_lo=-5.0, h_hi=50.0,
    sky_sat=0.16, sky_val=0.93, blue_bias_th=0.05,
    max_scale=-1.0, keep_lcc=True, fill=3, out="ortho.png",
    slab=3.0, density_q=0.20, roi_factor=3.0, roi_dilate_m=8.0, no_roi=True,
    scale_q=1.0, crop_quant=6.0,
    blend="norm", supersample=4.0,
    fp_gain=2.2,
    alpha_scale=6.0, s_min_px=5.0, s_max_px=24.0,
    gamma=1.02, gain=1.05, sat_boost=1.12,
    hole_fill=True, hole_tau=0.02, blur_px=0.0
):
    print(f"[stats] input points: {X.shape[0]}")
    Rm = quat_to_R(Rquat) if Rquat is not None else None

    # 颜色：SH 本地方向（俯视 -n）
    if sh_degree>0 and f_rest is not None and Rm is not None:
        rgb = sh_color_local(fdc, f_rest, sh_degree, Rm, -n)
    else:
        rgb = rgb_dc
    rgb = np.clip(rgb, 0, 1)

    # 初筛
    h = (X - p0) @ n
    S1,V1,blue = rgb_to_sv(rgb)
    sky_like = ((V1>=sky_val) & (S1<=sky_sat)) | (blue>blue_bias_th)
    if S.ndim==1 and (S.size%3==0): S=S.reshape(-1,3)
    maxaxis = S if S.ndim==1 else S.max(1)
    adapt = np.quantile(maxaxis, scale_q) if maxaxis.size>100 else np.max(maxaxis)
    too_big = (maxaxis > min(max_scale, adapt)) if max_scale > 0 else np.zeros_like(maxaxis, bool)
    keep = (h>=h_lo)&(h<=h_hi)&(~sky_like)&(~too_big)
    print(f"[stats] after color/scale/height: {int(keep.sum())}")

    if slab>0:
        med = np.median(h[keep]) if np.any(keep) else np.median(h)
        keep &= (np.abs(h-med)<=slab*0.5)
        print(f"[stats] after slab ±{slab/2:.2f}m: {int(keep.sum())}")

    X,rgb,S,h = X[keep], rgb[keep], S[keep], h[keep]
    opacity = opacity[keep] if opacity is not None else None
    Rm = Rm[keep] if Rm is not None else None
    if X.shape[0]==0:
        Image.fromarray(np.zeros((1024,1024,3),np.uint8)).save(out); return

    # ROI（可跳过）
    U_all=(X-p0)@u; V_all=(X-p0)@v
    if not no_roi:
        roi_pix=max(pix*roi_factor, pix)
        u0,v0=U_all.min(),V_all.min()
        ui=((U_all-u0)/roi_pix).astype(np.int32)
        vi=((V_all-v0)/roi_pix).astype(np.int32)
        Hr,Wr=vi.max()+1, ui.max()+1
        counts=np.zeros((Hr,Wr),np.int32); np.add.at(counts,(vi,ui),1)
        if counts.max()>0:
            th=np.quantile(counts[counts>0], density_q)
            mask=(counts>=max(1,th)).astype(np.uint8)
            rad=max(1,int(round(roi_dilate_m/roi_pix)))
            mask=cv2.dilate(mask, np.ones((2*rad+1,2*rad+1),np.uint8),1)
            keep_roi=mask[vi,ui].astype(bool)
        else:
            keep_roi=np.ones_like(ui,bool)
    else:
        keep_roi=np.ones_like(U_all,bool)

    X,rgb,S,h = X[keep_roi], rgb[keep_roi], S[keep_roi], h[keep_roi]
    opacity = opacity[keep_roi] if opacity is not None else None
    U, V = U_all[keep_roi], V_all[keep_roi]
    Rm = Rm[keep_roi] if Rm is not None else None
    print(f"[stats] after ROI: {X.shape[0]}")

    # 画幅
    q1,q2 = np.clip(crop_quant,0,40), np.clip(100-crop_quant,60,100)
    umin,umax = np.percentile(U,[q1,q2]); vmin,vmax = np.percentile(V,[q1,q2])
    pad_u=(umax-umin)*0.03; pad_v=(vmax-vmin)*0.03
    umin-=pad_u; umax+=pad_u; vmin-=pad_v; vmax+=pad_v

    W=int(np.ceil((umax-umin)/pix))+1; H=int(np.ceil((vmax-vmin)/pix))+1
    cols=np.clip(((U-umin)/pix).astype(np.int32),0,W-1)
    rows=np.clip(((vmax-V)/pix).astype(np.int32),0,H-1)

    # 深度排序
    order=np.argsort(h)
    cols,rows=cols[order],rows[order]; rgb=rgb[order]
    opacity = opacity[order] if opacity is not None else None
    S = S[order] if S.ndim==2 else S[order]
    Rm = Rm[order] if Rm is not None else None

    # 超采样
    base_scale = img_res / max(H, W)
    scale_up = base_scale * float(supersample)
    Hs, Ws = int(H*scale_up), int(W*scale_up)
    rows_s=(rows*scale_up).astype(int); cols_s=(cols*scale_up).astype(int)

    # 归一化混合
    img_acc = np.zeros((Hs, Ws, 3), np.float32)
    w_acc   = np.zeros((Hs, Ws), np.float32)

    P=np.stack([u,v],0).astype(np.float32)

    for i in range(len(rows_s)):
        r0,c0=rows_s[i],cols_s[i]
        s = S[i] if S.ndim==2 else np.array([S[i],S[i],S[i]],np.float32)
        R3 = Rm[i] if Rm is not None else np.eye(3, dtype=np.float32)
        C3 = (R3 @ np.diag(s**2) @ R3.T).astype(np.float32)
        C2 = (P @ C3 @ P.T).astype(np.float32)

        evals,evecs=np.linalg.eigh(C2 + 1e-8*np.eye(2, dtype=np.float32))
        sigM, sigm = math.sqrt(max(evals[1],1e-8)), math.sqrt(max(evals[0],1e-8))
        # 放大足迹，抑制马赛克/撕裂
        sM=max(sigM/pix*scale_up, s_min_px); sm=max(sigm/pix*scale_up, 0.8)
        sM=min(sM*fp_gain, s_max_px);       sm=min(sm*fp_gain, s_max_px)

        R2=evecs; Win=int(4*max(sM,sm)); Win=max(3,Win)
        y=np.arange(-Win,Win+1); x=np.arange(-Win,Win+1)
        yy,xx=np.meshgrid(y,x); XY=np.stack([xx,yy],-1).reshape(-1,2).astype(np.float32)
        XYr=(XY@R2).reshape(2*Win+1,2*Win+1,2)
        gx=XYr[...,0]/(sM+1e-6); gy=XYr[...,1]/(sm+1e-6)
        patch=np.exp(-0.5*(gx*gx+gy*gy)).astype(np.float32)

        rr0,cc0=r0-Win,c0-Win; rr1,cc1=r0+Win+1,c0+Win+1
        if rr1<=0 or rr0>=Hs or cc1<=0 or cc0>=Ws: continue
        pr0,pc0=max(0,rr0),max(0,cc0); pr1,pc1=min(Hs,rr1),min(Ws,cc1)
        patch=patch[(pr0-rr0):(pr1-rr1+2*Win+1),(pc0-cc0):(pc1-cc1+2*Win+1)]

        opa=float(opacity[i]) if opacity is not None else 1.0
        w = (alpha_scale * opa) * patch
        img_acc[pr0:pr1,pc0:pc1,:] += w[...,None] * rgb[i]
        w_acc[pr0:pr1,pc0:pc1]     += w

    out_big = img_acc / np.clip(w_acc[...,None], 1e-6, None)
    out_big = np.clip(out_big, 0, 1)

    # 色彩后处理
    if gamma!=1.0: out_big=np.clip(out_big**(1.0/gamma),0,1)
    if gain !=1.0: out_big=np.clip(out_big*gain,0,1)
    if sat_boost!=1.0:
        hsv=cv2.cvtColor((out_big*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[...,1]=np.clip(hsv[...,1]*sat_boost,0,255)
        out_big=cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)/255.0

    # 可选：孔洞填补（按权重）
    if hole_fill:
        w_norm = w_acc / (w_acc.max() + 1e-6)
        # 下采样同时保留权重图
        if downsample_to and downsample_to>0:
            L=max(W,H); sc=float(downsample_to)/max(1,L)
            newW=int(round(W*sc)); newH=int(round(H*sc))
            out_small=cv2.resize((out_big*255).astype(np.uint8),(max(1,newW),max(1,newH)),interpolation=cv2.INTER_LANCZOS4)
            w_small=cv2.resize(w_norm,(max(1,newW),max(1,newH)),interpolation=cv2.INTER_AREA)
        else:
            out_small=(cv2.resize((out_big*255).astype(np.uint8),(max(W,1),max(H,1)),interpolation=cv2.INTER_LANCZOS4))
            w_small=cv2.resize(w_norm,(max(W,1),max(H,1)),interpolation=cv2.INTER_AREA)

        hole = (w_small < hole_tau).astype(np.uint8)*255
        if hole.any():
            out_small = cv2.inpaint(out_small, hole, 3, cv2.INPAINT_TELEA)
        if blur_px > 0:
            out_small = cv2.GaussianBlur(out_small,(0,0), blur_px)

        if keep_lcc:
            occ=(np.sum(out_small,2)>0).astype(np.uint8)
            nlab,lab=cv2.connectedComponents(occ,4)
            if nlab>1:
                areas=[int(np.sum(lab==i)) for i in range(1,nlab)]
                k=1+int(np.argmax(areas)); mask=(lab==k)
                out_small[~mask]=0

        if fill>=3 and fill%2==1:
            out_small=cv2.medianBlur(out_small, fill)

        Image.fromarray(out_small).save(out)
        print(f"✅ Ortho saved: {out}  size={out_small.shape[1]}x{out_small.shape[0]}  pix={pix} m/px")
        return

    # 无孔洞流程
    out_big=(out_big*255).astype(np.uint8)
    if downsample_to and downsample_to>0:
        L=max(W,H); sc=float(downsample_to)/max(1,L)
        newW=int(round(W*sc)); newH=int(round(H*sc))
        out_small=cv2.resize(out_big,(max(1,newW),max(1,newH)),interpolation=cv2.INTER_LANCZOS4)
    else:
        out_small=cv2.resize(out_big,(max(W,1),max(H,1)),interpolation=cv2.INTER_LANCZOS4)

    if keep_lcc:
        occ=(np.sum(out_small,2)>0).astype(np.uint8)
        nlab,lab=cv2.connectedComponents(occ,4)
        if nlab>1:
            areas=[int(np.sum(lab==i)) for i in range(1,nlab)]
            k=1+int(np.argmax(areas)); mask=(lab==k)
            out_small[~mask]=0
    if fill>=3 and fill%2==1:
        out_small=cv2.medianBlur(out_small, fill)
    Image.fromarray(out_small).save(out)
    print(f"✅ Ortho saved: {out}  size={out_small.shape[1]}x{out_small.shape[0]}  pix={pix} m/px")

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--pix", type=float, default=0.01)
    ap.add_argument("--plane_thresh", type=float, default=0.04)
    ap.add_argument("--up", choices=["y","z","auto"], default="y")
    ap.add_argument("--upright_min_dot", type=float, default=0.90)
    ap.add_argument("--align_weight", type=float, default=0.10)
    ap.add_argument("--h_lo", type=float, default=-5.0)
    ap.add_argument("--h_hi", type=float, default=60.0)
    ap.add_argument("--sky_sat", type=float, default=0.16)
    ap.add_argument("--sky_val", type=float, default=0.93)
    ap.add_argument("--blue_bias", type=float, default=0.05)
    ap.add_argument("--max_scale", type=float, default=-1.0)
    ap.add_argument("--img_res", type=int, default=4096)
    ap.add_argument("--down", type=int, default=2048)
    ap.add_argument("--fill", type=int, default=3)
    ap.add_argument("--lcc", action="store_true")
    ap.add_argument("--out", default="ortho_map.png")
    # 过滤/裁边
    ap.add_argument("--slab", type=float, default=3.0)
    ap.add_argument("--density_q", type=float, default=0.20)
    ap.add_argument("--roi_factor", type=float, default=3.0)
    ap.add_argument("--roi_dilate_m", type=float, default=8.0)
    ap.add_argument("--no_roi", action="store_true", default=True)
    ap.add_argument("--scale_q", type=float, default=1.0)
    ap.add_argument("--crop_quant", type=float, default=6.0)
    # 渲染
    ap.add_argument("--blend", choices=["norm"], default="norm")
    ap.add_argument("--super", dest="super_scale", type=float, default=4.0)
    ap.add_argument("--fp_gain", type=float, default=2.2)
    ap.add_argument("--alpha_scale", type=float, default=6.0)
    ap.add_argument("--s_min_px", type=float, default=5.0)
    ap.add_argument("--s_max_px", type=float, default=24.0)
    ap.add_argument("--gamma", type=float, default=1.02)
    ap.add_argument("--gain", type=float, default=1.05)
    ap.add_argument("--sat_boost", type=float, default=1.12)
    ap.add_argument("--hole_fill", action="store_true", default=True)
    ap.add_argument("--hole_tau", type=float, default=0.02)
    ap.add_argument("--blur_px", type=float, default=0.0)
    args=ap.parse_args()

    X,rgb_dc,S,opacity,Rquat,fdc,f_rest,sh_degree=read_3dgs_ply(args.ply)
    n,p0,_=fit_plane_constrained(X, iters=3000, thresh=args.plane_thresh,
                                 up_axis=args.up, upright_min_dot=args.upright_min_dot,
                                 align_weight=args.align_weight)
    u,v=build_plane_axes(n)

    render_ortho(X, rgb_dc, S, Rquat, opacity, fdc, f_rest, sh_degree,
                 n, p0, u, v,
                 pix=args.pix, img_res=args.img_res, downsample_to=args.down,
                 h_lo=args.h_lo, h_hi=args.h_hi,
                 sky_sat=args.sky_sat, sky_val=args.sky_val, blue_bias_th=args.blue_bias,
                 max_scale=args.max_scale, keep_lcc=args.lcc, fill=args.fill, out=args.out,
                 slab=args.slab, density_q=args.density_q, roi_factor=args.roi_factor,
                 roi_dilate_m=args.roi_dilate_m, no_roi=args.no_roi,
                 scale_q=args.scale_q, crop_quant=args.crop_quant,
                 blend=args.blend, supersample=args.super_scale,
                 fp_gain=args.fp_gain,
                 alpha_scale=args.alpha_scale, s_min_px=args.s_min_px, s_max_px=args.s_max_px,
                 gamma=args.gamma, gain=args.gain, sat_boost=args.sat_boost,
                 hole_fill=args.hole_fill, hole_tau=args.hole_tau, blur_px=args.blur_px)

if __name__=="__main__":
    main()
