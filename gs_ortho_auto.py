import argparse, math, numpy as np, cv2
from plyfile import PlyData

Y00 = 0.2820947918

# ---------- Read 3DGS PLY ----------
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
        L    = int(round(math.sqrt(C + 1) - 1)); deg = max(0, L)
        f_rest = fr.reshape(-1, 3, C)
    rfs = [f for f in v.dtype.names if f.startswith("rot_")] \
       or [f for f in v.dtype.names if f.startswith("rotation")]
    Rquat = None
    if len(rfs) >= 4:
        Rquat = np.stack([v[rfs[i]] for i in range(4)], 1).astype(np.float32)
    opa = np.asarray(v["opacity"], np.float32) if "opacity" in v.dtype.names else None
    return X, fdc, f_rest, deg, Rquat, opa

# ---------- SH ----------
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

def quat_to_R(q):
    q=np.asarray(q,np.float32)
    if q.ndim==1: q=q[None,:]
    if np.mean(np.abs(q[:,3])) < np.mean(np.abs(q[:,0])):  
        q=q[:,[1,2,3,0]]
    q=q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-12)
    x,y,z,w=q[:,0],q[:,1],q[:,2],q[:,3]
    R=np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],1).reshape(-1,3,3).astype(np.float32)
    return R

def sh_color_topdown(fdc,f_rest,deg,Rquat,n):
    if deg<=0 or f_rest is None or Rquat is None:
        return np.clip(1/(1+np.exp(-(fdc*Y00))),0,1)
    Rm=quat_to_R(Rquat); v=(-n).astype(np.float32); v/=np.linalg.norm(v)+1e-12
    dirs=(Rm.transpose(0,2,1)@v[None,:].repeat(Rm.shape[0],0)[...,None]).squeeze(-1)
    dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)+1e-12
    B=eval_sh_basis(deg,dirs); feats=np.concatenate([fdc[:,:,None],f_rest],2)
    rgb=1/(1+np.exp(-(feats*B[:,None,:]).sum(2)))
    return np.clip(rgb,0,1)

# ---------- Ground fitting ----------
def fit_plane_once(X, upv, iters=2000, th=0.04, upright_min_dot=0.85, seed=0):
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

def auto_ground(X, iters=2000, th=0.04):
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

# ---------- Main-body ROI ----------
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

# ---------- Rendering (supports downward-only frame expansion) ----------
def softsplat_render(
    X, rgb, n, p0, u, v, opa=None,
    pix=0.010, pad=0.002, ss=3, beta=4.5,
    q_low=0.30, q_high=0.70, shift_u=0.0, shift_v=0.0, frame_scale=1.0
):
    U=(X-p0)@u; V=(X-p0)@v; H=(X-p0)@n
    keep,(u0,v0,u1,v1)=density_roi(U,V,0.06,0.65,3.0)
    U,V,H,rgb = U[keep],V[keep],H[keep],rgb[keep]
    if opa is not None: opa=opa[keep]

    # Adaptive height cropping (remove far background/clouds)
    h_lo,h_hi=np.quantile(H,[0.02,0.90])
    sel=(H>=h_lo)&(H<=h_hi)
    U,V,H,rgb = U[sel],V[sel],H[sel],rgb[sel]
    if opa is not None: opa=opa[sel]

    # Main-body quantile crop
    uu0,vv0,uu1,vv1=robust_bounds(U,V,q_low,q_high)
    u0,v0,u1,v1 = max(u0,uu0),max(v0,vv0),min(u1,uu1),min(v1,vv1)

    # padding
    pu=(u1-u0)*pad; pv=(v1-v0)*pad
    u0-=pu; u1+=pu; v0-=pv; v1+=pv

    # First symmetric scaling, then shift (downward expansion when shift_v>0)
    base_w=(u1-u0); base_h=(v1-v0)
    cu=(u0+u1)/2; cv=(v0+v1)/2
    half_u=base_w*0.5*frame_scale; half_v=base_h*0.5*frame_scale
    du=base_w*shift_u; dv=base_h*shift_v
    u0,u1 = cu-half_u+du, cu+half_u+du
    v0,v1 = cv-half_v+dv, cv+half_v+dv

    # Resolution
    W=int(np.ceil((u1-u0)/pix))+1
    Ht=int(np.ceil((v1-v0)/pix))+1

    # Supersampling grid
    W_hi, H_hi = W*ss, Ht*ss
    pix_hi = pix/ss

    # Continuous pixel coordinates
    c=(U-u0)/pix_hi; r=(v1-V)/pix_hi
    eps=1e-6
    c=np.clip(c,0,W_hi-1-eps); r=np.clip(r,0,H_hi-1-eps)
    j=np.floor(c).astype(np.int32); i=np.floor(r).astype(np.int32)
    fx=c-j; fy=r-i
    w00=(1-fx)*(1-fy); w10=fx*(1-fy); w01=(1-fx)*fy; w11=fx*fy

    # Weights: soft depth + opacity
    hmin,hmax=float(np.min(H)),float(np.max(H))
    h_norm=(H-hmin)/max(1e-6,(hmax-hmin))
    wdepth=np.exp(beta*h_norm).astype(np.float32)
    wopa=np.clip(opa,0.0,1.0).astype(np.float32) if opa is not None else 1.0

    acc=np.zeros((H_hi*W_hi,3),np.float32)
    den=np.zeros((H_hi*W_hi,),np.float32)
    flat=lambda ii,jj:(ii*W_hi+jj)

    for wij, jj, ii in [(w00,j,i),(w10,j+1,i),(w01,j,i+1),(w11,j+1,i+1)]:
        jj=np.clip(jj,0,W_hi-1); ii=np.clip(ii,0,H_hi-1)
        f=flat(ii,jj); w=(wij*wdepth*wopa).astype(np.float32)
        np.add.at(den,f,w); np.add.at(acc,f,(rgb*w[:,None]).astype(np.float32))

    img_hi=acc.reshape(H_hi,W_hi,3)
    Wsum_hi2=den.reshape(H_hi,W_hi).astype(np.float32)

    # Smoothing + normalization
    sigma=0.6*ss
    img_hi=cv2.GaussianBlur(img_hi,(0,0),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REPLICATE)
    Wsum_hi=cv2.GaussianBlur(Wsum_hi2,(0,0),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REPLICATE)[...,None]
    img_hi=np.divide(img_hi, np.clip(Wsum_hi,1e-6,None))

    # Downsample
    img = img_hi.reshape(Ht,ss,W,ss,3).mean(3).mean(1)

    # Small hole inpainting (based on weight threshold)
    wsum = Wsum_hi.reshape(Ht,ss,W,ss,1).mean(3).mean(1).squeeze(-1)
    hole=(wsum<0.02).astype(np.uint8)*255
    out8=(np.clip(img,0,1)*255).astype(np.uint8)
    if hole.any():
        out8=cv2.inpaint(out8,hole,3,cv2.INPAINT_TELEA)

    return out8

# ---------- main ----------
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
    ap.add_argument("--gain", type=float, default=1.06)
    ap.add_argument("--sat_boost", type=float, default=1.12)
    ap.add_argument("--opacity_min", type=float, default=0.03)
    ap.add_argument("--out_img", default="ortho.png")
    args=ap.parse_args()

    X,fdc,f_rest,L,Rquat,opa=read_3dgs_ply(args.ply)
    n,p0=auto_ground(X, iters=2000, th=0.04)

    # Coarse filtering (height/opacity)
    h=(X-p0)@n
    keep=np.ones(len(X),bool)
    if opa is not None: keep &= (opa>=args.opacity_min)
    lo,hi=np.quantile(h,[0.02,0.98])
    keep &= (h>=lo)&(h<=hi)
    X=X[keep]; fdc=fdc[keep]
    f_rest=f_rest[keep] if f_rest is not None else None
    Rquat=Rquat[keep] if Rquat is not None else None
    opa=opa[keep] if opa is not None else None

    rgb=sh_color_topdown(fdc,f_rest,L,Rquat,n)
    u,v=build_axes(n)

    out8=softsplat_render(
        X,rgb,n,p0,u,v,opa=opa,
        pix=args.pix,pad=args.pad,ss=args.ss,beta=args.beta,
        q_low=args.q_low,q_high=args.q_high,
        shift_u=args.shift_u,shift_v=args.shift_v,frame_scale=args.frame_scale
    )

    # Color tweak
    img=out8.astype(np.float32)/255.0
    if abs(args.gain-1.0)>1e-3 or abs(args.sat_boost-1.0)>1e-3:
        hsv=cv2.cvtColor((img*255).astype(np.uint8),cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[...,1]=np.clip(hsv[...,1]*args.sat_boost,0,255)
        hsv[...,2]=np.clip(hsv[...,2]*args.gain,0,255)
        img=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2RGB).astype(np.float32)/255.0
    img8=(np.clip(img,0,1)*255).astype(np.uint8)

    cv2.imwrite(args.out_img, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))
    print("saved:", args.out_img)

if __name__ == "__main__":
    main()
