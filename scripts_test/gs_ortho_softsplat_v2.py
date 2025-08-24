import argparse, math, numpy as np, cv2
from plyfile import PlyData

Y00 = 0.2820947918

def read_3dgs_ply(p):
    ply = PlyData.read(p); v = ply["vertex"].data
    X = np.stack([v["x"],v["y"],v["z"]],1).astype(np.float32)
    fdc = np.stack([v["f_dc_0"],v["f_dc_1"],v["f_dc_2"]],1).astype(np.float32)
    rest = [n for n in v.dtype.names if n.startswith("f_rest_")]
    f_rest,deg=None,0
    if rest:
        rest=sorted(rest,key=lambda s:int(s.split("_")[-1]))
        fr=np.stack([v[n] for n in rest],1).astype(np.float32)
        C=fr.shape[1]//3; L=int(round(math.sqrt(C+1)-1)); deg=max(0,L)
        f_rest=fr.reshape(-1,3,C)
    rfs=[f for f in v.dtype.names if f.startswith("rot_")] or [f for f in v.dtype.names if f.startswith("rotation")]
    Rquat=None
    if len(rfs)>=4: Rquat=np.stack([v[rfs[i]] for i in range(4)],1).astype(np.float32)
    opa=np.asarray(v["opacity"],np.float32) if "opacity" in v.dtype.names else None
    return X, fdc, f_rest, deg, Rquat, opa

def eval_sh_basis(deg, d):
    x,y,z=d[...,0],d[...,1],d[...,2]
    out=[np.full_like(x,Y00,np.float32)]
    if deg>=1:
        c1=0.4886025119; out+=[-c1*y,c1*z,-c1*x]
    if deg>=2:
        c2=[1.09254843059,0.315391565253,0.546274215296]
        out+=[c2[0]*x*y,-c2[0]*y*z,c2[1]*(3*z*z-1.0),-c2[0]*x*z,c2[2]*(x*x-y*y)]
    if deg>=3:
        c3=[0.590043589926,2.89061144264,0.457045799464,0.37317633259,1.44530572132]
        out+=[c3[0]*y*(3*x*x-y*y),-c3[1]*x*y*z,c3[2]*y*(5*z*z-1.0),
              -c3[3]*z*(5*z*z-3.0),c3[2]*x*(5*z*z-1.0),-c3[1]*z*(x*x-y*y),
              c3[4]*x*(x*x-3*y*y)]
    return np.stack(out,-1).astype(np.float32)

def quat_to_R(q):
    q=np.asarray(q,np.float32); 
    if q.ndim==1: q=q[None,:]
    if np.mean(np.abs(q[:,3]))<np.mean(np.abs(q[:,0])): q=q[:,[1,2,3,0]]
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

def fit_plane(X, up="y", iters=2000, th=0.04):
    rng=np.random.default_rng(0)
    upv=np.array([0,1,0],np.float32) if up=="y" else np.array([0,0,1],np.float32)
    best=(-1e9,None,None); N=X.shape[0]
    for _ in range(iters):
        i=rng.choice(N,3,False); p1,p2,p3=X[i]
        n=np.cross(p2-p1,p3-p1); L=np.linalg.norm(n)
        if L<1e-8: continue
        n/=L; 
        if np.dot(n,upv)<0: n=-n
        dot=float(np.dot(n,upv)); 
        if dot<0.85: continue
        d=np.abs((X-p1)@n); score=int((d<th).sum())+0.1*N*dot
        if score>best[0]: best=(score,n.astype(np.float32),p1.astype(np.float32))
    if best[1] is None: 
        p1,p2,p3=X[:3]; n=np.cross(p2-p1,p3-p1); n/=np.linalg.norm(n)+1e-12
        best=(0,n.astype(np.float32),p1.astype(np.float32))
    return best[1], best[2]

def build_axes(n):
    ref=np.array([0,0,1],np.float32) if abs(n[2])<0.9 else np.array([1,0,0],np.float32)
    u=np.cross(ref,n); u/=np.linalg.norm(u)+1e-12
    v=np.cross(n,u);   v/=np.linalg.norm(v)+1e-12
    return u.astype(np.float32), v.astype(np.float32)

def density_roi(U,V,cell=0.06,q=0.60,dilate_m=3.0):
    u0,v0=U.min(),V.min()
    ui=((U-u0)/cell).astype(np.int32); vi=((V-v0)/cell).astype(np.int32)
    H,W=vi.max()+1,ui.max()+1
    cnt=np.zeros((H,W),np.int32); np.add.at(cnt,(vi,ui),1)
    if cnt.max()==0: return np.ones_like(U,bool),(u0,v0,u0+W*cell,v0+H*cell)
    th=np.quantile(cnt[cnt>0],q); mask=(cnt>=max(1,th)).astype(np.uint8)
    rad=max(1,int(round(dilate_m/cell))); mask=cv2.dilate(mask,np.ones((2*rad+1,2*rad+1),np.uint8),1)
    nlab,lab=cv2.connectedComponents(mask,4)
    if nlab>1:
        areas=[int((lab==i).sum()) for i in range(1,nlab)]
        k=1+int(np.argmax(areas)); mask=(lab==k).astype(np.uint8)
    keep=mask[vi,ui].astype(bool)
    ys,xs=np.where(mask>0)
    umin=u0+xs.min()*cell; umax=u0+(xs.max()+1)*cell
    vmin=v0+ys.min()*cell; vmax=v0+(ys.max()+1)*cell
    return keep,(umin,vmin,umax,vmax)

def robust_bounds(U,V,q_low=0.20,q_high=0.80):
    umin,umax=np.quantile(U,[q_low,q_high]); vmin,vmax=np.quantile(V,[q_low,q_high])
    return float(umin),float(vmin),float(umax),float(vmax)

def softsplat_ss(X, rgb, n, p0, u, v, pix=0.010, pad=0.003,
                 ss=2, beta=5.0, q_low=0.20, q_high=0.80,
                 inpaint_tau=0.03):
    U=(X-p0)@u; V=(X-p0)@v; H=(X-p0)@n
    keep,(u0,v0,u1,v1)=density_roi(U,V,0.06,0.60,3.0)
    U,V,H,rgb = U[keep],V[keep],H[keep],rgb[keep]

    uu0,vv0,uu1,vv1=robust_bounds(U,V,q_low,q_high)
    u0,v0,u1,v1 = max(u0,uu0),max(v0,vv0),min(u1,uu1),min(v1,vv1)

    pu=(u1-u0)*pad; pv=(v1-v0)*pad
    u0-=pu; u1+=pu; v0-=pv; v1+=pv

    W=int(np.ceil((u1-u0)/pix))+1
    Hpx=int(np.ceil((v1-v0)/pix))+1

    # 超采样网格
    W_hi, H_hi = W*ss, Hpx*ss
    pix_hi = pix/ss

    c = (U-u0)/pix_hi; r = (v1-V)/pix_hi
    eps=1e-6
    c=np.clip(c,0,W_hi-1-eps); r=np.clip(r,0,H_hi-1-eps)
    j=np.floor(c).astype(np.int32); i=np.floor(r).astype(np.int32)
    fx=c-j; fy=r-i
    w00=(1-fx)*(1-fy); w10=fx*(1-fy); w01=(1-fx)*fy; w11=fx*fy

    # 软深度
    hmin,hmax = np.min(H), np.max(H)
    h_norm = (H - hmin) / max(1e-6, (hmax - hmin))
    wdepth = np.exp(beta * h_norm).astype(np.float32)

    acc=np.zeros((H_hi*W_hi,3),np.float32)
    den=np.zeros((H_hi*W_hi,),np.float32)
    flat=lambda ii,jj:(ii*W_hi+jj)

    for wij, jj, ii in [(w00,j,i),(w10,j+1,i),(w01,j,i+1),(w11,j+1,i+1)]:
        jj=np.clip(jj,0,W_hi-1); ii=np.clip(ii,0,H_hi-1)
        f=flat(ii,jj); w=(wij*wdepth).astype(np.float32)
        np.add.at(den,f,w); np.add.at(acc,f,(rgb*w[:,None]).astype(np.float32))

    img_hi = acc.reshape(H_hi,W_hi,3); Wsum_hi = den.reshape(H_hi,W_hi)[:,:,None]
    img_hi = np.divide(img_hi, np.clip(Wsum_hi,1e-6,None))

    # 下采样到目标分辨率（平均）
    img = img_hi.reshape(Hpx, ss, W, ss, 3).mean(3).mean(1)
    wsum = Wsum_hi.reshape(Hpx, ss, W, ss, 1).mean(3).mean(1)
    # 孔洞修复
    hole = (wsum.squeeze(-1) < inpaint_tau).astype(np.uint8)*255
    if hole.any():
        out8 = (np.clip(img,0,1)*255).astype(np.uint8)
        out8 = cv2.inpaint(out8, hole, 3, cv2.INPAINT_TELEA)
        img = out8.astype(np.float32)/255.0
    return img

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--up", choices=["y","z"], default="y")
    ap.add_argument("--pix", type=float, default=0.010)
    ap.add_argument("--pad", type=float, default=0.003)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--h_lo", type=float, default=-5.0)
    ap.add_argument("--h_hi", type=float, default=32.0)
    ap.add_argument("--opacity_min", type=float, default=0.03)
    ap.add_argument("--q_low", type=float, default=0.20)
    ap.add_argument("--q_high", type=float, default=0.80)
    ap.add_argument("--gain", type=float, default=1.06)
    ap.add_argument("--sat_boost", type=float, default=1.12)
    ap.add_argument("--out", default="ortho_soft_v2.png")
    args=ap.parse_args()

    X, fdc, f_rest, L, Rquat, opa = read_3dgs_ply(args.ply)
    # 地面与过滤
    n,p0 = fit_plane(X, up=args.up)
    h=(X-p0)@n; keep=(h>=args.h_lo)&(h<=args.h_hi)
    if opa is not None: keep &= (opa >= args.opacity_min)
    X=X[keep]; fdc=fdc[keep]
    f_rest=f_rest[keep] if f_rest is not None else None
    Rquat=Rquat[keep] if Rquat is not None else None

    rgb = sh_color_topdown(fdc,f_rest,L,Rquat,n)
    u,v = build_axes(n)

    img = softsplat_ss(X, rgb, n, p0, u, v,
                       pix=args.pix, pad=args.pad, ss=args.ss,
                       beta=args.beta, q_low=args.q_low, q_high=args.q_high)

    out = np.clip(img*args.gain, 0, 1)
    if abs(args.sat_boost-1.0)>1e-3:
        hsv=cv2.cvtColor((out*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[...,1]=np.clip(hsv[...,1]*args.sat_boost,0,255)
        out=cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)/255.0

    cv2.imwrite(args.out, cv2.cvtColor((out*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"✅ saved: {args.out}")

if __name__=="__main__":
    main()
