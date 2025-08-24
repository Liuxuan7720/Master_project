import os, argparse
import numpy as np
from plyfile import PlyData
from PIL import Image
import cv2

Y00 = 0.2820947918  # SH DC 系数

# ---------- 读取 3DGS PLY，得到 xyz, rgb(由DC), scale(线性), opacity, rotation(quat或None) ----------
def read_3dgs_ply(ply_path):
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)

    # 颜色：从 SH 的 DC 三通道估计
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
    rgb = 1.0 / (1.0 + np.exp(-(fdc * Y00)))  # sigmoid

    # 尺度：有的分支是 log-scale；这里自动探测并转回线性
    s0, s1, s2 = v["scale_0"], v["scale_1"], v["scale_2"]
    S = np.stack([s0, s1, s2], 1).astype(np.float32)
    if np.mean(S) < 0 or np.max(S) < 1e-3:   # 粗略判定为 log
        S = np.exp(S)

    # 不透明度（可选）
    opacity = None
    if "opacity" in v.dtype.names:
        opacity = np.asarray(v["opacity"], dtype=np.float32)
        opacity = np.clip(opacity, 0.0, 1.0)

    # 旋转（四元数），不同分支字段名略有差异：可能是 rot_0..3 或 rotation_0..3
    rot_fields = [f for f in v.dtype.names if f.startswith("rot_")] or \
                 [f for f in v.dtype.names if f.startswith("rotation")]
    Rquat = None
    if len(rot_fields) >= 4:
        # 取前四个通道作为 quat (x,y,z,w) 或 (w,x,y,z)，下面归一化后统一处理
        arr = np.stack([v[rot_fields[i]] for i in range(4)], 1).astype(np.float32)
        Rquat = arr

    return X, rgb, S, opacity, Rquat

# ---------- RANSAC 平面 ----------
def fit_plane_ransac(X, iters=3000, thresh=0.03, seed=123):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    best = (-1, None, None, None)
    if N < 3:
        n = np.array([0,1,0], np.float32); p0 = X.mean(0).astype(np.float32)
        return n, p0, np.ones((N,), bool)
    for _ in range(iters):
        ids = rng.choice(N, 3, replace=False)
        p1, p2, p3 = X[ids]
        n = np.cross(p2-p1, p3-p1); nn = np.linalg.norm(n)
        if nn < 1e-8: continue
        n = n/nn
        if n[1] < 0: n = -n
        d = np.abs((X - p1) @ n)
        inl = d < thresh
        cnt = int(inl.sum())
        if cnt > best[0]:
            best = (cnt, n, p1, inl)
    return best[1].astype(np.float32), best[2].astype(np.float32), best[3]

def build_plane_axes(n):
    ref = np.array([0,0,1], np.float32) if abs(n[2]) < 0.9 else np.array([1,0,0], np.float32)
    u = np.cross(ref, n); u /= (np.linalg.norm(u)+1e-12)
    v = np.cross(n, u);   v /= (np.linalg.norm(v)+1e-12)
    return u, v

# ---------- 颜色→HSV 辅助 ----------
def rgb_to_sv(rgb):
    R,G,B = rgb[:,0], rgb[:,1], rgb[:,2]
    Cmax = np.maximum(np.maximum(R,G), B)
    Cmin = np.minimum(np.minimum(R,G), B)
    V = Cmax
    S = np.where(Cmax < 1e-6, 0.0, (Cmax - Cmin) / (Cmax + 1e-6))
    blue_bias = B - np.maximum(R,G)
    return S, V, blue_bias

# ---------- 四元数→旋转矩阵（每个点一个） ----------
def quat_to_R(q):
    # 接受 (...,4)，自动归一化；兼容 (w,x,y,z) 或 (x,y,z,w)
    q = np.asarray(q, np.float32)
    if q.ndim == 1: q = q[None, :]
    # 让绝对值最大的分量作为 w（经验处理不同导出顺序）
    # 若最后一列绝对值的平均更大，就认为是 (x,y,z,w)
    if np.mean(np.abs(q[:,3])) < np.mean(np.abs(q[:,0])):
        # 假定是 (w,x,y,z) -> 旋转到 (x,y,z,w)
        q = q[:, [1,2,3,0]]
    # 归一化
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    x,y,z,w = q[:,0], q[:,1], q[:,2], q[:,3]
    R = np.stack([
        1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w),
        2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)
    ], axis=1).reshape(-1,3,3).astype(np.float32)
    return R

# ---------- 椭圆高斯渲染到地面正射 ----------
def render_ortho_elliptical(X, rgb, S, Rquat, n, p0, u, v,
                            pix=0.01, img_res=4096, downsample_to=2048,
                            h_lo=-1.0, h_hi=20.0,
                            sky_sat=0.22, sky_val=0.90, blue_bias_th=0.06,
                            max_scale=0.30, keep_lcc=True, fill=3, out="ortho_map.png"):
    # 高度
    h = (X - p0) @ n

    # 天空/远景剔除 + 尺度阈值
    S1, V1, blue_bias = rgb_to_sv(rgb)
    sky_like = ((V1 >= sky_val) & (S1 <= sky_sat)) | (blue_bias > blue_bias_th)

    # 处理尺度：如果 S 是 (N,) 或 (3N,) 重排；再取主轴最大
    if S.ndim == 1 and (S.size % 3 == 0): S = S.reshape(-1,3)
    if S.ndim == 1: maxaxis = S
    else:           maxaxis = S.max(axis=1)
    too_big = (maxaxis > max_scale)

    keep = (h >= h_lo) & (h <= h_hi) & (~sky_like) & (~too_big)

    X, rgb, S, h = X[keep], rgb[keep], S[keep], h[keep]
    if Rquat is not None: Rm = quat_to_R(Rquat[keep])
    else:                 Rm = None

    # 地面坐标
    U = (X - p0) @ u; Vv = (X - p0) @ v
    umin, umax = np.percentile(U, [2, 98]); vmin, vmax = np.percentile(Vv, [2, 98])
    pad_u = (umax - umin) * 0.05; pad_v = (vmax - vmin) * 0.05
    umin -= pad_u; umax += pad_u; vmin -= pad_v; vmax += pad_v

    W = int(np.ceil((umax - umin) / pix)) + 1
    H = int(np.ceil((vmax - vmin) / pix)) + 1
    cols = np.clip(((U - umin) / pix).astype(np.int32), 0, W-1)
    rows = np.clip(((vmax - Vv) / pix).astype(np.int32), 0, H-1)

    # 按“离地面近”优先（h小的先画，避免高空覆盖地面）
    order = np.argsort(h)
    cols, rows = cols[order], rows[order]
    rgb = rgb[order]
    if S.ndim == 2: S = S[order]
    else:           S = S[order]
    if Rm is not None: Rm = Rm[order]

    # 画布（先超采样）
    scale_up = img_res / max(H, W)
    Hs, Ws = int(H*scale_up), int(W*scale_up)
    hit = np.zeros((Hs, Ws), np.float32)
    img = np.zeros((Hs, Ws, 3), np.float32)

    # 将 (row,col) 放大到超采样坐标
    rows_s = (rows * scale_up).astype(int)
    cols_s = (cols * scale_up).astype(int)

    # 每个高斯的二维协方差：Sigma2d = P * R * diag(s^2) * R^T * P^T （正射）
    # P 的两行分别是 u^T 和 v^T
    P = np.stack([u, v], 0).astype(np.float32)  # (2,3)

    for i in range(len(rows_s)):
        r0, c0 = rows_s[i], cols_s[i]
        # 3D 协方差
        if S.ndim == 2: s = S[i]
        else:           s = np.array([S[i], S[i], S[i]], np.float32)
        if Rm is not None: R3 = Rm[i]
        else:              R3 = np.eye(3, dtype=np.float32)
        C3 = (R3 @ np.diag(s**2) @ R3.T).astype(np.float32)
        C2 = (P @ C3 @ P.T).astype(np.float32)          # 2x2

        # 取 2D 椭圆 3σ 区域作为绘制窗口
        # 用特征值决定核大小
        evals, evecs = np.linalg.eigh(C2 + 1e-8*np.eye(2,dtype=np.float32))
        sig_major = np.sqrt(max(evals[1], 1e-8))
        sig_minor = np.sqrt(max(evals[0], 1e-8))
        # 将 sigma 转成像素
        # 一个“像素”大约对应 pix 长度；在超采样画布中按 scale_up 放大
        s_pix_major = max(sig_major / pix * scale_up, 1.5)
        s_pix_minor = max(sig_minor / pix * scale_up, 1.0)

        R2 = evecs
        Win = int(4 * max(s_pix_major, s_pix_minor))
        if Win < 3: Win = 3
        y = np.arange(-Win, Win+1)
        x = np.arange(-Win, Win+1)
        yy, xx = np.meshgrid(y, x)  # 注意：行对应 y，列对应 x
        XY = np.stack([xx, yy], -1).reshape(-1,2).astype(np.float32)
        XY_rot = (XY @ R2).reshape(2*Win+1, 2*Win+1, 2)
        gx = XY_rot[...,0] / (s_pix_major+1e-6)
        gy = XY_rot[...,1] / (s_pix_minor+1e-6)
        patch = np.exp(-0.5*(gx*gx + gy*gy)).astype(np.float32)

        rr0, cc0 = r0-Win, c0-Win
        rr1, cc1 = r0+Win+1, c0+Win+1
        if rr1<=0 or rr0>=Hs or cc1<=0 or cc0>=Ws: 
            continue
        pr0, pc0 = max(0, rr0), max(0, cc0)
        pr1, pc1 = min(Hs, rr1), min(Ws, cc1)
        patch = patch[(pr0-rr0):(pr1-rr1+2*Win+1), (pc0-cc0):(pc1-cc1+2*Win+1)]

        img[pr0:pr1, pc0:pc1, :] += patch[...,None] * rgb[i]
        hit[pr0:pr1, pc0:pc1] += patch

    # 归一化 + 缩回目标大小
    img = img / np.clip(hit[...,None], 1e-4, None)
    img = np.clip(img, 0, 1)
    out_big = (img*255).astype(np.uint8)
    # out_big: 超采样画布；把它缩到 (H, W) 或者按 downsample_to 的长边限制
    if downsample_to is not None and downsample_to > 0:
        # 按长边等比缩放到 downsample_to
        long_side = max(W, H)
        scale = float(downsample_to) / max(1, long_side)
        newW = max(1, int(round(W * scale)))
        newH = max(1, int(round(H * scale)))
        out_small = cv2.resize(out_big, (newW, newH), interpolation=cv2.INTER_LANCZOS4)
    else:
        # 维持由 pix 决定的 (W, H)
        out_small = cv2.resize(out_big, (max(W,1), max(H,1)), interpolation=cv2.INTER_LANCZOS4)

    # 可选最大连通域与轻度空洞填补
    if keep_lcc:
        occ = (hit > 1e-3).astype(np.uint8)
        # 把占据图缩到 out_small 的分辨率（最近邻，避免伪值）
        Hsmall, Wsmall = out_small.shape[0], out_small.shape[1]
        occ_small = cv2.resize(occ, (Wsmall, Hsmall), interpolation=cv2.INTER_NEAREST)

        num_labels, labels = cv2.connectedComponents(occ_small, connectivity=4)
        if num_labels > 1:
            areas = [int(np.sum(labels == i)) for i in range(1, num_labels)]
            k = 1 + int(np.argmax(areas))
            mask = (labels == k)  # (H, W) 布尔
            # 直接用布尔索引更稳
            out_small[~mask] = 0

    if fill >= 3 and fill % 2 == 1:
        out_small = cv2.medianBlur(out_small, fill)

    Image.fromarray(out_small).save(out)
    print(f"✅ Ortho saved: {out}  size={out_small.shape[1]}x{out_small.shape[0]}  pix={pix} m/px")

# ---------- 可选：从任意相机位姿渲染透视图（满足“指定位置和角度”） ----------
def render_perspective_view(X, rgb, S, Rquat, cam_pos, look_at, up, fov_deg=60, res=2048, out="view.png"):
    # 构建相机坐标系
    cam_pos = np.asarray(cam_pos, np.float32)
    look_at = np.asarray(look_at, np.float32)
    up      = np.asarray(up, np.float32)
    zc = (look_at - cam_pos); zc /= (np.linalg.norm(zc)+1e-12)
    xc = np.cross(zc, up);    xc /= (np.linalg.norm(xc)+1e-12)
    yc = np.cross(xc, zc)
    Rw2c = np.stack([xc, yc, zc], 0)  # world->camera
    t = - (Rw2c @ cam_pos)

    # 透视投影
    Pcam = (Rw2c @ X.T).T + t
    z = Pcam[:,2]
    fx = fy = 0.5*res / np.tan(0.5*np.deg2rad(fov_deg))
    u = fx * (Pcam[:,0] / np.clip(z, 1e-4, None)) + res/2
    v = fy * (Pcam[:,1] / np.clip(z, 1e-4, None)) + res/2

    order = np.argsort(z)[::-1]  # 由远到近画（简单遮挡）
    u = u[order].astype(np.int32); v = v[order].astype(np.int32)
    rgb = rgb[order]

    img = np.zeros((res,res,3), np.float32)
    for i in range(len(u)):
        if 0<=u[i]<res and 0<=v[i]<res:
            img[v[i], u[i]] = rgb[i]
    out_img = (np.clip(img,0,1)*255).astype(np.uint8)
    Image.fromarray(out_img).save(out)
    print(f"✅ Perspective saved: {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--pix", type=float, default=0.01)
    ap.add_argument("--plane_thresh", type=float, default=0.04)
    ap.add_argument("--h_lo", type=float, default=-1.0)
    ap.add_argument("--h_hi", type=float, default=20.0)
    ap.add_argument("--sky_sat", type=float, default=0.22)
    ap.add_argument("--sky_val", type=float, default=0.90)
    ap.add_argument("--blue_bias", type=float, default=0.06)
    ap.add_argument("--max_scale", type=float, default=0.30)
    ap.add_argument("--img_res", type=int, default=4096)
    ap.add_argument("--down", type=int, default=2048)
    ap.add_argument("--fill", type=int, default=3)
    ap.add_argument("--lcc", action="store_true")
    ap.add_argument("--out", default="ortho_map.png")
    args = ap.parse_args()

    X, rgb, S, opacity, Rquat = read_3dgs_ply(args.ply)
    # 地面拟合
    n, p0, inl = fit_plane_ransac(X, iters=3000, thresh=args.plane_thresh)
    u, v = build_plane_axes(n)

    render_ortho_elliptical(
        X, rgb, S, Rquat, n, p0, u, v,
        pix=args.pix, img_res=args.img_res, downsample_to=args.down,
        h_lo=args.h_lo, h_hi=args.h_hi,
        sky_sat=args.sky_sat, sky_val=args.sky_val, blue_bias_th=args.blue_bias,
        max_scale=args.max_scale, keep_lcc=args.lcc, fill=args.fill, out=args.out
    )

if __name__ == "__main__":
    main()
