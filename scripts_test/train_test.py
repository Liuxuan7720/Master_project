#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def _fit_ground_plane_ransac(X, iters=2000, thresh=0.02, seed=123):
    """
    用最简单的 RANSAC 拟合地面平面。
    X: (N,3) numpy
    返回: n(法线,单位向量), p0(平面上一点), inlier_mask(bool)
    """
    if X.shape[0] < 3:
        # 点太少，返回水平面
        return np.array([0,1,0], dtype=np.float32), X.mean(0), np.ones((X.shape[0],), dtype=bool)
    rng = np.random.default_rng(seed)
    best_inliers = None
    best_count = -1
    N = X.shape[0]
    for _ in range(iters):
        ids = rng.choice(N, 3, replace=False)
        p1, p2, p3 = X[ids]
        n = np.cross(p2 - p1, p3 - p1)
        nn = np.linalg.norm(n)
        if nn < 1e-8: 
            continue
        n = n / nn
        # 统一法线朝向“上”（让 y 分量为正更稳定；若你的数据 Z 朝上，把索引改为 z）
        if n[1] < 0: 
            n = -n
        d = np.abs((X - p1) @ n)
        inliers = d < thresh
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
            best_n = n
            best_p0 = p1
    if best_inliers is None:
        best_inliers = np.ones((N,), dtype=bool)
        best_n = np.array([0,1,0], dtype=np.float32)
        best_p0 = X.mean(0)
    return best_n.astype(np.float32), best_p0.astype(np.float32), best_inliers

def _dc_to_rgb_from_features(dc_feats):
    """
    仅用 SH 的 DC 三通道估计颜色：rgb = sigmoid(0.2820947918 * dc)
    dc_feats: (N,3) torch.Tensor on CUDA/CPU
    """
    import torch
    Y00 = 0.2820947918
    return torch.sigmoid(dc_feats * Y00)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # === Plane-aware pruning (ground/sky/scale) ===
            if iteration >= opt.plane_prune_start and (iteration - opt.plane_prune_start) % opt.plane_prune_every == 0:
                with torch.no_grad():
                    # 取点坐标到 CPU 
                    X = scene.gaussians.get_xyz.detach().cpu().numpy()
                    # 拟合地面
                    n, p0, inlier_mask = _fit_ground_plane_ransac(
                        X, iters=2000, thresh=opt.plane_inlier_thresh
                    )
                    # 计算相对高度 h = (x - p0)·n
                    h = ((X - p0) @ n).astype(np.float32)
                    keep_h = (h >= opt.keep_height_lo) & (h <= opt.keep_height_hi)

                    # 取 DC 颜色，估计 HSV（先在 torch 上算 RGB，再到 CPU）
                    try:
                        dc = scene.gaussians.get_features[:, :3]  # (N, 3)
                        rgb = _dc_to_rgb_from_features(dc).clamp(0,1)
                        rgb_np = rgb.detach().cpu().numpy()
                        # RGB -> HSV 的近似（不引入额外库，手写简版）
                        R, G, B = rgb_np[:,0], rgb_np[:,1], rgb_np[:,2]
                        Cmax = np.maximum(np.maximum(R,G), B)
                        Cmin = np.minimum(np.minimum(R,G), B)
                        V = Cmax
                        S = np.where(Cmax < 1e-6, 0.0, (Cmax - Cmin) / (Cmax + 1e-6))
                        # 简单的“天空/云”规则：高亮 + 低饱和 或 偏蓝
                        sky_like = ((V >= opt.sky_val_thresh) & (S <= opt.sky_sat_thresh)) | ((B - np.maximum(R,G)) > opt.blue_bias)
                    except Exception:
                        # 如果你的分支特征名不同/取不到DC颜色，就只用几何规则
                        sky_like = np.zeros((X.shape[0],), dtype=bool)

                    # 过大尺度的点（远景/伪影）
                    # ----- 尺度阈值：统一成 (N,) 的“最大主轴长度” -----
                    try:
                        scales_t = scene.gaussians.get_scaling  # 可能是 torch.Tensor 或其它对象
                        s = scales_t.detach().cpu().numpy() if hasattr(scales_t, "detach") else np.asarray(scales_t)

                        # 某些实现把 scale 存在 log 空间：数值很小或为负，阈值前转回线性
                        if np.ndim(s) >= 1 and (np.mean(s) < 0.0 or np.max(s) < 1e-3):
                            s_lin = np.exp(s)
                        else:
                            s_lin = s

                        # 关键：如果是一维并且长度是 3 的倍数，按 (N,3) 重排再取每行最大轴
                        if s_lin.ndim == 1 and (s_lin.size % 3 == 0):
                            s_lin = s_lin.reshape(-1, 3)
                            maxaxis = s_lin.max(axis=1)            # (N,)
                        elif s_lin.ndim == 2 and s_lin.shape[1] == 3:
                            maxaxis = s_lin.max(axis=1)            # (N,)
                        elif s_lin.ndim == 1:
                            maxaxis = s_lin                         # (N,)
                        else:
                            s_lin = np.reshape(s_lin, (-1, 3))
                            maxaxis = s_lin.max(axis=1)

                        too_big = (maxaxis > opt.max_scale_thresh)  # (N,)
                    except Exception:
                        too_big = np.zeros((X.shape[0],), dtype=bool)

                    # ----- 统一形状，确保 (N,) -----
                    # ----- 统一形状，确保所有掩码长度都为 N (= X.shape[0]) -----
                    N = X.shape[0]

                    def _to_len_N(a, N):
                        a = np.asarray(a).reshape(-1)
                        # 如果长度是 3N（每个高斯3个主轴），重排后对三轴取最大
                        if a.size == 3 * N:
                            a = a.reshape(N, 3).max(axis=1)
                        # 长度> N（极少见）：截断；长度< N：用 False 填充
                        elif a.size > N:
                            a = a[:N]
                        elif a.size < N:
                            a = np.pad(a, (0, N - a.size), constant_values=False)
                        return a

                    keep_h     = _to_len_N(keep_h, N).astype(bool)
                    sky_like   = _to_len_N(sky_like, N).astype(bool)
                    too_big    = _to_len_N(too_big, N).astype(bool)
                    inlier_mask= _to_len_N(inlier_mask, N).astype(bool)

                    # ----- 最终保留掩码（只保留主体）-----
                    keep_mask = keep_h & (~sky_like) & (~too_big)
                    keep_mask = keep_mask | inlier_mask  # RANSAC 地面内点兜底保留

                    # 实际剔除
                    removed = int((~keep_mask).sum())
                    if removed > 0:
                        device = scene.gaussians.get_xyz.device
                        keep_mask_t = torch.from_numpy(keep_mask).to(device=device, dtype=torch.bool)

                        # 保证 tmp_radii 存在并且长度正确（有的分支只在 densify 流程里才创建它）
                        if not hasattr(scene.gaussians, "tmp_radii") or scene.gaussians.tmp_radii is None:
                            scene.gaussians.tmp_radii = torch.zeros(keep_mask_t.shape[0], device=device)
                        elif scene.gaussians.tmp_radii.shape[0] != keep_mask_t.shape[0]:
                            # 长度不一致就重新建一个（简单稳妥）
                            scene.gaussians.tmp_radii = torch.zeros(keep_mask_t.shape[0], device=device)

                        # 该分支的 prune_points 期望传入的是“保留掩码(valid_points_mask)”
                        scene.gaussians.prune_points(keep_mask_t)

                        print(f"\n[ITER {iteration}] Plane-aware pruning removed {removed} gaussians; remain {keep_mask.sum()}.")


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--plane_prune_start", type=int, default=7000, help="从该迭代开始做平面/天空剔除")
    parser.add_argument("--plane_prune_every", type=int, default=1000, help="每多少迭代做一次剔除")
    parser.add_argument("--plane_inlier_thresh", type=float, default=0.03, help="RANSAC 内点阈值(米)")
    parser.add_argument("--keep_height_lo", type=float, default=-0.5, help="相对地面高度下界(米)")
    parser.add_argument("--keep_height_hi", type=float, default=6.0, help="相对地面高度上界(米)")
    parser.add_argument("--sky_sat_thresh", type=float, default=0.25, help="低饱和度认为可能是云/天空")
    parser.add_argument("--sky_val_thresh", type=float, default=0.80, help="高亮度认为可能是云/天空")
    parser.add_argument("--blue_bias", type=float, default=0.05, help="B通道较R/G高出该值也视作蓝天空")
    parser.add_argument("--max_scale_thresh", type=float, default=0.25, help="过大尺度(米)的高斯直接剔除(远景膨胀/伪影)")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # 先 extract
    model = lp.extract(args)
    opt   = op.extract(args)
    pipe  = pp.extract(args)

    # 把新增参数挂到 opt 上
    for k in [
        "plane_prune_start","plane_prune_every","plane_inlier_thresh",
        "keep_height_lo","keep_height_hi","sky_sat_thresh",
        "sky_val_thresh","blue_bias","max_scale_thresh"
    ]:
        setattr(opt, k, getattr(args, k))

    # 再调用 training
    training(model, opt, pipe,
             args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
