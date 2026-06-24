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

import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy
from tool.traj_loss import TrajectoryLossManager, RigidityLossManager, TrackDataset

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

# ========== 时序轨迹损失相关配置 ==========
TRAJ_LOSS_CONFIG = {
    'lambda_traj': 0.1,           # 轨迹损失权重
    'lambda_rigid': 0.01,         # 刚性损失权重 
    'lambda_normal': 0.01,        # 法线一致性损失权重 
    'traj_loss_start_iter': 1000, # 从第几次迭代开始计算轨迹损失
    'traj_loss_interval': 10,     # 每隔多少次迭代计算一次轨迹损失
    'traj_files': "/media/DGST_data/trajectory/031/left_eye/all_trajectory3D.json",  # 轨迹文件路径
    'preferred_cam_idx': 9,       # 优先使用的相机索引（0=正脸视角，根据你的数据集调整）
}


def load_trajectory_data(source_path, traj_files=None):
    """
    使用 TrackDataset 加载轨迹数据。
    
    Args:
        source_path: 数据集根目录
        traj_files: 轨迹文件路径列表（可选）
    
    Returns:
        tracks: list of dict, 格式为 TrajectoryLossManager 所需的格式
                [{'id': ..., 'points': ..., 't_start': ..., 'weight': ...}, ...]
    """
    import glob
    
    # 自动查找轨迹文件
    if traj_files is None:
        possible_patterns = [
            os.path.join(source_path, 'trajectories.json'),
            os.path.join(source_path, 'tracks.json'),
            os.path.join(source_path, 'traj', '*.json'),
            os.path.join(source_path, 'trajectory', '*.json'),
        ]
        
        traj_files = []
        for pattern in possible_patterns:
            matched = glob.glob(pattern)
            if matched:
                traj_files.extend(matched)
                break  # 只使用第一个匹配的模式
    
    if not traj_files:
        print(f"[Warning] No trajectory files found in {source_path}. Trajectory loss will be disabled.")
        return None
    
    # 确保是列表
    if isinstance(traj_files, str):
        traj_files = [traj_files]
    
    print(f"Loading trajectory data from: {traj_files}")
    
    # 使用 TrackDataset 加载
    track_dataset = TrackDataset()
    track_dataset.load_json_files(traj_files)
    
    # 转换为 TrajectoryLossManager 需要的格式
    tracks = []
    for track_id, info in track_dataset._tracks.items():
        tracks.append({
            'id': track_id,
            'points': info['positions'],  # [L, 3] 的坐标列表
            't_start': info['start_frame'],
            'weight': 1.0
        })
    
    print(f"Loaded {len(tracks)} trajectories from {len(traj_files)} file(s).")
    return tracks


class TemporalSequentialSampler:
    """
    按时间顺序采样连续帧，用于轨迹损失计算。
    
    FourDGSdataset_window[i] 返回 [cam_t, cam_t+1]
    需要验证返回的帧确实是时间连续的（避免跨相机边界）。
    """
    def __init__(self, train_camera_dataset, window_size=2, preferred_cam_idx=0):
        print(f"[TemporalSampler] Initializing...", flush=True)
        self.window_size = window_size
        self.dataset = train_camera_dataset
        
        # 获取第一个窗口来了解时间结构
        first_window = train_camera_dataset[0]  # [cam_t, cam_t+1]
        t0 = first_window[0].time
        t1 = first_window[1].time
        self.expected_time_step = t1 - t0  # 相邻帧的时间间隔
        self.tolerance = self.expected_time_step * 0.5
        
        # 计算单个相机的帧数（基于 time 的归一化范围）
        # time = frame_idx / total_frames, 所以 total_frames ≈ 1 / time_step
        self.num_frames = int(round(1.0 / self.expected_time_step)) if self.expected_time_step > 0 else 300
        print(f"[TemporalSampler] Time step: {self.expected_time_step:.6f}, estimated {self.num_frames} frames per camera.", flush=True)
    
    def _is_temporally_valid(self, window):
        """检查窗口内的帧是否时间连续。"""
        t_curr = window[0].time
        t_next = window[1].time
        time_diff = t_next - t_curr
        return abs(time_diff - self.expected_time_step) <= self.tolerance
    
    def get_temporal_batch(self, t_idx):
        """获取时间索引 t_idx 开始的连续 2 帧。"""
        t_idx = min(t_idx, len(self.dataset) - 1)
        window = self.dataset[t_idx]  # [cam_t, cam_t+1]
        
        if not self._is_temporally_valid(window):
            return None, t_idx
        
        # 计算真正的帧索引
        actual_frame_idx = int(window[0].time * 300)
        return window, actual_frame_idx
    
    def __len__(self):
        return len(self.dataset)
    
    def get_random_temporal_batch(self):
        """随机选择一个有效的时间窗口。"""
        for _ in range(10):
            t_idx = random.randint(0, len(self) - 1)
            batch, frame_idx = self.get_temporal_batch(t_idx)
            if batch is not None:
                return batch, frame_idx
        return self.get_temporal_batch(0)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def scene_reconstruction(
    dataset,
    opt,
    hyper,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    gaussians,
    scene,
    stage,
    tb_writer,
    train_iter,
    timer,
):
    first_iter = 0

    gaussians.training_setup(opt)
    # 利用 checkpoint 恢复模型
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    # 根据数据集是否使用白色背景，设置背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # 创建CUDA事件以测量迭代时间
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    # 初始化用于存储视点的列表
    viewpoint_stack = None
    # 初始化用于记录损失和PSNR的指数移动平均值（EMA）
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    # 创建一个进度条
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    # ========== 初始化轨迹损失管理器 ==========
    traj_manager = None
    rigid_manager = None
    temporal_sampler = None
    ema_traj_loss = 0.0
    ema_rigid_loss = 0.0
    
    if stage == "fine":
        # 加载轨迹数据
        tracks = load_trajectory_data(
            dataset.source_path, 
            traj_files=TRAJ_LOSS_CONFIG.get('traj_files')
        )
        if tracks is not None:
            traj_manager = TrajectoryLossManager(tracks, device='cuda')
            print("[Trajectory Loss] TrajectoryLossManager initialized.", flush=True)
            rigid_manager = RigidityLossManager(sigma=0.01, k_neighbors=10)
            print("[Trajectory Loss] RigidityLossManager initialized.", flush=True)
            # 使用配置的优先相机索引（正脸视角）
            preferred_cam = TRAJ_LOSS_CONFIG.get('preferred_cam_idx', 0)
            temporal_sampler = TemporalSequentialSampler(
                train_cams, 
                window_size=2, 
                preferred_cam_idx=preferred_cam
            )
            print(f"[Trajectory Loss] Enabled with {len(tracks)} trajectories.", flush=True)
        else:
            print(f"[Trajectory Loss] Disabled (no trajectory data found).", flush=True)

    if not viewpoint_stack and not opt.dataloader: # 粗阶段没进来，精细阶段不知道
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)

    batch_size = opt.batch_size
    print("data loading done")
    # 使用数据加载器
    if opt.dataloader:
        # 获取训练相机
        if stage == "coarse":
            viewpoint_stack = scene.getTrainCameras_T0()
        else:
            viewpoint_stack = scene.getTrainCameras()
 
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=16,
                collate_fn=list,
            )
            random_loader = False
        else:
            # 粗阶段进了这里
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack,
                batch_size=batch_size ,
                shuffle=True,
                num_workers=16,
                collate_fn=list,
            )
            random_loader = True
        # 将数据加载器转换为迭代器
        loader = iter(viewpoint_stack_loader)


 
    # for i, batch in enumerate(viewpoint_stack):
    #     print(batch[0].time)
    #     print(batch[1].time)
    #     print(batch[2].time)  
    #     print(len(batch))   
    #     assert 0

    # dynerf, zerostamp_init
    # breakpoint()
    # 用于粗阶段的初始化 ，只有dynerf需要
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack, 0)

        viewpoint_stack = temp_list.copy()

    else:
        load_in_memory = False
        #
    count = 0
    # 迭代训练过程
    for iteration in range(first_iter, final_iter + 1):
        # 如果网络连接为空
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                ( custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer,) = network_gui.receive()
                if custom_cam != None:
                    count += 1
                    # 计算视点索引
                    viewpoint_index = (count) % len(video_cams)
                    if (count // (len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    # print(viewpoint_index)
                    # 获取相应视点
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    # print(custom_cam.time, viewpoint_index, count)
                    # 渲染网络图像
                    net_image = render(
                        custom_cam,
                        gaussians,
                        pipe,
                        background,
                        scaling_modifer,
                        stage=stage,
                        cam_type=scene.dataset_type,
                    )["render"]
                    # 将图像转换为字节
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                # 发生图像字节和数据集路径
                network_gui.send(net_image_bytes, dataset.source_path)
                # 如果需要训练且迭代次数未超过设定值或不需要保持连接，则跳出循环
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None
        # 记录迭代开始时间
        iter_start.record()
        # 更新高斯模型的学习率
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 每1000次迭代增加一次SH的级别，直到最大度数
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera

        # dynerf's branch
        if opt.dataloader and not load_in_memory:
            try:
                # 从数据加载器中获取视点相机
                if stage == 'fine':
                    viewpoint_cams = next(loader)[0]
                    # print(len(viewpoint_cams))
                else:
                    viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(
                        viewpoint_stack,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=32,
                        collate_fn=list,
                    )
                    random_loader = True
                loader = iter(viewpoint_stack_loader)
        else:
            if stage == "fine":
                print("asdasdasdasdas dasd阿萨大大萨达萨达")
            idx = 0
            viewpoint_cams = []
            while idx < batch_size:

                viewpoint_cam = viewpoint_stack.pop(
                    randint(0, len(viewpoint_stack) - 1)
                )
                if not viewpoint_stack:
                    viewpoint_stack = temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx += 1
            if len(viewpoint_cams) == 0:
                continue
        # breakpoint()
        # Render 渲染
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        rot_list = []

        all_mean_3D_deform = []

        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                stage=stage,
                cam_type=scene.dataset_type,
            )
            image, p3ds, viewspace_point_tensor, visibility_filter, radii, rot = (
                render_pkg["render"],
                render_pkg["means3D"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["rotations"],
            )
            images.append(image.unsqueeze(0))
            if scene.dataset_type != "PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image = viewpoint_cam["image"].cuda()

            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            rot_list.append(rot)

            all_mean_3D_deform.append(p3ds[None,:,:])
        # print("rot",len(rot_list))
        # print("rot_list.shpe", rot_list[0].shape)
        # print("***********", gt_images[0].shape)

        all_mean_3D_deform = torch.cat(all_mean_3D_deform, dim=0)
        # assert 0
        # 计算各个视点的最大半径
        radii = torch.cat(radii_list, 0).max(dim=0).values
        # 计算可见性过滤器
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        # 合并图像张量
        image_tensor = torch.cat(images, 0)
        gt_image_tensor = torch.cat(gt_images, 0)
        # Loss 计算损失
        # breakpoint()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
        

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm
        
        loss = Ll1

        n_cams = len(viewpoint_cams)
        l_momentum = None
        traj_loss_val = torch.tensor(0.0, device='cuda')
        rigid_loss_val = torch.tensor(0.0, device='cuda')

        # if n_cams>=3 :
        #     ## MOMENTUM LOSS
        #     l_momentum = all_mean_3D_deform[2,:,:] - 2*all_mean_3D_deform[1,:,:] + all_mean_3D_deform[0,:,:]  
        #     l_momentum = 10000*torch.linalg.norm(l_momentum, dim=-1, ord=1).mean() # mean l1 norm
        
        #     loss+= l_momentum

        # ========== 计算时序轨迹损失 ==========
        if (stage == "fine" and traj_manager is not None and 
            iteration >= TRAJ_LOSS_CONFIG['traj_loss_start_iter'] and
            iteration % TRAJ_LOSS_CONFIG['traj_loss_interval'] == 0):
            
            # 获取连续两帧的相机（按时间顺序，已验证时间连续性）
            temporal_batch, t_start_idx = temporal_sampler.get_random_temporal_batch()
            
            # temporal_batch 可能为 None（如果跨相机边界）
            if temporal_batch is not None and len(temporal_batch) >= 2:
                # 渲染连续两帧获取3D点
                p3ds_list = []
                for t_cam in temporal_batch:
                    t_render_pkg = render(
                        t_cam,
                        gaussians,
                        pipe,
                        background,
                        stage=stage,
                        cam_type=scene.dataset_type,
                    )
                    p3ds_list.append(t_render_pkg["means3D"])
                
                # 堆叠为 [2, N, 3]
                mus_pred_frames = torch.stack(p3ds_list, dim=0)
                
                # 计算轨迹损失
                traj_loss_val = traj_manager.compute_loss(
                    mus_pred_frames,
                    t_start_frame=t_start_idx,
                    total_seq_len=temporal_sampler.num_frames
                )
                
                # 计算刚性损失
                rigid_loss_val = rigid_manager.compute_loss(
                    mus_pred_frames,
                    t_start_frame=t_start_idx,
                    traj_manager=traj_manager
                )
                
                # 添加到总损失
                loss += TRAJ_LOSS_CONFIG['lambda_traj'] * traj_loss_val
                loss += TRAJ_LOSS_CONFIG['lambda_rigid'] * rigid_loss_val
                
                # 更新EMA用于日志
                ema_traj_loss = 0.4 * traj_loss_val.item() + 0.6 * ema_traj_loss
                ema_rigid_loss = 0.4 * rigid_loss_val.item() + 0.6 * ema_rigid_loss

        # 如果阶段是"fine" 且时间平滑权重不为0
        if stage == "fine" and hyper.time_smoothness_weight != 0: #0.01
            # tv_loss = 0
            # 计算时间平滑正则化损失
            tv_loss = gaussians.compute_regulation(
                hyper.time_smoothness_weight,
                hyper.l1_time_planes,
                hyper.plane_tv_weight,
            )
            # 将时间平滑损失添加到总损失中
            loss += tv_loss
        # 如果DSSIM 损失权重不为0
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor, gt_image_tensor)
            # 将DSSIM 损失添加到总损失中

            loss += opt.lambda_dssim * (1.0 - ssim_loss)
        if opt.lambda_lpips !=0:
            lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
            loss += opt.lambda_lpips * lpipsloss


        # wandb.log({"loss": loss.item(), "psnr": psnr_})
        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        # 初始化视点点张量的梯度为零
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        # 累加各个视点点张量的梯度
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = (
                viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            )

        # 记录迭代结束时间
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                postfix_dict = {
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "psnr": f"{psnr_:.{2}f}",
                    "point": f"{total_point}",
                }
                if l_momentum is not None:
                    postfix_dict["l_mom"] = f"{l_momentum:.{4}f}"
                if traj_manager is not None and iteration >= TRAJ_LOSS_CONFIG['traj_loss_start_iter']:
                    postfix_dict["traj"] = f"{ema_traj_loss:.{6}f}"
                    postfix_dict["rigid"] = f"{ema_rigid_loss:.{6}f}"
                progress_bar.set_postfix(postfix_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                [pipe, background],
                stage,
                scene.dataset_type,
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (
                    (iteration < 1000 and iteration % 10 == 9)
                    or (iteration < 3000 and iteration % 50 == 49)
                    or (iteration < 60000 and iteration % 100 == 99)
                ):
                    # breakpoint()
                    render_training_image(
                        scene,
                        gaussians,
                        [test_cams[iteration % len(test_cams)]],
                        render,
                        pipe,
                        background,
                        stage + "test",
                        iteration,
                        timer.get_elapsed_time(),
                        scene.dataset_type,
                    )
                    render_training_image(
                        scene,
                        gaussians,
                        [train_cams[iteration % len(train_cams)]],
                        render,
                        pipe,
                        background,
                        stage + "train",
                        iteration,
                        timer.get_elapsed_time(),
                        scene.dataset_type,
                    )
                    # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification 密集化
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning 跟踪用于剪枝的图像空间中最大半径
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor_grad, visibility_filter
                )

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration * (
                        opt.opacity_threshold_fine_init
                        - opt.opacity_threshold_fine_after
                    ) / (opt.densify_until_iter)
                    densify_threshold = (
                        opt.densify_grad_threshold_fine_init
                        - iteration
                        * (
                            opt.densify_grad_threshold_fine_init
                            - opt.densify_grad_threshold_after
                        )
                        / (opt.densify_until_iter)
                    )
                # 如果达到密集化条件且点数小于 360000，则执行密集化操作
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                    and gaussians.get_xyz.shape[0] < 360000
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    if stage == "coarse":
                        gaussians.densify(
                            densify_threshold,
                            opacity_threshold,
                            scene.cameras_extent,
                            size_threshold,
                            5,
                            5,
                            scene.model_path,
                            iteration,
                            stage,
                        )
                # 如果达到剪枝条件且点数大于 200000，则执行剪枝操作
                if (
                    iteration > opt.pruning_from_iter
                    and iteration % opt.pruning_interval == 0
                    and gaussians.get_xyz.shape[0] > 200000
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    if stage == "coarse":
                        gaussians.prune(
                            densify_threshold,
                            opacity_threshold,
                            scene.cameras_extent,
                            size_threshold,
                        )

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if (
                    iteration % opt.densification_interval == 0
                    and gaussians.get_xyz.shape[0] < 360000
                    and opt.add_point
                ):

                    gaussians.grow(5, 5, scene.model_path, iteration, stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            # 如果当前迭代次数在检查点保存列表中，则保存检查点
            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path
                    + "/chkpnt"
                    + f"_{stage}_"
                    + str(iteration)
                    + ".pth",
                )



def training(
    dataset,
    hyper,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    expname,
):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()

    scene_reconstruction(
        dataset,
        opt,
        hyper,
        pipe,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        debug_from,
        gaussians,
        scene,
        "coarse",
        tb_writer,
        opt.coarse_iterations,
        timer,
    )
    scene_reconstruction(
        dataset,
        opt,
        hyper,
        pipe,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        debug_from,
        gaussians,
        scene,
        "fine",
        tb_writer,
        opt.iterations,
        timer,
    )

def prepare_output_and_logger(expname):
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    stage,
    dataset_type,
):
    if tb_writer:
        tb_writer.add_scalar(
            f"{stage}/train_loss_patches/l1_loss", Ll1.item(), iteration
        )
        tb_writer.add_scalar(
            f"{stage}/train_loss_patchestotal_loss", loss.item(), iteration
        )
        tb_writer.add_scalar(f"{stage}/iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        #
        validation_configs = (
            {
                "name": "test",
                "cameras": [
                    scene.getTestCameras()[idx % len(scene.getTestCameras())]
                    for idx in range(10, 5000, 299)
                ],
            },
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras_T0()[idx % len(scene.getTrainCameras_T0())]
                    for idx in range(10, 5000, 299)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(
                            viewpoint,
                            scene.gaussians,
                            stage=stage,
                            cam_type=dataset_type,
                            *renderArgs,
                        )["render"],
                        0.0,
                        1.0,
                    )
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(
                            viewpoint.original_image.to("cuda"), 0.0, 1.0
                        )
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(
                                stage
                                + "/"
                                + config["name"]
                                + "_view_{}/render".format(viewpoint.image_name),
                                image[None],
                                global_step=iteration,
                            )
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(
                                    stage
                                    + "/"
                                    + config["name"]
                                    + "_view_{}/ground_truth".format(
                                        viewpoint.image_name
                                    ),
                                    gt_image[None],
                                    global_step=iteration,
                                )
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask

                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(
                        stage + "/" + config["name"] + "/loss_viewpoint - l1_loss",
                        l1_test,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        stage + "/" + config["name"] + "/loss_viewpoint - psnr",
                        psnr_test,
                        iteration,
                    )

        if tb_writer:
            tb_writer.add_histogram(
                f"{stage}/scene/opacity_histogram",
                scene.gaussians.get_opacity,
                iteration,
            )

            tb_writer.add_scalar(
                f"{stage}/total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
            tb_writer.add_scalar(
                f"{stage}/deformation_rate",
                scene.gaussians._deformation_table.sum()
                / scene.gaussians.get_xyz.shape[0],
                iteration,
            )
            tb_writer.add_histogram(
                f"{stage}/scene/motion_histogram",
                scene.gaussians._deformation_accum.mean(dim=-1) / 100,
                iteration,
                max_bins=500,
            )

        torch.cuda.empty_cache()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    # 清空CUDA上下文中的缓存
    # torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[3000, 7000, 14000]
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[14000, 20000, 30_000, 45000, 60000],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    args.model_path = os.path.join("/media/DGST_data/", args.expname)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)


    training(
        lp.extract(args),
        hp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.expname,
    )

    # All done
    print("\nTraining complete.")
    # wandb.finish()
