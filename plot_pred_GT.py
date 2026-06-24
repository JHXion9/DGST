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
import imageio
import open3d as o3d
import numpy as np
import torch
from scene import Scene
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
            read_extrinsics_binary, read_intrinsics_binary
from tool.utils import multithread_write, get_k_w2c, smooth_tracks
import json
from scipy.spatial import cKDTree
from tool.eval import calculate_3d_epe, calculate_accuracy_within_thresholds, calculate_survival_rate
from tool.vis import create_3d_animation
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def plot_gt_pred_3d_trajectories_mp4(
    args,
    measure_PC,
    save_path,
    axis_range=1.0,       # 刻度间隔
    trail_len=20,
    fps=20,
    title="GT vs Pred 3D Trajectories",
    cover_0_10=False,     # True 时至少覆盖 [0,10]
    delta_x=0.0,          # 绕 x 轴旋转角度（度）
    delta_y=0.0,          # 绕 y 轴旋转角度（度）
    delta_z=0.0,          # 绕 z 轴旋转角度（度）
    rotate_target="both"  # "pred" | "gt" | "both" | "none"
):
    """
    生成 GT / Pred 三维轨迹动画（mp4）
    输入轨迹形状:
    - GT: (N, T, 3)
    - pred: (N, T, 3)
    每一帧仅显示最近 trail_len 帧轨迹。
    """

    def build_rotation_matrix(dx_deg, dy_deg, dz_deg):
        dx = np.deg2rad(dx_deg)
        dy = np.deg2rad(dy_deg)
        dz = np.deg2rad(dz_deg)

        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(dx), -np.sin(dx)],
            [0.0, np.sin(dx),  np.cos(dx)],
        ], dtype=np.float64)

        Ry = np.array([
            [ np.cos(dy), 0.0, np.sin(dy)],
            [0.0,         1.0, 0.0],
            [-np.sin(dy), 0.0, np.cos(dy)],
        ], dtype=np.float64)

        Rz = np.array([
            [np.cos(dz), -np.sin(dz), 0.0],
            [np.sin(dz),  np.cos(dz), 0.0],
            [0.0,         0.0,        1.0],
        ], dtype=np.float64)

        # 旋转顺序: 先 x，再 y，再 z
        # 列向量约定下等价于 R = Rz @ Ry @ Rx
        return Rz @ Ry @ Rx

    def rotate_motion_only(traj, R):
        """
        traj: (N, T, 3)
        以每个点第0帧为锚点，仅旋转位移向量
        """
        anchor = traj[:, :1, :]               # (N,1,3)
        disp = traj - anchor                  # (N,T,3)
        disp_rot = disp @ R.T                 # (N,T,3)
        return anchor + disp_rot

    GT_dir = f"/media/DGST_data/trajectory/{args.ID}"
    with open(os.path.join(GT_dir, "_xyz.json"), "r") as file:
        GT = json.load(file)

    GT = np.array(GT)                    # (N, T, 3)
    measure_PC = np.array(measure_PC)    # (T, M, 3)

    # 用 GT 第一帧与预测第一帧做最近邻对应
    tree = cKDTree(measure_PC[0])
    _, indices = tree.query(GT[:, 0, :])

    pred = measure_PC[:, indices]        # (T, N, 3)
    pred = pred.transpose(1, 0, 2)       # -> (N, T, 3)

    # 按要求放大 100 倍
    GT = np.asarray(GT, dtype=np.float32) * 100.0
    pred = np.asarray(pred, dtype=np.float32) * 100.0

    if GT.shape != pred.shape:
        raise ValueError(f"GT 和 pred 形状不一致: {GT.shape} vs {pred.shape}")
    if GT.ndim != 3 or GT.shape[2] != 3:
        raise ValueError(f"输入形状必须为 (N, T, 3)，当前是 {GT.shape}")

    # 旋转运动方向
    R = build_rotation_matrix(delta_x, delta_y, delta_z)
    if rotate_target == "pred":
        pred = rotate_motion_only(pred, R)
    elif rotate_target == "gt":
        GT = rotate_motion_only(GT, R)
    elif rotate_target == "both":
        GT = rotate_motion_only(GT, R)
        pred = rotate_motion_only(pred, R)
    elif rotate_target == "none":
        pass
    else:
        raise ValueError("rotate_target 必须是 'pred' | 'gt' | 'both' | 'none'")

    N, T, _ = GT.shape
    tick_step = float(axis_range) if axis_range > 0 else 1.0

    # 根据全部点计算边界
    all_pts = np.concatenate([GT.reshape(-1, 3), pred.reshape(-1, 3)], axis=0)
    mins = all_pts.min(axis=0).astype(np.float64)
    maxs = all_pts.max(axis=0).astype(np.float64)

    # 每个轴最小显示跨度至少为 1
    for d in range(3):
        if maxs[d] - mins[d] < 1.0:
            c = 0.5 * (maxs[d] + mins[d])
            mins[d] = c - 0.5
            maxs[d] = c + 0.5

    # 可选：至少覆盖 0~10
    if cover_0_10:
        mins = np.minimum(mins, np.array([0.0, 0.0, 0.0]))
        maxs = np.maximum(maxs, np.array([10.0, 10.0, 10.0]))

    # 对齐到刻度网格
    mins = np.floor(mins / tick_step) * tick_step
    maxs = np.ceil(maxs / tick_step) * tick_step

    xlim = (mins[0], maxs[0])
    ylim = (mins[1], maxs[1])
    zlim = (mins[2], maxs[2])

    xticks = np.arange(xlim[0], xlim[1] + 1e-9, tick_step)
    yticks = np.arange(ylim[0], ylim[1] + 1e-9, tick_step)
    zticks = np.arange(zlim[0], zlim[1] + 1e-9, tick_step)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    with imageio.get_writer(save_path, fps=fps, codec="libx264") as writer:
        for t in range(T):
            ax.cla()

            s = max(0, t - trail_len + 1)
            e = t + 1

            for i in range(N):
                gt_seg = GT[i, s:e, :]
                pd_seg = pred[i, s:e, :]

                ax.plot(gt_seg[:, 0], gt_seg[:, 1], gt_seg[:, 2],
                        color="red", linewidth=1.2, alpha=0.9)
                ax.plot(pd_seg[:, 0], pd_seg[:, 1], pd_seg[:, 2],
                        color="green", linewidth=1.2, alpha=0.9)

                ax.scatter(GT[i, t, 0], GT[i, t, 1], GT[i, t, 2],
                           color="red", s=8, alpha=1.0)
                ax.scatter(pred[i, t, 0], pred[i, t, 1], pred[i, t, 2],
                           color="green", s=8, alpha=1.0)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_zticks(zticks)

            ax.set_box_aspect((1, 1, 1))
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(
                f"{title} | frame {t+1}/{T} | trail={trail_len} | "
                f"tick={tick_step:g} | d=({delta_x},{delta_y},{delta_z})"
            )

            gt_proxy, = ax.plot([], [], [], color="red", label="GT")
            pred_proxy, = ax.plot([], [], [], color="green", label="Pred")
            ax.legend(handles=[gt_proxy, pred_proxy], loc="upper right")

            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(frame)

    plt.close(fig)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type,w2c,k):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    Point_clouds = []
    # breakpoint()
    print("point nums:",gaussians._xyz.shape[0])
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()

        
        rendering= render(view, gaussians, pipeline, background,cam_type=cam_type)["render"]
        xyz= render(view, gaussians, pipeline, background,cam_type=cam_type)["means3D"]

        xyz_np = xyz.cpu().numpy()
        

        Point_clouds.append(xyz_np)
        

        #-----------------
        # if idx ==0:break
        render_images.append(to8b(rendering).transpose(1,2,0))
        # print(to8b(rendering).shape)
        render_list.append(rendering)
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            gt_list.append(gt)
        # if idx >= 10:
            # break
        
    return Point_clouds


def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, args):
    skip_train = args.skip_train
    skip_test = args.skip_test
    skip_video = args.skip_video
    source = args.source
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # print(gaussians.get_xyz.shape)
        # print(gaussians.compute_deformation(0))

        # if not skip_train:
        #     render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type)
        if not skip_test:
            datadir = source
            cam_id = 9
            k, w2c = get_k_w2c(datadir,cam_id)
            
           
            Point_clouds = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type,w2c,k)

            plot_gt_pred_3d_trajectories_mp4(args, Point_clouds, os.path.join(dataset.model_path, "test", "GT_vs_Pred_3D_Trajectories.mp4"), axis_range=1)
            
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--ID", default=0, type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--source", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

   
    #print("1")
    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args)