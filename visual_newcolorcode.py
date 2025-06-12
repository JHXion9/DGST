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
import open3d as o3d
from scipy.interpolate import UnivariateSpline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import json
import imageio
import open3d as o3d
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
import time
# import torch.multiprocessing as mp
import threading
import concurrent.futures
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
            read_extrinsics_binary, read_intrinsics_binary
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from tool.colormap import colormap
import math
from scipy.spatial import cKDTree
from tool.utils import smooth_tracks, get_k_w2c, multithread_write
from tool.eval import calculate_3d_mte, calculate_accuracy_within_thresholds, calculate_survival_rate
from tool.vis import create_3d_animation, filter_points, make_lineset
# 点云采样间隔
traj_frac = 4
# 轨迹长度
traj_length = 15
near, far = 0.2, 1000
view_scale = 1/4
w, h = 2200, 3208
fps = 30
def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()


def init_camera(y_angle=-10, center_dist=1, cam_height=1.3, f_ratio=0.82):
    # 摄像机绕y轴的旋转角度
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k

# 计算轨迹
def calculate_all_trajectories(scene_data, data_dir, k, w2c):
    # 提取前景物体的三维坐标
    in_pts = [data['means3D'][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    
    # fin_pts = filter_points(in_pts,data_dir,k,w2c)
    fin_pts = in_pts
    
    num_lines = len(fin_pts[0])
    # 计算移动方向
    all_directions = []
    for pts in fin_pts:
        directions = np.diff(pts, axis=0)
        directions = np.vstack((directions, directions[-1]))  # 保持与pts相同的长度
        all_directions.append(directions)
   
    out_pts = []
    out_directions = []
    for t in range(len(fin_pts))[traj_length:]:
        out_pts.append(np.array(fin_pts[t - traj_length:t+1 ]).reshape(-1, 3))
        out_directions.append(np.array(all_directions[t - traj_length:t ]).reshape(-1, 3))

    return make_lineset(out_pts, num_lines)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def rgbd2pcd(im, depth, w2c, k, show_depth=False):
    # 定义了相机近端和远端的深度值
    d_near = 1.5
    d_far = 6
    # 相机内外参矩阵
    invk = torch.inverse(torch.tensor(k).cuda().float())
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    # 将深度图转换为一维数组
    radial_depth = depth[0].reshape(-1)
    # 计算默认像素点对应的射线，并对其进行归一化处理
    def_rays = (invk @ def_pix.T).T
    def_radial_rays = def_rays / torch.linalg.norm(def_rays, ord=2, dim=-1)[:, None]
    # 将射线与深度相乘，得到相机坐标系下的点
    pts_cam = def_radial_rays * radial_depth[:, None]
    # 提取相机坐标系下的点 Z坐标
    z_depth = pts_cam[:, 2]
    # 将相机坐标系下的点转换到世界坐标系下。
    pts4 = torch.concat((pts_cam, pix_ones), 1)
    pts = (c2w @ pts4.T).T[:, :3]
    # 如果show_depth为True，则将点的颜色根据深度进行映射，否则使用图像的颜色作为点的颜色。
    if show_depth:
        cols = ((z_depth - d_near) / (d_far - d_near))[:, None].repeat(1, 3)
    else:
        cols = torch.permute(im, (1, 2, 0)).reshape(-1, 3)
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols

def render_set_online(args, model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    id = args.ID
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    traj_path = os.path.join(model_path, name, f"ours_{iteration}", "traj")
    makedirs(traj_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    frame_count = 0  # 用于保存图像的计数器
    current_frame_idx = 0  # 用于追踪当前帧
    render_list = []
    Point_clouds = []  # 如果您不显示点云，可以移除此列表
    scene_data = []
 

    print("point nums:", gaussians._xyz.shape[0])

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        Render = render(view, gaussians, pipeline, background, cam_type=cam_type)
        rendering = Render["render"]
        xyz = Render["means3D"]
        rendervar = Render["rendervar"]


        xyz_np = xyz.cpu().numpy()
        Point_clouds.append(xyz_np)  # 如果您不显示点云，可以移除此行

        scene_data.append(rendervar)
        render_list.append(rendering)

    # Open3D 可视化
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=int(550 ), height=int(802 ), visible=True)

    datadir = f"/media/DGST_data/Data/{id}"  # 替换为你的路径
    cam_id = 9
    k, w2c = get_k_w2c(datadir, cam_id)

    # 初始化轨迹
    linesets = calculate_all_trajectories(scene_data, datadir, k, w2c)
    lines = o3d.geometry.LineSet()
    if linesets:
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
    vis.add_geometry(lines)

    # (可选) 初始化点云
    # pcd = o3d.geometry.PointCloud()
    # if Point_clouds:  # 确保 Point_clouds 不为空
    #     pcd.points = o3d.utility.Vector3dVector(Point_clouds[0])
    #     vis.add_geometry(pcd)

    # 调整相机内参
    view_k = k * view_scale
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    cparams.extrinsic = w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(h * view_scale)
    cparams.intrinsic.width = int(w * view_scale)
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = view_scale  # 如果您显示点云，设置点的大小
    render_options.light_on = False

    def update_and_capture(vis):
        nonlocal frame_count, current_frame_idx, Point_clouds, linesets, lines #, pcd
        
        # 修改这里：一次前进多帧
        for _ in range(traj_length):  # 一次前进 traj_length 帧

            if current_frame_idx >= len(Point_clouds): # 如果不显示点云，可以改为 len(linesets)
                print("Reached the end of the sequence.")
                return False
                
            # (可选) 更新点云
            # if Point_clouds:
            #    pcd.points = o3d.utility.Vector3dVector(Point_clouds[current_frame_idx])
            #    vis.update_geometry(pcd)

            # 更新轨迹
            lt = max(0, current_frame_idx - traj_length)
            if lt < len(linesets):
                lines.points = linesets[lt].points
                lines.colors = linesets[lt].colors
                lines.lines = linesets[lt].lines
                vis.update_geometry(lines)

            vis.poll_events()
            vis.update_renderer()

            
                
            frame_count += 1
            current_frame_idx += 1
        # 捕获屏幕 (仅 2D 图像)
        image_path = os.path.join(traj_path, f"frame_{frame_count:05d}.png")
        vis.capture_screen_image(image_path, do_render=True)
        return False

    vis.register_key_callback(ord("S"), update_and_capture)
    vis.run()
    vis.destroy_window()

    print("Done rendering!")
    return Point_clouds 

def render_sets(args, dataset : ModelParams, hyperparam, pipeline : PipelineParams):
    iteration = args.iteration
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # print( scene.gaussians._xyz.shape)
        # print(scene.gaussians._features_dc.shape)
        # print(scene.gaussians._features_rest.shape)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # print(gaussians.get_xyz.shape)
        # print(gaussians.compute_deformation(0))

        Point_cloud = render_set_online(args, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type)
        
        # calculate_metrics(args,dataset.model_path, Point_cloud)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--ID", default=0, type=str)
    parser.add_argument("--GT_name", default=0, type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_visual", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
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
    render_sets(args, model.extract(args), hyperparam.extract(args), pipeline.extract(args))