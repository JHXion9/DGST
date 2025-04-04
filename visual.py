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
# 点云采样间隔
traj_frac = 4
# 轨迹长度
traj_length = 15
near, far = 0.2, 1000
view_scale = 1
w, h = 2200, 3208
fps = 30
def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()

def smooth_tracks(tracks, smoothing_factor=0.3):
    """
    对轨迹进行平滑处理。

    参数:
    tracks (numpy.ndarray): 轨迹数据，形状为 (F, N, 3)，其中 F 是帧数，N 是关键点数量。
    smoothing_factor (float): 平滑因子，值越大，平滑程度越高。

    返回:
    numpy.ndarray: 平滑后的轨迹数据，形状为 (F, N, 3)。
    """
    smoothed_tracks = np.zeros_like(tracks)
    for i in range(tracks.shape[1]):
        for j in range(3):
            spline = UnivariateSpline(np.arange(tracks.shape[0]), tracks[:, i, j], s=smoothing_factor)
            smoothed_tracks[:, i, j] = spline(np.arange(tracks.shape[0]))
    return smoothed_tracks

def calculate_3d_epe(gt_tracks, predicted_tracks):
    """
    计算3D端点误差 (EPE)

    参数:
    gt_tracks: 地面实况轨迹，形状为 (T, N, 3)，其中 T 是时间步数, N 是目标数量, 3 是坐标 (x, y, z)
    predicted_tracks: 预测轨迹，形状为 (T, N, 3)

    返回:
    mean_epe: EPE的均值
    """
    # 确保输入形状一致
    assert gt_tracks.shape == predicted_tracks.shape, "地面实况轨迹和预测轨迹的形状必须一致"
    
    # 获取时间步数和目标数量
    T, N, _ = gt_tracks.shape
    
    # 初始化EPE数组
    epe = np.zeros(T)
    
    # 计算每个时间步的EPE
    for t in range(T):
        # 计算当前时间步的欧几里得距离
        distance = np.linalg.norm(gt_tracks[t, :, :] - predicted_tracks[t, :, :], axis=1)
        # 计算平均距离
        epe[t] = np.mean(distance)
    
    # 计算EPE的均值
    mean_epe = np.mean(epe)
    
    return mean_epe*13

def calculate_accuracy_within_thresholds(gt_tracks, predicted_tracks, threshold_5mm=0.038, threshold_10mm=0.076):
    """
    计算落在给定阈值内的真实3D位置点的百分比。
                单位1 = 13cm
    参数:
    gt_tracks (numpy.ndarray): 真实轨迹的3D点数组，形状为 (N, 3)。
    predicted_tracks (numpy.ndarray): 预测轨迹的3D点数组，形状为 (N, 3)。
    threshold_5cm (float): 5mm阈值，默认为 0.038。
    threshold_10cm (float): 10mm阈值，默认为0.076。

    返回:
    tuple: 包含两个浮点数，分别表示落在5毫米和10毫米阈值内的点的百分比。
    """
    # 计算每个预测点与真实点之间的欧几里得距离
    distances = np.linalg.norm(predicted_tracks - gt_tracks, axis=1)

    # 判断每个点是否在5mm阈值内
    within_threshold_5mm = distances <= threshold_5mm

    # 判断每个点是否在10mm阈值内
    within_threshold_10mm = distances <= threshold_10mm

    # 计算5mm阈值内的百分比
    percentage_within_threshold_5mm = np.mean(within_threshold_5mm) * 100

    # 计算10mm阈值内的百分比
    percentage_within_threshold_10mm = np.mean(within_threshold_10mm) * 100

    return percentage_within_threshold_5mm, percentage_within_threshold_10mm

def calculate_survival_rate(gt_tracks, pred_tracks, video_length, failure_threshold=0.384):
    """
    计算Survival率。

    参数:
    gt_tracks (numpy.ndarray): 真实的3D关键点轨迹，形状为 (F, N, 3)，其中 F 是帧数，N 是关键点数量。
    pred_tracks (numpy.ndarray): 预测的3D关键点轨迹，形状为 (F, N, 3)。
    video_length (int): 视频的总帧数。
    failure_threshold (float): 跟踪失败的阈值，默认为5cm = 0.384。

    返回:
    float: Survival率。
    """
    num_frames, num_keypoints, _ = gt_tracks.shape
    #failure_frames = []
    failure_frames = 0


    for frame in range(num_frames):
        distances = np.linalg.norm(gt_tracks[frame] - pred_tracks[frame], axis=1)
        if (distances > failure_threshold).any():
            failure_frames += 1


    # if not failure_frames:
    #     average_survival_frames = video_length
    # else:
    #     average_survival_frames = np.mean(failure_frames)

    survival_rate = (video_length-failure_frames) / video_length
    return survival_rate

def calculate_median_trajectory_error(gt_tracks, pred_tracks):
    """
    计算中位轨迹误差（MTE）。

    参数:
    gt_tracks (numpy.ndarray): 真实的3D关键点轨迹，形状为 (F, N, 3)，其中 F 是帧数，N 是关键点数量。
    pred_tracks (numpy.ndarray): 预测的3D关键点轨迹，形状为 (F, N, 3)。

    返回:
    float: 中位轨迹误差（MTE）。
    """
    num_frames, num_keypoints, _ = gt_tracks.shape
    distances = []

    for frame in range(num_frames):
        frame_distances = np.linalg.norm(gt_tracks[frame] - pred_tracks[frame], axis=1)
        distances.extend(frame_distances)

    median_error = np.median(distances)
    return median_error*13

def create_3d_animation(gt_tracks, pred_tracks, output_dir, filename='trajectory_animation.gif'):
    """
    创建一个3D动画来可视化真实的和预测的3D轨迹。

    参数:
    gt_tracks (numpy.ndarray): 真实的3D轨迹，形状为 (F, N, 3)，其中 F 是帧数，N 是关键点数量。
    pred_tracks (numpy.ndarray): 预测的3D轨迹，形状为 (F, N, 3)。
    output_dir (str): 输出动画文件的目录。
    filename (str): 输出动画文件的名称，默认为 'trajectory_animation.gif'。
    """
    gt_tracks = gt_tracks * 30
    pred_tracks = pred_tracks * 30
    # 创建画布和轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 初始化点云和轨迹
    gt_points = [ax.plot([], [], [], 'o', markersize=2, color='blue')[0] for _ in range(gt_tracks.shape[1])]
    pred_points = [ax.plot([], [], [], 'o', markersize=2, color='red')[0] for _ in range(pred_tracks.shape[1])]
    gt_trajectories = [ax.plot([], [], [], '-', linewidth=0.5, color='blue')[0] for _ in range(gt_tracks.shape[1])]
    pred_trajectories = [ax.plot([], [], [], '-', linewidth=0.5, color='red')[0] for _ in range(pred_tracks.shape[1])]

    # 初始化函数
    def init():
        x_min = min(np.min(gt_tracks[:, :, 0]), np.min(pred_tracks[:, :, 0]))
        x_max = max(np.max(gt_tracks[:, :, 0]), np.max(pred_tracks[:, :, 0]))
        y_min = min(np.min(gt_tracks[:, :, 1]), np.min(pred_tracks[:, :, 1]))
        y_max = max(np.max(gt_tracks[:, :, 1]), np.max(pred_tracks[:, :, 1]))
        z_min = min(np.min(gt_tracks[:, :, 2]), np.min(pred_tracks[:, :, 2]))
        z_max = max(np.max(gt_tracks[:, :, 2]), np.max(pred_tracks[:, :, 2]))
        
        ax.set_xlim(np.floor(x_min * 10) / 10, np.ceil(x_max * 10) / 10)
        ax.set_ylim(np.floor(y_min * 10) / 10, np.ceil(y_max * 10) / 10)
        ax.set_zlim(np.floor(z_min * 10) / 10, np.ceil(z_max * 10) / 10)
        
        # 设置刻度间隔为0.1
        # ax.set_xticks(np.arange(np.floor(x_min * 10) / 10, np.ceil(x_max * 10) / 10 + 0.1, 0.1))
        # ax.set_yticks(np.arange(np.floor(y_min * 10) / 10, np.ceil(y_max * 10) / 10 + 0.1, 0.1))
        # ax.set_zticks(np.arange(np.floor(z_min * 10) / 10, np.ceil(z_max * 10) / 10 + 0.1, 0.1))
        
        # 添加图例
        ax.legend([gt_points[0], pred_points[0]], ['Ground Truth', 'Prediction'], loc='upper left')
        
        return gt_points + pred_points + gt_trajectories + pred_trajectories

    # 更新函数
    def update(frame):
        for i in range(gt_tracks.shape[1]):
            # 更新真实轨迹的点
            x_gt = gt_tracks[frame, i, 0]
            y_gt = gt_tracks[frame, i, 1]
            z_gt = gt_tracks[frame, i, 2]
            gt_points[i].set_data([x_gt], [y_gt])
            gt_points[i].set_3d_properties([z_gt], 'z')

            # 更新真实轨迹的线
            if frame > 0:
                x_traj_gt, y_traj_gt, z_traj_gt = gt_trajectories[i].get_data_3d()
                x_traj_gt = np.concatenate((x_traj_gt, [x_gt]))
                y_traj_gt = np.concatenate((y_traj_gt, [y_gt]))
                z_traj_gt = np.concatenate((z_traj_gt, [z_gt]))
            else:
                x_traj_gt, y_traj_gt, z_traj_gt = [x_gt], [y_gt], [z_gt]
            gt_trajectories[i].set_data(x_traj_gt, y_traj_gt)
            gt_trajectories[i].set_3d_properties(z_traj_gt, 'z')

            # 更新预测轨迹的点
            x_pred = pred_tracks[frame, i, 0]
            y_pred = pred_tracks[frame, i, 1]
            z_pred = pred_tracks[frame, i, 2]
            pred_points[i].set_data([x_pred], [y_pred])
            pred_points[i].set_3d_properties([z_pred], 'z')

            # 更新预测轨迹的线
            if frame > 0:
                x_traj_pred, y_traj_pred, z_traj_pred = pred_trajectories[i].get_data_3d()
                x_traj_pred = np.concatenate((x_traj_pred, [x_pred]))
                y_traj_pred = np.concatenate((y_traj_pred, [y_pred]))
                z_traj_pred = np.concatenate((z_traj_pred, [z_pred]))
            else:
                x_traj_pred, y_traj_pred, z_traj_pred = [x_pred], [y_pred], [z_pred]
            pred_trajectories[i].set_data(x_traj_pred, y_traj_pred)
            pred_trajectories[i].set_3d_properties(z_traj_pred, 'z')

        return gt_points + pred_points + gt_trajectories + pred_trajectories

    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(gt_tracks), init_func=init, blit=True, interval =50 )

    # 保存动画为GIF文件（可选）
    output_path = os.path.join(output_dir, filename)
    ani.save(output_path, writer='pillow')

def get_k_w2c(datadir, cam_id):
    cameras_extrinsic_file = os.path.join(datadir, "sparse_/images.bin")
    cameras_intrinsic_file = os.path.join(datadir, "sparse_/cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    value = f'image{cam_id}.jpg'
    for idx, key in enumerate(cam_extrinsics):
        if cam_extrinsics[key].name == value:
            extr_id = key
            intr_id = cam_extrinsics[key].camera_id

    extr = cam_extrinsics[extr_id]
    R = qvec2rotmat(extr.qvec)
    T = np.array(extr.tvec)
    k = cam_intrinsics[intr_id].params
    k = np.array([[k[0], 0, k[1]],
                  [0, k[0], k[2]],
                  [0, 0, 1]])
    r =R.T
    tt = -r@T
    extrinsic_matrix = np.hstack([r,tt.reshape(3,1)])
    extrinsic_matrix = np.vstack([extrinsic_matrix,np.array([0,0,0,1])])
    w2c = np.linalg.inv(extrinsic_matrix)
    return k, w2c

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)

def setup_camera(w, h, k, w2c, near=0.2, far=1000):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    # 获取相机在世界坐标系中的中心点位置
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)

    fovX = 2*math.atan(w/(2*fx))
    fovY = 2*math.atan(h/(2*fy))
    tanHalfFovX = math.tan(fovX/2)
    tanHalfFovY = math.tan(fovY/2)
    top = tanHalfFovY * near
    bottom = -top
    right = tanHalfFovX * near
    left = -right
   

    # 根据内参和指定的near 和 far 裁剪平面距离，创建一个openGL风格的投影矩阵，用于将3D点从摄像机坐标系变换到规范化设备坐标系（NDC）
    opengl_proj = torch.tensor([[2 * near/(right-left), 0.0                    , (right+left) / (right-left), 0.0],
                                [0.0                  , 2 * near / (top-bottom), (top+bottom) / (top - bottom), 0.0],
                                [0.0                  , 0.0                    , far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    # 完整的变换矩阵，这个矩阵可以将世界坐标系中的点直接变换到NDC。
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c, # w2c
        projmatrix=full_proj,# 相机的投影矩阵 用于将点从世界坐标系变换到规范化设备坐标系
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug= False
    )
    return cam

def xjh_render(w2c, k, timestep_data):
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        im, _, depth, = Renderer(raster_settings=cam)(**timestep_data)
        return im, depth

def init_camera(y_angle=-10, center_dist=1, cam_height=1.3, f_ratio=0.82):
    # 摄像机绕y轴的旋转角度
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k

def direction_to_color(direction):
    """
    根据方向向量生成颜色，这里假设方向向量已经归一化
    可以根据需要自定义颜色映射
    """
    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.array([1.0, 1.0, 1.0])  # 白色
    
    x, y = direction[:2]  # 假设方向向量是3维的，只考虑x和y
    if x > 0 and y > 0:
        return np.array([1.0, 1.0, 0.6])  # 浅黄色
    elif x < 0 and y > 0:
        return np.array([0.6, 1.0, 0.6])  # 浅绿色
    elif x < 0 and y < 0:
        return np.array([0.6, 0.6, 1.0])  # 浅蓝色
    elif x > 0 and y < 0:
        return np.array([1.0, 0.6, 0.6])  # 浅红色
    elif x > 0:
        return np.array([1.0, 1.0, 0.6])  # 浅黄色
    elif x < 0:
        return np.array([0.6, 0.6, 1.0])  # 浅蓝色
    elif y > 0:
        return np.array([0.6, 1.0, 0.6])  # 浅绿色
    else:
        return np.array([1.0, 0.6, 0.6])  # 浅红色
    
direction_color_map = {
    (1, 1): np.array([0.5020, 0.5020, 0.2510])*255,  # 浅黄色
    (-1, 1): np.array([0.2510, 0.5020, 0])*255,  # 浅绿色
    (-1, -1): np.array([0, 0.2510, 0.5020])*255,  # 浅蓝色
    (1, -1): np.array([0.7529, 0.2510, 0])*255,  # 浅红色
    (1, 0): np.array([0.5020, 0.5020, 0.2510])*255,  # 浅黄色
    (-1, 0): np.array([0, 0.2510, 0.5020])*255,  # 浅蓝色
    (0, 1): np.array([0.2510, 0.5020, 0])*255,  # 浅绿色
    (0, -1): np.array([0.7529, 0.2510, 0])*255,  # 浅红色
}

def get_direction_key(direction):
    """
    将方向向量转换为查找表的键
    """
    x, y = direction[:2]
    return (np.sign(x), np.sign(y))

def make_lineset(all_pts, cols, num_lines):
    linesets = []
    # 遍历每一个[t，t + traj_frac]
    #------------- 设计方向颜色轨迹
    # for pts, directions in zip(all_pts, all_directions):
    #     lineset = o3d.geometry.LineSet()
    #     lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    #     cols = np.array([colormap[get_direction_key(d)] for d in directions])
       
    #     print("线的数量",num_lines)
    #     print("点shape",pts.shape)
    #     print("颜色shape",cols.shape)
    #     lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
    #     # 生成点的索引
    #     pt_indices = np.arange(len(lineset.points))
    #     # 生成线的索引
    #     line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        
    #     lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
    #     linesets.append(lineset)
    for pts in all_pts:
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        # 生成点的索引
        pt_indices = np.arange(len(lineset.points))
        # print("pt_indices",pt_indices.shape)
        # 生成线的索引
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        # print("line_indices",line_indices.shape)
        # print("line_indices",line_indices)
        
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets

def filter_points(pcd, data_dir, k, w2c):
    # mask_dir = os.path.join(data_dir, "mask/cam09/")
    # # 获取目录中所有文件的列表
    # mask_files = os.listdir(mask_dir)

    # # 按文件名排序
    # mask_files.sort()

    # # 读取第一帧的mask图像
    # first_mask_path = os.path.join(mask_dir, mask_files[0])
    # 读取第一帧的mask图像
    first_mask_path = os.path.join(data_dir, 'mask/frame_00001.jpg')
    first_mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)

    # 检查图像是否成功读取
    if first_mask is None:
        raise ValueError(f"Failed to read mask image: {first_mask_path}")

    all_foreground_points = []
    foreground_indices = []

    # 遍历所有帧
    for frame_idx, points in enumerate(pcd):
        if frame_idx == 0:  # 仅在第一帧进行筛选
            # 将点云转换为numpy数组
            points_homo = np.hstack((points, np.ones((points.shape[0], 1))))
            points_cam = np.dot(w2c, points_homo.T).T
            points_cam = points_cam[:, :3]

            points_img = np.dot(k, points_cam.T).T
            points_img = (points_img[:, :2] / points_img[:, 2][:, None]).astype(int)
            
            # 筛选前景点云
            foreground_points = []
            foreground_indices = []
            for i in range(points_img.shape[0]):
                x, y = points_img[i]
                if 0 <= x < first_mask.shape[1] and 0 <= y < first_mask.shape[0] and first_mask[y, x] > 0:
                    foreground_points.append(points[i])
                    foreground_indices.append(i)
            
            # 将筛选后的前景点云添加到列表中
            all_foreground_points.append(np.array(foreground_points))
        else:
            # 后续帧直接使用第一帧筛选后的前景点云进行筛选
            foreground_points = points[foreground_indices]
            all_foreground_points.append(foreground_points)

    return all_foreground_points
# 计算轨迹
def calculate_trajectories(scene_data, data_dir, k, w2c):
    # 提取前景物体的三维坐标
    in_pts = [data['means3D'][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    
    fin_pts = filter_points(in_pts,data_dir,k,w2c)
    
    num_lines = len(fin_pts[0])
    cols = np.repeat(colormap[np.arange(len(fin_pts[0])) % len(colormap)][None], traj_length, 0).reshape(-1, 3)
    # 计算移动方向
    all_directions = []
    for pts in fin_pts:
        directions = np.diff(pts, axis=0)
        directions = np.vstack((directions, directions[-1]))  # 保持与pts相同的长度
        all_directions.append(directions)
   

    #cols = np.repeat(colormap[np.arange(len(in_pts[0])) % len(colormap)][None], traj_length, 0).reshape(-1, 3)
    #print("颜色shape111",cols.shape)
    out_pts = []
    out_directions = []
    for t in range(len(fin_pts))[traj_length:]:
        out_pts.append(np.array(fin_pts[t - traj_length:t+1 ]).reshape(-1, 3))
        out_directions.append(np.array(all_directions[t - traj_length:t ]).reshape(-1, 3))
    # a = np.array(out_pts)
    # print("out_pts",a.shape)
    # assert 0
    # print("out_pts",len(out_pts))
    # print("num_lines",num_lines)
    return make_lineset(out_pts, cols, num_lines)
    
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

def render_set(id, gtname, model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    op_dir = f"/media/DGST_data/trajectory/{id}-{gtname}"
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    Point_clouds = []
    Depth = []
    data_list = []
    scene_data = []
    measure_PC = []
    # breakpoint()
    print("point nums:",gaussians._xyz.shape[0])
    with open(os.path.join(op_dir,'_xyz.json'), 'r') as file:
        data_list = json.load(file)

    # 将 Python 列表转换回 numpy 数组
    GT = np.array(data_list)
    
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # breakpoint()
        
        
        Render = render(view, gaussians, pipeline, background,cam_type=cam_type)
        rendering = Render["render"]
        xyz = Render["means3D"]
        depth = Render["depth"]
        rendervar = Render["rendervar"]
       
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # print(idx)
        
        
        

        #-----------------增加
        xyz_np = xyz.cpu().numpy()
       
        if idx == 0:
            tree = cKDTree(xyz_np)
            distance, indices = tree.query(GT[idx])
            print("GT值",GT[idx])
            print("临近点",xyz_np[indices[0]])
            print("indices",indices)
            
        
        measure_PC.append([xyz_np[indices[0]],xyz_np[indices[1]]])

        Point_clouds.append(xyz_np)

        data_list.append(xyz.tolist())

       # Depth.append(depth)
        scene_data.append(rendervar)
 

        #-----------------
        # if idx ==0:break
        render_images.append(to8b(rendering).transpose(1,2,0))
        # print(to8b(rendering).shape)
        render_list.append(rendering)
    
    # multithread_write(Depth, depth_path)
    
    # # print("writing rendering images.")

    # multithread_write(render_list, render_path)
   
    EPE = calculate_3d_epe(GT, np.array(measure_PC))
    print("EPE: ", EPE)
    
    e2, e4 = calculate_accuracy_within_thresholds(GT, np.array(measure_PC))
    print("e2, e4: ", e2, e4)

    survival = calculate_survival_rate(GT, np.array(measure_PC), 150)
    print("Survival: ", survival)

    MTE = calculate_median_trajectory_error(GT, np.array(measure_PC))
    print("MTE: ", MTE)
    
    with open(os.path.join(model_path,"results.json"), 'r') as file:
        data = json.load(file)
    data['ours_30000']['EPE'] = EPE
    data['ours_30000']['e2'] = e2
    data['ours_30000']['e4'] = e4
    data['ours_30000']['Survival'] = survival
    data['ours_30000']['MTE'] = MTE
    with open(os.path.join(model_path,"results.json"), 'w') as file:
        json.dump(data, file, indent=4)
    
    create_3d_animation(smooth_tracks(GT),smooth_tracks(np.array(measure_PC)), model_path,'GT-pred_trajectory.gif')

    # open3D 可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(w ), height=int(h ), visible=True)
    datadir = f"/media/DGST_data/Data/{id}"
    cam_id = 9
    k, w2c = get_k_w2c(datadir,cam_id)

#     # 用cv2保存depth图片
#     #multithread_write(depth, depth_path)
#    # print(scene_data[0].keys())
#     multithread_write(im, depth_path)
#     multithread_write(depth, depth_path)
    
    # 如果只需点云轨迹  就不需要下行代码
    # init_pts, init_cols = rgbd2pcd(render_list[0], Depth[0], w2c, k)
    

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = init_pts
    # pcd.colors = init_cols
    # vis.add_geometry(pcd)

    linesets = None
    lines = None

    linesets = calculate_trajectories(scene_data, datadir,k ,w2c)
    lines = o3d.geometry.LineSet()
    lines.points = linesets[0].points
    lines.colors = linesets[0].colors
    lines.lines = linesets[0].lines
    
    vis.add_geometry(lines)

    # 调整相机的内参矩阵，以便在可视化时 适当缩放视图
    view_k = k * view_scale
    view_k[2, 2] = 1
    view_control = vis.get_view_control() 
    # cparams 用于存储相机参数
    cparams = o3d.camera.PinholeCameraParameters()
    cparams.extrinsic = w2c
    view_k[1,2]=0
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(h * view_scale)
    cparams.intrinsic.width = int(w * view_scale) 

    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = view_scale
    render_options.light_on = False
    start_time = time.time()
    num_timesteps = len(Point_clouds)
    while True:
        passed_time = time.time() - start_time
        passed_frames = passed_time * fps # 意味着1s视频渲染20帧
        t = int(passed_frames % (num_timesteps - traj_length)) + traj_length  # Skip t that don't have full traj.
        # cam_params = view_control.convert_to_pinhole_camera_parameters()
        # view_k = cam_params.intrinsic.intrinsic_matrix
        # k = view_k / view_scale
        # k[2, 2] = 1
        # w2c = cam_params.extrinsic


        # 如果只需点云轨迹  就不需要下行代码
        # pts, cols = rgbd2pcd(render_list[t], Depth[t], w2c, k)
        # pcd.points = pts
        # pcd.colors = cols
        # vis.update_geometry(pcd)

        lt = t - traj_length
        lines.points = linesets[lt].points
        lines.colors = linesets[lt].colors
        lines.lines = linesets[lt].lines
        vis.update_geometry(lines)

        if not vis.poll_events():
            break
        vis.update_renderer()
    vis.destroy_window()
    del view_control
    del vis
    del render_options
    print("Done rendering!!!!")


def render_sets(id, gtname,dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams):
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

        
        render_set(id, gtname, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type)

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
    render_sets(args.ID,args.GT_name, model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args))