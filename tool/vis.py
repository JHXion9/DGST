import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import cv2
from tqdm import tqdm
import torch
from tool.colormap import colormap
from tool.utils import get_k_w2c


def create_3d_animation(gt_tracks, pred_tracks, output_dir, filename='trajectory_animation.gif'):
    """
    创建一个3D动画来可视化真实的和预测的3D轨迹。

    参数:
    gt_tracks (numpy.ndarray): 真实的3D轨迹，形状为 (F, N, 3)，其中 F 是帧数，N 是关键点数量。
    pred_tracks (numpy.ndarray): 预测的3D轨迹，形状为 (F, N, 3)。
    output_dir (str): 输出动画文件的目录。
    filename (str): 输出动画文件的名称，默认为 'trajectory_animation.gif'。
    """
    gt_tracks = gt_tracks * 13
    pred_tracks = pred_tracks * 13
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

direction_color_map = {
    0: np.array([255, 0, 0]),     # 0-45 (红色)
    1: np.array([255, 165, 0]),   # 45-90 (橙色)
    2: np.array([255, 255, 0]),   # 90-135 (黄色)
    3: np.array([0, 255, 0]),     # 135-180 (绿色)
    4: np.array([0, 255, 255]),   # 180-225 (青色)
    5: np.array([0, 0, 255]),     # 225-270 (蓝色)
    6: np.array([128, 0, 128]),   # 270-315 (紫色)
    7: np.array([255, 192, 203]), # 315-360 (粉色)
    8: np.array([255, 255, 255])
}

def get_direction_key(direction):
    """
    将方向向量转换为查找表的键 (基于角度)
    """
    x, y = direction[:2]
    if x == 0 and y == 0:
      return 8
    angle = math.atan2(y, x)
    angle_deg = math.degrees(angle) % 360  # 转换为 0-360 度

    # 每 45 度一个区间
    key = int(angle_deg / 45)
    return key

def make_lineset(all_pts, num_lines):
    linesets = []
    # 遍历每一个[t，t + traj_frac]
    for pts in tqdm(all_pts):
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))

        # 生成点的索引
        pt_indices = np.arange(len(lineset.points))

        # 生成线的索引
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]

        # 计算每条线段的方向和颜色
        cols = []
        for start, end in line_indices:
            direction = pts[start] - pts[end]  # 使用线段的起点和终点计算方向
            color = direction_color_map.get(get_direction_key(direction), [255, 255, 255]) / 255.0
            cols.append(color)

        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)

    return linesets




def filter_points(pcd, data_dir, k, w2c):
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


