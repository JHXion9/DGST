import numpy as np
from scipy.interpolate import UnivariateSpline
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
            read_extrinsics_binary, read_intrinsics_binary
import os
import concurrent.futures
import torchvision
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

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

def get_k_w2c(datadir, cam_id):
    cameras_extrinsic_file = os.path.join(datadir, "sparse_/images.bin")
    cameras_intrinsic_file = os.path.join(datadir, "sparse_/cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    value = f'{cam_id}.png'
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

