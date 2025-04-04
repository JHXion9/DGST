import open3d as o3d
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary
import math

# 点云采样间隔
traj_frac = 8
# 轨迹长度
traj_length = 15

w, h = 2200, 3208


direction_color_map = {
    0: np.array([255, 0, 0]),  # 0-45 (红色)
    1: np.array([255, 165, 0]),  # 45-90 (橙色)
    2: np.array([255, 255, 0]),  # 90-135 (黄色)
    3: np.array([0, 255, 0]),  # 135-180 (绿色)
    4: np.array([0, 255, 255]),  # 180-225 (青色)
    5: np.array([0, 0, 255]),  # 225-270 (蓝色)
    6: np.array([128, 0, 128]),  # 270-315 (紫色)
    7: np.array([255, 192, 203]),  # 315-360 (粉色)
    8: np.array([255, 255, 255])
}


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
    r = R.T
    tt = -r @ T
    extrinsic_matrix = np.hstack([r, tt.reshape(3, 1)])
    extrinsic_matrix = np.vstack([extrinsic_matrix, np.array([0, 0, 0, 1])])
    w2c = np.linalg.inv(extrinsic_matrix)
    return k, w2c


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


def trajectory(points):
    pts_slice = [point[::traj_frac].contiguous().float().cpu().numpy() for point in points]
    num_lines = len(pts_slice[0])
    pts_group = np.array(pts_slice).reshape(-1, 3)

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts_group, np.float64))

    # 生成点的索引
    pt_indices = np.arange(len(lineset.points))

    # 生成线的索引
    line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]

    # 计算每条线段的方向和颜色
    cols = []
    for start, end in line_indices:
        direction = pts_group[start] - pts_group[end]  # 使用线段的起点和终点计算方向
        color = direction_color_map.get(get_direction_key(direction), [255, 255, 255]) / 255.0
        cols.append(color)

    lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
    lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))

    return lineset


to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)


def render_set(id, model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    traj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "traj")
    makedirs(traj_path, exist_ok=True)
    Point_clouds = []
    frame_count = 0
    datadir = "./data/multipleview/{}".format(id)
    cam_id = 3

    renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # 设置背景颜色为白色

    # 相机参数设置
    k, w2c = get_k_w2c(datadir, cam_id)
    renderer.setup_camera(k, w2c, w, h)

    # 创建材质记录
    line_material = o3d.visualization.rendering.MaterialRecord()
    line_material.shader = "unlitLine"  # 使用无光照的线条着色器
    line_material.line_width = 1.0  # 设置线条宽度

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        Render = render(view, gaussians, pipeline, background, cam_type=cam_type)
        xyz = Render["means3D"]
        Point_clouds.append(xyz)

        if len(Point_clouds) == traj_length:
            lineset = trajectory(Point_clouds)
            lines = o3d.geometry.LineSet()
            lines.points = lineset.points
            lines.lines = lineset.lines
            lines.colors = lineset.colors

            # 添加几何体到场景
            renderer.scene.clear_geometry()  # 清除之前的几何体
            renderer.scene.add_geometry("head_traj", lines, line_material)

            # 渲染图像
            img = renderer.render_to_image()

            # 保存图像
            image_path = os.path.join(traj_path, f"frame_{frame_count:04d}.png")
            o3d.io.write_image(image_path, img)

            frame_count += 1
            Point_clouds.pop(0)
    print("Done rendering!!!!")


def render_sets(id, dataset: ModelParams, hyperparam, iteration: int, pipeline: PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        cam_type = scene.dataset_type
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(id, dataset.model_path, "video", scene.loaded_iter, scene.getVideoCameras(), gaussians,
                   pipeline, background, cam_type)


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
    parser.add_argument("--skip_visual", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering ", args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(args.ID, model.extract(args), hyperparam.extract(args), args.iteration,
                pipeline.extract(args))
