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
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
# import torch.multiprocessing as mp
import threading
import concurrent.futures
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
            read_extrinsics_binary, read_intrinsics_binary
from tool.utils import multithread_write, get_k_w2c, smooth_tracks
import json
from scipy.spatial import cKDTree
from tool.eval import calculate_3d_mte, calculate_accuracy_within_thresholds, calculate_survival_rate
from tool.vis import create_3d_animation
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def voxel_grid_downsample(points, voxel_size):
    # 计算每个点所在的体素网格的索引
    coords = (points / voxel_size).astype(int)  # 将点的坐标除以体素大小并转换为整数索引
    
    # 创建一个字典来存储每个体素中的点
    voxel_dict = {}
    for i, coord in enumerate(coords):
        # 将索引转换为字典的键
        key = tuple(coord)
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(points[i])
    
    # 从每个体素中选择一个代表性的点
    sampled_points = np.array([np.mean(voxel_dict[key], axis=0) for key in voxel_dict])
    
    return sampled_points
def transform_to_camera_coordinate(point_cloud, w2c):
    #point_cloud = point_cloud.cpu()
    point_cloud = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1))), axis=-1)
    
    transformed_points_homogeneous = np.dot(w2c, point_cloud.T)

    # 转置并将齐次坐标转换为非齐次坐标 (x, y, z, w) -> (x/w, y/w, z/w)  通常w=1，如果w!=1, 需要做透视除法
    transformed_points = transformed_points_homogeneous.T[:, :3] / transformed_points_homogeneous.T[:, 3:]
    return transformed_points

def project_points_to_image(point_cloud, camera_matrix):
    image_points = []
    for i in range(len(point_cloud)):
        
        image_point = np.dot(camera_matrix, point_cloud[i])

        #print(image_point)
        
        if image_point[2] != 0:
            image_point /= image_point[2]
            #print(image_point)
            # if image_point[0] > 0 and image_point[0] < 1100 and image_point[1] > 0 and image_point[1] < 1604:
            image_points.append(image_point[:2])
            

    return image_points
def point_tracking(visual_path, images, point_clouds, w2c, k):
    circle_image=[]
    for idx, image in images:
        
        transformed_point_cloud = transform_to_camera_coordinate(point_clouds[idx], w2c)
        
        

        image_points = project_points_to_image(transformed_point_cloud, k)
       
        print(f"第{idx+1}帧图片")
        print(f"存在{len(image_points)}个点")
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        for point in image_points:
            cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)  # 绘制绿色点
        #cv2.imwrite(f"./output/multipleview/face/test/ours_30000/renders_pc/{idx}.png", image)
        circle_image.append(image)
       
        
        # 判断path是否存在, 如果不存在新建
        if not os.path.exists(os.path.join(visual_path,"renders_pc")):
            os.makedirs(os.path.join(visual_path,"renders_pc"))
        cv2.imwrite(os.path.join(visual_path,"renders_pc",f"{idx}.png"), cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    imageio.mimwrite(os.path.join(visual_path,"video_circle.mp4"), circle_image, fps=30)

def calculate_metrics(args,model_path, measure_PC):
    GT_dir = f"/media/DGST_data/trajectory/{args.ID}-{args.GT_name}"
    with open(os.path.join(GT_dir,'_xyz.json'), 'r') as file:
        GT = json.load(file)
    GT = np.array(GT)
    measure_PC = np.array(measure_PC)
   
    tree = cKDTree(measure_PC[0])
    distance, indices = tree.query(GT[0])
    print("GT值",GT[0])
    print("临近点",measure_PC[0,indices[0]])
    print("indices",indices)
    
    EPE = calculate_3d_mte(GT, np.array(measure_PC[:,indices]))
    print("EPE: ", EPE)
    
    e5, e10 = calculate_accuracy_within_thresholds(GT, np.array(measure_PC[:,indices]))
    print("e5, e10: ", e5, e10)

    survival = calculate_survival_rate(GT, np.array(measure_PC[:,indices]), 150)
    print("Survival: ", survival)

    data = {}
    data['ours_30000'] = {
    'EPE': EPE,
    'e5': e5,
    'e10': e10,
    'Survival': survival
    }

    with open(os.path.join(model_path,"results.json"), 'w') as file:
        json.dump(data, file, indent=4)
    
    create_3d_animation(smooth_tracks(GT),smooth_tracks(measure_PC[:,indices]), model_path,'GT-pred_trajectory.gif')

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
        # breakpoint()
        
        rendering= render(view, gaussians, pipeline, background,cam_type=cam_type)["render"]
        xyz= render(view, gaussians, pipeline, background,cam_type=cam_type)["means3D"]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # print(idx)
        # print(means3D)
        # print(means3D.shape)
        

        #-----------------增加
        xyz_np = xyz.cpu().numpy()
        
        #print("xyz_np:",xyz_np.shape)
        #sampled_points = voxel_grid_downsample(xyz_np, voxel_size=0.5)
        #print("sampled_points:",sampled_points.shape)
        Point_clouds.append(xyz_np)
        
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz_np)
        # o3d.io.write_point_cloud("./output/multipleview/face/sync.ply", pcd)

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
        
    time2=time()
    
    print("FPS:",(len(views)-1)/(time2-time1))
    print("writing training images.")

    # multithread_write(gt_list, gts_path)
    # print("writing rendering images.")

    # multithread_write(render_list, render_path)
    # 读取renders文件夹下的图片，将其保存为images
    # images = []
    # for i in range(len(render_list)):
    #     image = cv2.imread(os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
    #     images.append((i, image))
    
    # visual_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    # point_tracking(visual_path, images, Point_clouds, w2c, k)
    print("DONE!!!!")
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30)
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
            calculate_metrics(args,dataset.model_path, Point_clouds)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type)
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