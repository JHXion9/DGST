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
import wandb
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

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

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

    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)

    batch_size = opt.batch_size
    print("data loading done")
    # 使用数据加载器
    if opt.dataloader:
        # 获取训练相机
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
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack,
                batch_size=batch_size,
                shuffle=True,
                num_workers=16,
                collate_fn=list,
            )
            random_loader = True
        # 将数据加载器转换为迭代器
        loader = iter(viewpoint_stack_loader)

    # dynerf, zerostamp_init
    # breakpoint()
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
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                stage=stage,
                cam_type=scene.dataset_type,
            )
            image, viewspace_point_tensor, visibility_filter, radii, rot = (
                render_pkg["render"],
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
        # print("rot",len(rot_list))
        # print("rot_list.shpe", rot_list[0].shape)
        # print("***********", gt_images[0].shape)
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

        # 如果阶段是“fine” 且时间平滑权重不为0
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
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "psnr": f"{psnr_:.{2}f}",
                        "point": f"{total_point}",
                    }
                )
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
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
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
    torch.cuda.empty_cache()
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

    # wandb.init(
    # # set the wandb project where this run will be logged
    # project="my-awesome-project",

    # # track hyperparameters and run metadata
    # config={
    # "architecture": "4DGS",
    # "dataset": "NERSEMBLE",
    # "epochs": 10000,
    # }
    # )

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
