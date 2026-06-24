from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                R,T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask=None
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
    
                mask = caminfo.mask
            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                              mask=mask)
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)


class FourDGSdataset_window(Dataset):
    """
    滑动窗口数据集，返回连续的 2 帧（t 和 t+1）用于轨迹损失计算。
    """
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type = dataset_type
        self.window_size = 2  # 只需要 t 和 t+1
        

    def _process_single_item(self, original_index):
        """
        内部辅助方法，用于将原始数据集中的单个条目转换为一个 Camera 对象。
        这部分逻辑是从你提供的原代码中提取出来的。
        """

        image, w2c, time = self.dataset[original_index]
        R, T = w2c
        
        FovX = focal2fov(self.dataset.focal[0], image.shape[2])
        FovY = focal2fov(self.dataset.focal[0], image.shape[1])
        mask = None # 保持与原代码一致

        # 注意：这里的 data_device 硬编码为 "cuda"，可能需要根据实际运行环境调整
        return Camera(colmap_id=original_index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, gt_alpha_mask=None,
                      image_name=f"{original_index}", uid=original_index, data_device=torch.device("cuda"), time=time,
                      mask=mask)

    def __getitem__(self, index):
        # 窗口长度为2，步长为1
        # index 代表窗口的起始位置，返回 [cam_t, cam_t+1]
        
        if index + 1 >= len(self.dataset):
            raise IndexError(f"Window starting at {index} goes out of bounds.")

        window_of_cameras = []
        for i in range(self.window_size):
            camera_obj = self._process_single_item(index + i)
            window_of_cameras.append(camera_obj)
        
        return window_of_cameras

    def __len__(self):
        # 窗口数量 = 原始长度 - 窗口大小 + 1
        original_length = len(self.dataset)
        
        if original_length < self.window_size:
            return 0
        
        return original_length - self.window_size + 1