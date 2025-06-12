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
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type = dataset_type
        

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
        # 窗口长度为3，步长为1
        # index 代表窗口的起始位置
        
        # 确保当前窗口在有效范围内 (由 __len__ 已经保证，但作为防御性编程仍可保留)
        if index + 2 >= len(self.dataset):
            raise IndexError(f"Window starting at {index} goes out of bounds. Max index for a window is {len(self.dataset) - 3}.")

        window_of_cameras = []
        for i in range(3):
            # 获取并处理连续的三个数据点
            camera_obj = self._process_single_item(index + i)
            window_of_cameras.append(camera_obj)
        
        return window_of_cameras

    def __len__(self):
        # 计算滑动窗口的数量
        # 如果原始数据集长度为 N，窗口长度为 W，步长为 S，则窗口数量为 (N - W) / S + 1
        # 这里 W=3, S=1，所以是 len(self.dataset) - 3 + 1 = len(self.dataset) - 2

        original_length = len(self.dataset)
        window_size = 3

        if original_length < window_size:
            # 如果数据集太短，不足以形成一个完整的窗口，则返回 0
            return 0
        
        return original_length - window_size + 1