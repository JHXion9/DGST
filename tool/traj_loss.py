import torch
import json
from typing import List, Dict, Any

class TrackDataset:
    """
    处理多份轨迹 JSON 文件的类：
    - 支持加载多个文件
    - 重新分配轨迹 id
    - 返回 {new_id: {"length": ..., "positions": ...}}
    - 支持按 start_frame 查询
    """

    def __init__(self) -> None:
        # 内部存储结构：
        # self._tracks[new_id] = {
        #     "start_frame": int,
        #     "length": int,
        #     "positions": list,       # 原来的 keypoints
        # }
        self._tracks: Dict[str, Dict[str, Any]] = {}

    def load_json_files(self, file_paths: List[str]) -> None:
        """
        输入多个 json 文件路径，读取并合并到内部字典中，
        为每条轨迹分配新的 id（从 0 开始递增，字符串类型）。
        """
        next_id = len(self._tracks)

        for path in file_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)  # 例如：{"5679": {...}, "1234": {...}}

            # 为了稳定性，这里按 (start_frame, 原始轨迹 id) 排个序再插入
            items = list(data.items())
            items.sort(
                key=lambda kv: (
                    kv[1].get("start_frame", 0),
                    int(kv[0]) if str(kv[0]).isdigit() else kv[0],
                )
            )

            for old_id, track in items:
                new_id = str(next_id)

                self._tracks[new_id] = {
                    "start_frame": track["start_frame"],
                    "length": track["length"],
                    "positions": track["keypoints"],  # 名字改成 positions，更符合“位置信息”
                }

                next_id += 1

    def get_all_tracks(self) -> Dict[str, Dict[str, Any]]:
        """
        返回对外用的轨迹总字典：
        key   = 新轨迹 id（字符串）
        value = {"length": 轨迹长度, "positions": 该轨迹的位置信息}
        内部的 start_frame 不暴露出去。
        """
        return {
            track_id: {
                "length": info["length"],
                "positions": info["positions"],
            }
            for track_id, info in self._tracks.items()
        }

    def get_tracks_by_start_frame(self, start_frame: int) -> Dict[str, Dict[str, Any]]:
        """
        输入起始帧，返回从该帧开始的所有轨迹：
        key   = 新轨迹 id
        value = {"length": 轨迹长度, "positions": 位置信息}
        """
        result: Dict[str, Dict[str, Any]] = {}

        for track_id, info in self._tracks.items():
            if info["start_frame"] == start_frame:
                result[track_id] = {
                    "length": info["length"],
                    "positions": info["positions"],
                }

        return result



class TrajectoryLossManager:
    def __init__(self, tracks, device='cuda'):
        """
        初始化轨迹损失管理器。
        
        Args:
            tracks: list of dict, 每个字典包含:
                - 'id': (int/str) 轨迹唯一ID
                - 'points': (Tensor or np.array) shape [L, 3] 轨迹点坐标
                - 't_start': (int) 轨迹开始的全局帧索引
                - 'weight': (float, optional) 该轨迹的置信度权重, 默认为1.0
            device: torch.device, 计算设备
        """
        self.device = device
        self.bindings = {}  # {track_id: gaussian_index} 存储动态绑定关系
        self.tracks_data = {} # 存储处理后的轨迹数据
        self.frame_to_track_indices = {} # 帧索引 -> 活跃轨迹ID列表
        self.max_frame_seen = 0

        # --- 1. 预处理数据 (移至 GPU 并建立索引) ---
        print(f"Pre-processing {len(tracks)} trajectories for loss computation...", flush=True)
        
        for i, tr in enumerate(tracks):
            tid = tr['id']
            # 转为 Tensor 并移动到 GPU，避免训练时频繁 IO
            pts = torch.as_tensor(tr['points'], dtype=torch.float32, device=device)
            t_start = tr['t_start']
            length = pts.shape[0]
            weight = float(tr.get('weight', 1.0))
            
            # 存储处理后的数据
            self.tracks_data[tid] = {
                'points': pts,
                't_start': t_start,
                'length': length,
                't_end': t_start + length - 1, # 闭区间 [start, end]
                'weight': weight
            }
            
            self.max_frame_seen = max(self.max_frame_seen, t_start + length)

            # 建立倒排索引：Frame -> List of Track IDs
            for t in range(t_start, t_start + length):
                if t not in self.frame_to_track_indices:
                    self.frame_to_track_indices[t] = []
                self.frame_to_track_indices[t].append(tid)

        print(f"Pre-processing done. Max frame index found: {self.max_frame_seen - 1}", flush=True)

    def reset_bindings(self):
        """
        在一个 Epoch 开始前调用，清空绑定关系。
        """
        self.bindings = {}

    def _compute_single_frame_loss(self, pred_points, global_t):
        """
        内部辅助函数：计算指定某一帧(global_t)的 Loss。
        无需修改，逻辑通用。
        """
        # 如果当前帧没有任何轨迹数据，返回 0
        if global_t not in self.frame_to_track_indices:
            return torch.tensor(0.0, device=self.device)
        
        active_track_ids = self.frame_to_track_indices[global_t]
        
        total_loss = torch.zeros([], device=self.device)
        total_steps = 0.0
        
        for tid in active_track_ids:
            track_info = self.tracks_data[tid]
            
            local_offset = global_t - track_info['t_start']
            if local_offset < 0 or local_offset >= track_info['length']:
                continue
                
            gt_point = track_info['points'][local_offset]

            # --- 动态绑定逻辑 ---
            if tid in self.bindings:
                bind_idx = self.bindings[tid]
            else:
                dists = torch.sum((pred_points - gt_point) ** 2, dim=-1)
                bind_idx = int(torch.argmin(dists).item())
                self.bindings[tid] = bind_idx

            # --- 计算 Loss ---
            pred_point = pred_points[bind_idx]
            loss_val = torch.sum((pred_point - gt_point) ** 2)
            total_loss += track_info['weight'] * loss_val
            total_steps += 1
            
            # --- 清理逻辑 ---
            if global_t == track_info['t_end']:
                self.bindings.pop(tid, None)

        return total_loss / max(total_steps, 1.0)

    def compute_loss(self, mus_pred_frames, t_start_frame, total_seq_len):
        """
        主调用函数：计算直接轨迹匹配损失。
        针对 [t, t+1] 两帧输入进行了逻辑修改。
        
        Args:
            mus_pred_frames: Tensor [2, N, 3], 包含 [t, t+1] 两帧的预测点云
            t_start_frame: int, mus_pred_frames[0] 对应的全局时间戳 (即 t)
            total_seq_len: int, 序列总长度 (用于判断是否是最后一步)
            
        Returns:
            loss: scalar tensor
        """
        loss = torch.zeros([], device=self.device)
        
        # --- 1. 标准情况：监督窗口的第一帧 (Index 0 -> Global t_start_frame) ---
        # 只要循环从 t=0 开始，这就覆盖了 [0, 1, ..., T-2]
        loss += self._compute_single_frame_loss(
            pred_points=mus_pred_frames[0], 
            global_t=t_start_frame
        )
        
        # --- 2. 边界处理：序列结尾 (End Boundary) ---
        # 如果当前窗口是最后一步 [T-2, T-1]
        # mus_pred_frames[0] 是 T-2 (上面已经监督了)
        # mus_pred_frames[1] 是 T-1 (最后一帧)，必须在这里补算，否则会被漏掉
        if t_start_frame == total_seq_len - 2:
            loss += self._compute_single_frame_loss(
                pred_points=mus_pred_frames[1],
                global_t=total_seq_len - 1
            )
            
        return loss
    
class RigidityLossManager:
    def __init__(self, sigma=0.01, k_neighbors=10):
        """
        初始化刚性损失管理器。

        Args:
            sigma (float): 高斯权重的带宽参数。
                           较小的 sigma 意味着刚性约束仅作用于极近的邻居（局部化）；
                           较大的 sigma 意味着约束范围更广（平滑化）。
            k_neighbors (int): 局部邻域大小 K。
        """
        self.sigma = sigma
        self.k_neighbors = k_neighbors

    def compute_loss(self, mus_pred_frames, t_start_frame, traj_manager):
        """
        计算高斯加权的局部刚性正则化损失。
        
        Args:
            mus_pred_frames: Tensor [2, N, 3], 包含 [t, t+1] 两帧的预测点云。
            t_start_frame: int, 对应 mus_pred_frames[0] 的全局时间戳 (即 t)。
            traj_manager: TrajectoryLossManager 实例, 用于获取当前的 active tracks 和 bindings。

        Returns:
            loss: scalar tensor
        """
        device = mus_pred_frames.device
        
        # 1. 提取 t 和 t+1 帧的点云
        # points_curr: P^t
        # points_next: P^{t+1}
        # 必须保证输入至少有两帧
        if mus_pred_frames.shape[0] < 2:
            return torch.tensor(0.0, device=device)

        points_curr = mus_pred_frames[0]
        points_next = mus_pred_frames[1]
        
        # 2. 获取当前帧 (t) 活跃的轨迹 ID
        if t_start_frame not in traj_manager.frame_to_track_indices:
            return torch.tensor(0.0, device=device)
        
        active_track_ids = traj_manager.frame_to_track_indices[t_start_frame]
        
        # 3. 筛选出有效的锚点高斯索引 (Anchor Indices)
        anchor_gaussian_indices = []
        
        # 获取 t+1 帧的活跃轨迹集合，用于检查连续性
        # 如果当前是最后一帧，next_frame_tracks 可能为空
        next_frame_tracks = traj_manager.frame_to_track_indices.get(t_start_frame + 1, [])
        next_frame_tracks_set = set(next_frame_tracks)
        
        for tid in active_track_ids:
            # A. 必须已经被 Direct Loss 绑定到某个高斯球上
            if tid not in traj_manager.bindings:
                continue

            # B. 连续性检查：该轨迹必须在下一帧依然存在
            # 只有当轨迹跨越了 t -> t+1，我们才认为它定义的局部区域运动是可信的
            if tid not in next_frame_tracks_set:
                continue
            
            anchor_gaussian_indices.append(traj_manager.bindings[tid])
        
        # 如果当前没有有效的锚点，返回 0
        if not anchor_gaussian_indices:
            return torch.tensor(0.0, device=device)

        # 转为 Tensor [M] (M 是当前帧有效的锚点数)
        anchor_indices = torch.tensor(anchor_gaussian_indices, dtype=torch.long, device=device)
        
        # --- 开始向量化计算 ---

        # 4. 在 t 时刻建立局部拓扑结构 (KNN)
        
        # 获取锚点在 t 时刻的位置: P_i^t -> [M, 3]
        centers_curr = points_curr[anchor_indices]

        # 计算距离矩阵：M 个锚点 vs N 个全局高斯
        # dists_mat: [M, N]
        # 注意：如果在非常大的场景下 (N > 10w)，这里可能需要优化，但在人脸 Patch (N~2w) 场景下非常快。
        dists_mat = torch.cdist(centers_curr, points_curr) 

        # 找到最近的 K 个邻居 (包含自身，距离为0)
        # knn_dists_curr: [M, K] (即公式中的 || P_i^t - P_j^t ||)
        # knn_indices:    [M, K] (邻居 j 的索引)
        knn_dists_curr, knn_indices = torch.topk(dists_mat, k=self.k_neighbors, dim=1, largest=False)

        # 5. 获取 t+1 时刻对应点的位置
        
        # 获取锚点在 t+1 时刻的位置: P_i^{t+1} -> [M, 3]
        centers_next = points_next[anchor_indices]
        
        # 获取邻居在 t+1 时刻的位置: P_j^{t+1}
        # 利用 t 时刻找到的索引 knn_indices 去 t+1 时刻取值
        # Flatten index for gathering: [M*K]
        flat_knn_indices = knn_indices.view(-1)
        
        # neighbors_next: [M, K, 3]
        neighbors_next = points_next[flat_knn_indices].view(anchor_indices.shape[0], self.k_neighbors, 3)

        # 6. 计算 t+1 时刻的相对距离
        
        # centers_next 扩维: [M, 1, 3]
        # neighbors_next:   [M, K, 3]
        # d_next:           [M, K] (即公式中的 || P_i^{t+1} - P_j^{t+1} ||)
        d_next = torch.norm(centers_next.unsqueeze(1) - neighbors_next, dim=-1)

        # 7. 计算高斯权重 w_{ij}
        # 公式: w = exp( - d_curr^2 / 2sigma^2 )
        # 关键：权重基于 t 时刻的静态结构，使用 no_grad 阻断梯度反传给权重
        with torch.no_grad():
            w_ij = torch.exp( - (knn_dists_curr ** 2) / (2 * self.sigma ** 2) )

        # 8. 计算最终损失
        # Loss = w_ij * ( (d_next - d_curr) )^2
        # d_curr 就是 knn_dists_curr
        diff = d_next - knn_dists_curr
        
        # 加权平方误差
        loss_matrix = w_ij * (diff ** 2)
        
        # 返回均值
        return loss_matrix.mean()
    
