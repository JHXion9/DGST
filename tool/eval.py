
import numpy as np

def calculate_3d_epe(gt_tracks, predicted_tracks):
    """
    计算3D端点误差 (EPE) / Median Trajectory Error (MTE)
    计算估计轨迹和真实轨迹之间的距离中值

    参数:
    gt_tracks: 地面实况轨迹，形状为 (N, T, 3)，其中 T 是时间步数, N 是目标数量, 3 是坐标 (x, y, z)
    predicted_tracks: 预测轨迹，形状为 (N, T, 3)

    返回:
    mte: Median Trajectory Error (MTE)
    """
    # 确保输入形状一致
    assert gt_tracks.shape == predicted_tracks.shape, "地面实况轨迹和预测轨迹的形状必须一致"
    
    # 获取时间步数和目标数量
    N, T, _ = gt_tracks.shape
    
    # 计算每个关键点的中值误差
    median_errors = []
    
    for n in range(N):
        # 计算第n个关键点在所有时间步的欧几里得距离
        distances = np.linalg.norm(gt_tracks[n, :, :] - predicted_tracks[n, :, :], axis=1)
        # 计算该轨迹的中值误差（而不是平均误差）
        median_error = np.median(distances)
        median_errors.append(median_error)
    
    # 计算所有关键点中值误差的平均值（而不是中值）
    mte = np.mean(median_errors)
    # a = np.array(median_errors)
    return mte * 100

def calculate_accuracy_within_thresholds(gt_tracks, predicted_tracks, threshold_list=[0.1, 0.15, 0.2, 0.25]):
    """
    计算落在给定阈值内的真实3D位置点的百分比。
                单位1 = 16cm
    参数:
    gt_tracks (numpy.ndarray): 真实轨迹的3D点数组，形状为 (N, T ,3)。
    predicted_tracks (numpy.ndarray): 预测轨迹的3D点数组，形状为 (N, T, 3)。
    threshold_5cm (float): 5mm阈值，默认为 0.031。
    threshold_10cm (float): 10mm阈值，默认为0.062。

    返回:
    tuple: 包含两个浮点数，分别表示落在5毫米和10毫米阈值内的点的百分比。
    """
    # 将阈值列表转换为numpy数组，然后转换为实际单位
    threshold_array = np.array(threshold_list) * 0.01
    
    # 计算每个预测点与真实点之间的欧几里得距离
    distances = np.linalg.norm(predicted_tracks - gt_tracks, axis=2)

    # 判断距离是否在每个阈值内
    within_thresholds = [distances <= threshold for threshold in threshold_array]
    # 计算每个阈值内的百分比
    percentages_within_thresholds = [np.mean(within) * 100 for within in within_thresholds]

    # 计算所有阈值均值百分比
    mean_percentage_within_thresholds = np.mean(percentages_within_thresholds)

    return mean_percentage_within_thresholds

def calculate_survival_rate(gt_tracks, pred_tracks, video_length, failure_threshold=0.3):
    """
    计算Survival率（跟踪失败前的平均帧数）。

    参数:
    gt_tracks (numpy.ndarray): 真实的3D关键点轨迹，形状为 (N, T, 3)，其中 T 是帧数，N 是关键点数量。
    pred_tracks (numpy.ndarray): 预测的3D关键点轨迹，形状为 (N, T, 3)。
    video_length (int): 视频的总帧数。
    failure_threshold (float): 跟踪失败的阈值，默认为3mm = 0.3。

    返回:
    float: Survival率（平均跟踪帧数）。
    """
    failure_threshold = failure_threshold * 0.01  # 将阈值转换为实际单位
    num_keypoints, num_frames, _ = gt_tracks.shape
    
    survival_frames = []
    
    # 对每个关键点轨迹单独计算存活帧数
    for keypoint_idx in range(num_keypoints):
        tracked_frames = 0
        
        for frame in range(num_frames):
            # 计算当前关键点在当前帧的L2误差
            distance = np.linalg.norm(
                gt_tracks[keypoint_idx, frame, :] - pred_tracks[keypoint_idx, frame, :], 
                axis=0
            )
            
            # 如果误差超过阈值，则认为跟踪失败
            if distance > failure_threshold:
                break
            else:
                tracked_frames += 1
        
        survival_frames.append(tracked_frames)
    
    # 计算所有轨迹的平均存活帧数
    average_survival_frames = np.mean(survival_frames)
    
    return average_survival_frames/150
