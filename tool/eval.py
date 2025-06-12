
import numpy as np

def calculate_3d_epe(gt_tracks, predicted_tracks):
    """
    计算3D端点误差 (EPE)

    参数:
    gt_tracks: 地面实况轨迹，形状为 (N, T, 3)，其中 T 是时间步数, N 是目标数量, 3 是坐标 (x, y, z)
    predicted_tracks: 预测轨迹，形状为 (N, T, 3)

    返回:
    mean_epe: EPE的均值
    """
    # 确保输入形状一致
  
    assert gt_tracks.shape == predicted_tracks.shape, "地面实况轨迹和预测轨迹的形状必须一致"
    
    # 获取时间步数和目标数量
    N, T, _ = gt_tracks.shape
    
    # 初始化EPE数组
    epe = np.zeros(T)
    
    # 计算每个时间步的EPE
    for t in range(T):
        # 计算当前时间步的欧几里得距离
        distance = np.linalg.norm(gt_tracks[:, t, :] - predicted_tracks[:, t, :], axis=1)
        # 计算平均距离
        epe[t] = np.mean(distance)
    
    # 计算EPE的均值
    mean_epe = np.mean(epe)
    
    return mean_epe*90

def calculate_accuracy_within_thresholds(gt_tracks, predicted_tracks, threshold_5mm=0.005, threshold_10mm=0.011):
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
    # 计算每个预测点与真实点之间的欧几里得距离
    distances = np.linalg.norm(predicted_tracks - gt_tracks, axis=2)

    # 判断每个点是否在5mm阈值内
    within_threshold_5mm = distances <= threshold_5mm

    # 判断每个点是否在10mm阈值内
    within_threshold_10mm = distances <= threshold_10mm

    # 计算5mm阈值内的百分比
    percentage_within_threshold_5mm = np.mean(within_threshold_5mm) * 100

    # 计算10mm阈值内的百分比
    percentage_within_threshold_10mm = np.mean(within_threshold_10mm) * 100

    return percentage_within_threshold_5mm, percentage_within_threshold_10mm

def calculate_survival_rate(gt_tracks, pred_tracks, video_length, failure_threshold=0.055):
    """
    计算Survival率。

    参数:
    gt_tracks (numpy.ndarray): 真实的3D关键点轨迹，形状为 (N, T, 3)，其中 T 是帧数，N 是关键点数量。
    pred_tracks (numpy.ndarray): 预测的3D关键点轨迹，形状为 (N, T, 3)。
    video_length (int): 视频的总帧数。
    failure_threshold (float): 跟踪失败的阈值，默认为5cm = 0.055。

    返回:
    float: Survival率。
    """
    num_keypoints, num_frames, _ = gt_tracks.shape
    #failure_frames = []
    failure_frames = 0


    for frame in range(num_frames):
        distances = np.linalg.norm(gt_tracks[:,frame,:] - pred_tracks[:,frame,:], axis=1)
        if (distances > failure_threshold).any():
            failure_frames += 1


    # if not failure_frames:
    #     average_survival_frames = video_length
    # else:
    #     average_survival_frames = np.mean(failure_frames)

    survival_rate = (video_length-failure_frames) / video_length
    return survival_rate
