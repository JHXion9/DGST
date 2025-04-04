import os
import cv2
import numpy as np
import argparse

def process_images_and_masks(input_dir, mask_dir):
    """
    处理图像和掩码，提取感兴趣区域并将结果替换原图像。

    Args:
        input_dir: 包含 cam01-cam06 文件夹的路径。
        mask_dir: 包含掩码图像的文件夹。
    """

    # 自动检测 input_dir 中的 camXX 文件夹
    image_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and d.startswith('cam')]
    image_dirs = sorted(image_dirs)  # 确保 cam 文件夹按顺序排列


    for image_dir in image_dirs:
        # 构建图像文件夹的完整路径
        image_dir_path = os.path.join(input_dir, image_dir)

        # 遍历图像文件夹中的所有图像
        for image_filename in os.listdir(image_dir_path):
            if image_filename.startswith('frame_') and image_filename.endswith('.png'):
                # 从文件名中提取帧编号
                try:
                    frame_num_str = image_filename.split('_')[1].split('.')[0]
                    cam_str = image_dir[3:]  # 从文件夹名中获取相机编号

                except IndexError:
                    print(f"Skipping file with unexpected name format: {image_filename}")
                    continue

                # 构建对应的掩码文件名
                mask_filename = f"cam{cam_str}_frame_{frame_num_str}.jpg"
                mask_path = os.path.join(mask_dir, mask_filename)

                # 检查掩码文件是否存在
                if not os.path.exists(mask_path):
                    print(f"Mask file not found: {mask_path}, skipping image: {image_filename}")
                    continue

                # 读取图像和掩码
                image_path = os.path.join(image_dir_path, image_filename)
                image = cv2.imread(image_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                if mask is None:
                    print(f"Failed to load mask: {mask_path}")
                    continue

                if image.shape[:2] != mask.shape[:2]:
                    print(f"Image and mask dimensions do not match for {image_filename}. Skipping.")
                    continue

                # 进行图像和掩码操作
                normalized_mask = mask / 255.0
                normalized_mask_3channel = cv2.merge([normalized_mask, normalized_mask, normalized_mask])
                result = (image * normalized_mask_3channel).astype(np.uint8)
                background = (1 - normalized_mask_3channel) * 255
                result = result + background.astype(np.uint8)

                # 将处理后的图像写回原位置（替换原图像）
                cv2.imwrite(image_path, result)
                print(f"Processed and saved: {image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference images')

    parser.add_argument('-img_dir', type=str, required=True)
    parser.add_argument('-matting_dir', type=str, required=True)
    args = parser.parse_args()

    process_images_and_masks(args.img_dir, args.matting_dir)