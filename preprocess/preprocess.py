import numpy as np
import os
import cv2
import glob
import os.path as osp
import argparse
import shutil
from PIL import Image
import torch
from torchvision import transforms
from maskmodel1 import BiSeNet  # Assuming BiSeNet is defined in a model.py file



def create_face_mask(im, parsing_anno, stride, save_im=False, save_path='vis_results/face_mask.jpg'):
    # Convert the image to a numpy array
    im = np.array(im)
    # Copy the parsing annotation for visualization
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    # Resize the parsing annotation according to the stride
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    
    # Create a blank mask with the same shape as the image
    face_mask = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]), dtype=np.uint8)
    
    # Define the face region labels (modify these according to your model's output)
    #face_labels = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]  # Example labels for face regions
    # face_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    face_labels = [1, 2, 3, 4, 5, 6 , 10, 11, 12, 13]
    # Set the face region to white in the mask
    for label in face_labels:
        face_mask[vis_parsing_anno == label] = 255
    
    #_, face_mask = cv2.threshold(face_mask, 100, 255, cv2.THRESH_BINARY)

    

    # Save the result if specified
    if save_im:
        # 将face_mask的shape改为原始图片大小
        face_mask = cv2.resize(face_mask, (2200, 3208), interpolation=cv2.INTER_NEAREST)

        unique_values = np.unique(face_mask)

        # 打印唯一的像素值
        #print("Unique pixel values in the mask:", unique_values)

        # # 检查是否只有0和255
        # if set(unique_values) == {0, 255}:
        #     print("The mask contains only 0 and 255.")
        # else:
        #     print("The mask contains other values besides 0 and 255.")

        cv2.imwrite(save_path, face_mask)
        # from PIL import Image
        # im = Image.fromarray(face_mask)
        # im.save(save_path)
    
    return face_mask

def face_mask_generation(i, respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):
    print(f"开始生成第{i}个视角人脸mask!!!")
    # Create the result directory if it does not exist
    if not os.path.exists(respth):
        os.makedirs(respth)

    # Define the number of classes
    n_classes = 19
    # Initialize the BiSeNet model
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    # Load the model checkpoint
    save_pth = osp.join('./res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    # Define the image transformation
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        # Iterate over all images in the data directory
        for image_path in os.listdir(dspth):
            # Load and resize the image
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            # Perform inference
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # Print unique parsing labels
            #print(np.unique(parsing))

            # Create and save the face mask
            create_face_mask(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))


def write_ims(seq, camid, time_step, img):
    time_str = str(time_step+1).zfill(5)  
    p = f"./multipleview/{seq}/cam{str(camid+1).zfill(2)}"
    path = f"./multipleview/{seq}/cam{str(camid+1).zfill(2)}/frame_{time_str}.png"
    os.makedirs(p, exist_ok=True)
    cv2.imwrite(path, img)
    return 0 

def get_image_names(folder_path):
    if not os.path.exists(folder_path):
        print("文件夹路径不存在")
        return []

    # 使用glob模块获取文件夹中所有图片文件的路径
    image_files = glob.glob(os.path.join(folder_path, '*.png'))

    # 从文件路径中提取文件名
    image_names = [os.path.basename(file) for file in image_files]
    return image_names

def filter_mask(mask):
    # 将白色部分（255）转换为1，其他部分为0
    mask[mask == 255] = 1

    # 计算每个连通区域的面积
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 遍历每个连通区域
    for i in range(1, num_labels):  # 跳过背景（标签0）
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area < 100000:
            # 将面积小于50的部分设置为0
            mask[labels == i] = 0

    # 将1转换回255以便显示
    mask[mask == 1] = 255
    return mask

def face_segmentation(image_dir, mask_dir, output_dir):
    print("开始切割人脸!!!")

    for i in range(1,17):
        # if i==2:
        #     continue
        os.makedirs(output_dir+f'{str(i).zfill(2)}', exist_ok=True)
        im_cam_dir = os.path.join(image_dir+f'{str(i).zfill(2)}')
        mk_cam_dir = os.path.join(mask_dir+f'{str(i).zfill(2)}')
        out_cam_dir = os.path.join(output_dir+f'{str(i).zfill(2)}')
        for image_path in sorted(os.listdir(im_cam_dir)):
            # 按顺序输出image_path
            image = cv2.imread(os.path.join(im_cam_dir,image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_mask = cv2.imread(os.path.join(mk_cam_dir,image_path), cv2.IMREAD_GRAYSCALE)
            image_mask = np.where(image_mask > 127, 255, 0).astype(np.uint8)
            image_mask = filter_mask(image_mask)
            image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2RGB)

            
            out = image * (image_mask // 255)
            # 将背景黑色改为白色
            out[image_mask == 0] = 255      
            
            # plt.rcParams['figure.figsize'] = [8, 8]
            # plt.rcParams['figure.dpi'] = 100
            # plt.axis('off')
            # plt.imshow(image_mask)
            # plt.show()
            cv2.imwrite(os.path.join(out_cam_dir,image_path), out[:, :, ::-1])

def get_background(background_dir, output_dir):
    cam1 = 'image_222200042.jpg'
    cam2 = 'image_222200044.jpg'
    cam3 = 'image_222200046.jpg'
    cam4 = 'image_222200040.jpg'
    cam5 = 'image_222200036.jpg'  #
    cam6 = 'image_222200048.jpg'
    cam7 = 'image_220700191.jpg'
    cam8 = 'image_222200041.jpg'
    cam9 = 'image_222200037.jpg'
    cam10 = 'image_222200038.jpg'  #
    cam11 = 'image_222200047.jpg'
    cam12 = 'image_222200043.jpg'
    cam13 = 'image_222200049.jpg'
    cam14 = 'image_222200039.jpg'
    cam15 = 'image_222200045.jpg' #
    cam16 = 'image_221501007.jpg'
    cam_identifiers = [cam1, cam2, cam3, cam4, cam5, cam6, cam7, cam8, cam9, cam10, cam11, cam12, cam13, cam14, cam15, cam16]
            # 遍历所有摄像头标识符
    os.makedirs(output_dir, exist_ok=True)
    for i, cam in enumerate(cam_identifiers):
        background_path = os.path.join(background_dir, cam)
        shutil.copy(background_path, os.path.join(output_dir,f'cam{str(i+1).zfill(2)}.png'))

def getfromvideo(seq, path, time):
    cam1 = 'cam_222200042.mp4'
    cam2 = 'cam_222200044.mp4'
    cam3 = 'cam_222200046.mp4'
    cam4 = 'cam_222200040.mp4'
    cam5 = 'cam_222200036.mp4'  #
    cam6 = 'cam_222200048.mp4'
    cam7 = 'cam_220700191.mp4'
    cam8 = 'cam_222200041.mp4'
    cam9 = 'cam_222200037.mp4'
    cam10 = 'cam_222200038.mp4'  #
    cam11 = 'cam_222200047.mp4'
    cam12 = 'cam_222200043.mp4'
    cam13 = 'cam_222200049.mp4'
    cam14 = 'cam_222200039.mp4'
    cam15 = 'cam_222200045.mp4' #
    cam16 = 'cam_221501007.mp4'
    cam_identifiers = [cam1, cam2, cam3, cam4, cam5, cam6, cam7, cam8, cam9, cam10, cam11, cam12, cam13, cam14, cam15, cam16]
            # 遍历所有摄像头标识符
    for i, cam in enumerate(cam_identifiers):
        # 打开视频文件
        
        video_path = os.path.join(path, cam)
        cap = cv2.VideoCapture(video_path)

        # 检查视频是否成功打开
        if not cap.isOpened():
            print("Error: Could not open video.")

        frame_count = 1
        output_folder = os.path.join(f'/media/DGST_data/Data/{seq}',f'cam{str(i+1).zfill(2)}')
        os.makedirs(output_folder, exist_ok=True)
        # 循环读取视频的每一帧
        while True:
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # 保存帧为图像文件
            frame_filename = os.path.join(output_folder, f'frame_{frame_count:05d}.png')
            cv2.imwrite(frame_filename, frame)
            if frame_count == time:
                break
            frame_count += 1
            
        # 释放视频捕获对象
        cap.release()
        print(f"第{i+1}视角, Saved {frame_count} frames.")
    print("getfromvideo done!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference images')

    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--id-name', type=str, required=True)
    parser.add_argument('--site', type=str, required=True)

    args = parser.parse_args()

    seq = args.id
    seq_name = args.id_name 
    site = args.site  # 需要跟踪的位置
    time = 150
    

    image_dir = f'/media/DGST_data/Data/{seq}/cam'
    mask_dir = f'/media/DGST_data/Data/{seq}/mask/cam'
    face_seg_output_dir = f'/media/DGST_data/Data/{seq}/segmentation/cam'
    background_dir = f'/media/Nersemble/BACKGROUND/{seq}/BACKGROUND'
    background_output_dir = f'/media/DGST_data/Data/{seq}/BACKGROUND'
    video_dir = f'/media/DGST_data/raw_data/{site}/{seq}/{seq_name}/'
    
    
    getfromvideo(seq, video_dir, time)
    # 生成人脸 mask
    for i in range(1, 17):
        data_dir = image_dir + f'{str(i).zfill(2)}'
        omask_dir = mask_dir + f'{str(i).zfill(2)}' 
        face_mask_generation(i, respth=omask_dir, dspth=data_dir, cp='79999_iter.pth')
    # 分割人脸
    # face_segmentation(image_dir, mask_dir, fance_seg_output_dir)

    # 获得原图背景
    get_background(background_dir, background_output_dir)

    print("数据处理已完成")
    
    