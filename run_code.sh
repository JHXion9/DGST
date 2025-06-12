#ID=$1 031 056 124
Face=("056") # 定义场景ID数组

# 遍历每个场景ID

for SCENE in ${Face[@]};
do
# CUDA_VISIBLE_DEVICES=1 python train.py -s "/media/DGST_data/Data/${SCENE}" --port 6022 --expname "out/${SCENE}" --configs arguments/multipleview/face.py 

CUDA_VISIBLE_DEVICES=1 python render.py --ID "${SCENE}" --source "/media/DGST_data/Data/${SCENE}" --model_path "/media/DGST_data/out/${SCENE}"  --skip_train --skip_video --configs arguments/multipleview/face.py  

CUDA_VISIBLE_DEVICES=1 python metrics.py --model_path "/media/DGST_data/out/${SCENE}"
done


# training instruct
# CUDA_VISIBLE_DEVICES=1  python train.py -s /media/DGST_data/Data/124 --port 6022 --expname out/124 --configs arguments/multipleview/face.py 
# CUDA_VISIBLE_DEVICES=0  python train.py -s /media/DGST_data/Data/056 --port 6019 --expname out/056 --configs arguments/multipleview/face.py &
# CUDA_VISIBLE_DEVICES=1  python train.py -s  /media/DGST_data/Data/031 --port 6020 --expname out/031 --configs arguments/multipleview/face.py 
# CUDA_VISIBLE_DEVICES=3  python train.py -s  data/multipleview/face2_0.7 --port 6020 --expname "multipleview/face2_0.7" --configs arguments/multipleview/face3.py 

# sleep 1m


# 2D point tracking

# CUDA_VISIBLE_DEVICES=1 python render.py --ID 056 --source /media/DGST_data/Data/056 --model_path /media/DGST_data/out/056  --skip_train --skip_video --configs arguments/multipleview/face.py  


# sleep 1m

# # # compute metrics
# CUDA_VISIBLE_DEVICES=2 python metrics.py --model_path /media/DGST_data/out/135 &
# CUDA_VISIBLE_DEVICES=0 python metrics.py --model_path /media/DGST_data/out/055 
# CUDA_VISIBLE_DEVICES=2 python metrics.py --model_path "output/multipleview/face2_0.5/" &
# CUDA_VISIBLE_DEVICES=3 python metrics.py --model_path "output/multipleview/face2_0.7/" 


# open3d可视化 点云运动轨迹，并保存为图片

# CUDA_VISIBLE_DEVICES=0 python visual.py --ID 055 --GT_name "stubble" --model_path /media/DGST_data/out/055  --skip_train --skip_video --configs arguments/multipleview/face.py 
# CUDA_VISIBLE_DEVICES=0 python visual_newcolorcode.py --ID 056 --GT_name "left_face" --model_path /media/DGST_data/out/056  --skip_train --skip_video --configs arguments/multipleview/face.py 
# CUDA_VISIBLE_DEVICES=1 python visual.py --ID 038 --GT_name "right-eyes_mouse" --model_path "output/multipleview/038"  --skip_train --skip_video --configs arguments/multipleview/face.py   
# CUDA_VISIBLE_DEVICES=0 python visual_newcolorcode.py --ID 055 --GT_name "stubble" --model_path /media/DGST_data/out/055  --skip_train --skip_video --configs arguments/multipleview/face.py 


# # 导出150帧点云
# python export_perframe_3DGS.py --iteration 14000 --configs arguments/multipleview/face.py --model_path /media/DGST_data/out/031 

