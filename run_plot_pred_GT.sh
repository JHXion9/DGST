#ID=$1 031 056 124
Face=("031" "033" "038" "056" "063" "124" "196" "264")  # 定义场景ID数组
# Face=("124" )

# 遍历每个场景ID

for SCENE in ${Face[@]};
do
CUDA_VISIBLE_DEVICES=3 python plot_pred_GT.py \
    --ID "${SCENE}" \
    --source "/media/DGST_data/Data/${SCENE}" \
    --model_path "/media/DGST_data/out/${SCENE}"  \
    --skip_train --skip_video --configs arguments/multipleview/face.py  

done
