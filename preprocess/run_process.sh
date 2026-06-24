ID=$1


python preprocess.py --id $ID --id-name EMO-2-surprise+fear



cd ../BackgroundMattingV2

for time in $(seq 1 150)
do

if [ ! -d "/media/DGST_data/Data/$ID/img" ]; then
    mkdir /media/DGST_data/Data/$ID/img
fi

for i in $(seq 1 16)
do
    # 构建源文件路径
    cam_id=$(printf "cam%02d" $i)
    formatted_time=$(printf "%05d" "$time")
    # 构建目标文件路径
    
    target_file="/media/DGST_data/Data/$ID/img/${cam_id}_frame_${formatted_time}.png"

    cp /media/DGST_data/Data/$ID/$cam_id/frame_${formatted_time}.png $target_file
done

python inference_images.py \
--model-refine-mode sampling \
--model-type mattingrefine \
--model-backbone resnet101 \
--output-type pha \
--images-src /media/DGST_data/Data/$ID/img \
--images-bgr /media/DGST_data/Data/$ID/BACKGROUND/ \
--output-dir /media/DGST_data/Data/$ID/matting \
--model-checkpoint /media/DenseGSTracking/BackgroundMattingV2/assets/pytorch_resnet101.pth

rm -rf /media/DGST_data/Data/$ID/img

done

cd ../preprocess
python matting.py -img_dir /media/DGST_data/Data/$ID -matting_dir /media/DGST_data/Data/$ID/matting/pha
