# 3D Facial Dense Gaussian Points-Based Dynamic Tracking Method

```bash
git clone https://github.com/JHXion9/DGST.git
cd DGST
conda create -n DGST python=3.7 
conda activate DGST

pip install -r requirements.txt
```
In our environment, we use pytorch=1.13.1+cu116.

## Required Submodules

This project depends on the following submodules located in `./submodules/`:

1. [Depth-Diff Gaussian Rasterization](https://github.com/ingra14m/depth-diff-gaussian-rasterization.git)
2. [Simple KNN](https://github.com/camenduru/simple-knn.git)
```bash
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

You need put [model_pth](https://drive.google.com/file/d/1F3IIl6m9By12-hM7X8wpchyFgMY2oPQE/view?usp=drive_link) in ./preprocess/

You also need place [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2.git)  and [LLFF](https://github.com/Fyusion/LLFF.git) in the parent directory for processing data. 


## Data Preparation

**For multipleviews scenes:**
If you want to train your own dataset of multipleviews scenes,you can orginize your dataset as follows:

```
├── data
|   | multipleview
│     | (your dataset name) 
│   	  | cam01
|     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | cam02
│     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | ...
```
After that,you can use the  `multipleviewprogress.sh` we provided to generate related data of poses and pointcloud.You can use it as follows:
```bash
bash multipleviewprogress.sh (youe dataset name)
# or
cd preprocess
sh run_process.sh {id}
```
You need to ensure that the data folder is orginized as follows after running multipleviewprogress.sh:
```
├── data
|   | multipleview
│     | (your dataset name) 
│   	  | cam01
|     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | cam02
│     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | ...
│   	  | sparse_
│     		  ├── cameras.bin
│     		  ├── images.bin
│     		  ├── ...
│   	  | points3D_multipleview.ply
│   	  | poses_bounds_multipleview.npy
```


## Training

For training multipleviews scenes,you are supposed to build a configuration file named (you dataset name).py under "./arguments/mutipleview",after that,run 

(or watching the run_code.sh for details)
```python
python train.py -s  data/multipleview/(your dataset name) --port 6017 --expname "multipleview/(your dataset name)" --configs arguments/multipleview/(you dataset name).py 
```

## Rendering

Watching the run_code.sh for details

## Evaluation

Watching the run_code.sh for details

