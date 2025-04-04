workdir=$1
python scripts/extractimages.py $workdir

# Feature Extraction
colmap feature_extractor \
  --database_path ./colmap_tmp/database.db \
  --image_path ./colmap_tmp/images \
  --ImageReader.camera_model SIMPLE_PINHOLE \
  --ImageReader.single_camera 0 \
  --SiftExtraction.max_image_size 4096 \
  --SiftExtraction.max_num_features 16384 \
  --SiftExtraction.peak_threshold 0.0066699999999999997 \
  --SiftExtraction.edge_threshold 10 \
  --SiftExtraction.first_octave -1 \
  --SiftExtraction.num_octaves 4 \
  --SiftExtraction.octave_resolution 3 \
  --SiftExtraction.dsp_num_scales 10 \
  --SiftExtraction.dsp_min_scale 0.16667000000000001 \
  --SiftExtraction.dsp_max_scale 3 \
  --SiftExtraction.use_gpu 1 \
  --SiftExtraction.gpu_index -1 \
  --SiftExtraction.estimate_affine_shape 1 \
  --SiftExtraction.domain_size_pooling 1

# Exhaustive Matching
colmap exhaustive_matcher \
  --database_path ./colmap_tmp/database.db \
  --SiftMatching.use_gpu 1 \
  --SiftMatching.max_ratio 0.8 \
  --SiftMatching.max_distance 0.69999999999999996 \
  --SiftMatching.cross_check 1 \
  --SiftMatching.max_num_matches 32768 \
  --SiftMatching.gpu_index -1 \
  --ExhaustiveMatching.block_size 50

# Mapper (Sparse Reconstruction)
mkdir ./colmap_tmp/sparse
colmap mapper \
  --database_path ./colmap_tmp/database.db \
  --image_path ./colmap_tmp/images \
  --output_path ./colmap_tmp/sparse \
  --Mapper.multiple_models 1 \
  --Mapper.ba_refine_focal_length 1 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 1 \
  --Mapper.min_num_matches 15 \
  --Mapper.init_min_num_inliers 100 \
  --Mapper.init_max_error 4 \
  --Mapper.filter_max_reproj_error 4 \
  --Mapper.filter_min_tri_angle 1.5 \
  --Mapper.ba_local_function_tolerance 0 \
  --Mapper.ba_global_images_ratio 1.1 \
  --Mapper.ba_global_points_ratio 1.1 \
  --Mapper.ba_global_function_tolerance 0 \
  --Mapper.ba_global_max_refinement_change 0.0005 \
  --Mapper.ba_local_max_refinement_change 0.001 \
  --Mapper.abs_pose_max_error 12 \
  --Mapper.abs_pose_min_inlier_ratio 0.25 \
  --Mapper.init_max_forward_motion 0.95 \
  --Mapper.init_min_tri_angle 16 \
  --Mapper.max_reg_trials 3

# Copy sparse model
mkdir -p /media/DGST_data/Data/$workdir/sparse_
cp -r ./colmap_tmp/sparse/0/* /media/DGST_data/Data/$workdir/sparse_

# Image Undistortion
mkdir ./colmap_tmp/dense
colmap image_undistorter \
  --image_path ./colmap_tmp/images \
  --input_path ./colmap_tmp/sparse/0 \
  --output_path ./colmap_tmp/dense \
  --output_type COLMAP \
  --max_image_size 2000

# Patch Match Stereo (Dense Reconstruction)
colmap patch_match_stereo \
  --workspace_path ./colmap_tmp/dense \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency 1 \
  --PatchMatchStereo.gpu_index -1 \
  --PatchMatchStereo.window_radius 5 \
  --PatchMatchStereo.window_step 1 \
  --PatchMatchStereo.num_samples 15 \
  --PatchMatchStereo.num_iterations 5 \
  --PatchMatchStereo.filter 1 \
  --PatchMatchStereo.filter_min_ncc 0.1 \
  --PatchMatchStereo.filter_min_triangulation_angle 3 \
  --PatchMatchStereo.filter_geom_consistency_max_cost 1 \
  --PatchMatchStereo.filter_min_num_consistent 2 \
  --PatchMatchStereo.geom_consistency_regularizer 0.3 \
  --PatchMatchStereo.geom_consistency_max_cost 3

# Stereo Fusion
colmap stereo_fusion \
  --workspace_path ./colmap_tmp/dense \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path ./colmap_tmp/dense/fused.ply \
  --StereoFusion.min_num_pixels 5 \
  --StereoFusion.max_num_pixels 10000 \
  --StereoFusion.max_traversal_depth 100 \
  --StereoFusion.check_num_images 50 \
  --StereoFusion.max_reproj_error 2 \
  --StereoFusion.max_depth_error 0.01 \
  --StereoFusion.max_normal_error 10

# Downsample Point Cloud
python scripts/downsample_point.py ./colmap_tmp/dense/fused.ply /media/DGST_data/Data/$workdir/points3D_multipleview.ply

# Run LLFF's imgs2poses.py
python LLFF/imgs2poses.py ./colmap_tmp/

# Copy LLFF's results
cp ./colmap_tmp/poses_bounds.npy /media/DGST_data/Data/$workdir/poses_bounds_multipleview.npy

rm -rf ./colmap_tmp




