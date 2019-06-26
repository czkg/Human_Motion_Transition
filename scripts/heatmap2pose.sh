set -ex
# command
python3 ../src/heatmap2pose.py \
	--n_joints 17 \
	--dim_heatmap 64 \
	--input_path ../res/groundtruth/heatmaps \
	--output_path ../res/groundtruth/poses