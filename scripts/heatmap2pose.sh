set -ex
# command
python3 ../src/heatmap2pose.py \
	--n_joints 21 \
	--dim_heatmap 64 \
	--input_path ../res/vaedmp_hm/heatmaps \
	--output_path ../res/vaedmp_hm/poses