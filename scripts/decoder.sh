set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/decoder.py \
	--name vaedmp \
	--model vaedmp \
	--checkpoints_dir ../results \
	--dim_heatmap 64 \
	--sigma 0.05 \
	--dim_heatmap 64 \
	--num_joints 21 \
	--z_dim 32 \
	--u_dim 32 \
	--hidden_dim 128 \
	--noise_dim 32 \
	--transform_dim 64 \
	--init_type normal \
	--init_gain 0.8 \
	--batch_size 1 \
	--epoch 100 \
	--input_path ../res/rln \
	--output_path ../res/rln_heatmaps \
	--is_decoder