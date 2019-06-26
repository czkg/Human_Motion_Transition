set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/decoder.py \
	--name vae \
	--model vae \
	--checkpoints_dir ../results \
	--dim_heatmap 64 \
	--sigma 0.05 \
	--z_dim 512 \
	--pca_dim 2048 \
	--init_type normal \
	--init_gain 0.8 \
	--batch_size 32 \
	--epoch 0 \
	--input_path ../res/gan \
	--output_path ../res/gan/heatmaps \
	--path_length 10 \
	--is_decoder