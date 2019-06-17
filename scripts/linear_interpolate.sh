set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/linear_interpolate.py \
	--name vae \
	--model vae \
	--checkpoints_dir ../results \
	--dim_heatmap 64 \
	--sigma 0.05 \
	--z_dim 512 \
	--pca_dim 2048 \
	--init_type normal \
	--init_gain 0.8 \
	--batch_size 512 \
	--epoch latest \
	--is_decoder