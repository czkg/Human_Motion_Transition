set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/pytorch2tensorflow.py \
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
	--epoch 20 \
	--input_path ../dataset/Human3.6m/heatmaps/S1/Eating \
	--output_path ../res/vae \
	--path_length 15 \
	--gpu_ids -1