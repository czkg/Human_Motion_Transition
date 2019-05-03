set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/train.py \
	--dataroot ../dataset/Human3.6m/3d_poses \
	--dataset_mode pose \
	--name vae \
	--model vae \
	--checkpoints_dir ../results \
	--niter 30 \
	--niter_decay 30 \
	--dim_heatmap 64 \
	--sigma 0.2 \
	--z_dim 4096 \
	--pca_dim 32768 \
	--lr 0.0003 \
	--beta1 0.5