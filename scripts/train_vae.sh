set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/train.py \
	--dataroot ../dataset/Human3.6m/heatmaps \
	--dataset_mode pose \
	--name vae \
	--model vae \
	--checkpoints_dir ../results \
	--niter 50 \
	--niter_decay 30 \
	--dim_heatmap 64 \
	--sigma 0.05 \
	--z_dim 512 \
	--pca_dim 2048 \
	--lr 0.00005 \
	--beta1 0.9 \
	--init_type normal \
	--init_gain 0.8 \
	--batch_size 32 \
	--no_html \
	--continue_train \
	--epoch 0