set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/train.py \
	--dataroot ../dataset/Human3.6m/heatmaps \
	--dataset_mode pose \
	--name vae2 \
	--model vae2 \
	--checkpoints_dir ../results \
	--niter 50 \
	--niter_decay 10 \
	--sigma 0.05 \
	--dim_heatmap 64 \
	--num_joints 17 \
	--z_dim 512 \
	--pca_dim 2048 \
	--lr 1e-4 \
	--beta1 0.9 \
	--init_type kaiming \
	--init_gain 0.8 \
	--batch_size 16 \
	--no_html \