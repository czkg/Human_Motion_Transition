set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/train.py \
	--dataroot ../dataset/cmu_poses/small_normalized_poses \
	--dataset_mode pose2 \
	--name vae2 \
	--model vae2 \
	--checkpoints_dir ../results \
	--niter 40 \
	--niter_decay 20 \
	--sigma 0.05 \
	--x_dim 72 \
	--z_dim 128 \
	--pca_dim 256 \
	--lr 1e-5 \
	--beta1 0.9 \
	--init_type xavier \
	--init_gain 0.8 \
	--batch_size 16 \
	--no_html \