set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/train.py \
	--dataroot ../dataset/cmu/sequence \
	--dataset_mode path \
	--name vaedmp \
	--model vaedmp \
	--checkpoints_dir ../results \
	--niter 50 \
	--niter_decay 50 \
	--path_length 30 \
	--sigma 0.05 \
	--x_dim 72 \
	--z_dim 32 \
	--u_dim 32 \
	--hidden_dim 512 \
	--noise_dim 32 \
	--transform_dim 128 \
	--lr 1e-4 \
	--beta1 0.9 \
	--init_type normal \
	--init_gain 0.8 \
	--batch_size 32 \
	--no_html \