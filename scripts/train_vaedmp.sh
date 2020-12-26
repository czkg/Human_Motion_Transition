set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/train.py \
	--dataroot ../dataset/cmu/train \
	--dataset_mode path \
	--name vaedmp \
	--model vaedmp \
	--checkpoints_dir ../results \
	--niter 10 \
	--niter_decay 30 \
	--path_length 30 \
	--sigma 0.05 \
	--x_dim 72 \
	--z_dim 16 \
	--u_dim 16 \
	--hidden_dim 128 \
	--noise_dim 16 \
	--transform_dim 64 \
	--lr 1e-5 \
	--beta1 0.9 \
	--init_type xavier \
	--init_gain 0.8 \
	--batch_size 16 \
	--no_html \