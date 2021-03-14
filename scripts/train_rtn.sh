set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/train.py \
	--dataroot ../res/vaedmp \
	--dataset_mode path \
	--name rtn \
	--model rtn \
	--checkpoints_dir ../results \
	--niter 70 \
	--niter_decay 30 \
	--sigma 0.05 \
	--num_joints 21 \
	--x_dim 32 \
	--hidden_dim 512 \
	--z_dim 128 \
	--lr 1e-4 \
	--beta1 0.9 \
	--init_type xavier \
	--init_gain 0.8 \
	--batch_size 16 \
	--plot_freq 100 \
	--lafan_use_heatmap \
	--no_html