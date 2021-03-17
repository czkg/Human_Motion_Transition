set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/train.py \
	--dataroot ../dataset/lafan/train_set_new \
	--dataset_mode lafan \
	--name vaedmp \
	--model vaedmp \
	--checkpoints_dir ../results \
	--niter 70 \
	--niter_decay 30 \
	--sigma 0.05 \
	--dim_heatmap 64 \
	--num_joints 21 \
	--z_dim 32 \
	--u_dim 32 \
	--hidden_dim 128 \
	--noise_dim 32 \
	--transform_dim 64 \
	--lr 1e-4 \
	--beta1 0.9 \
	--init_type kaiming \
	--init_gain 0.8 \
	--batch_size 16 \
	--lafan_mode seq \
	--lafan_window 50 \
	--lafan_offset 6 \
	--lafan_samplerate 2 \
	--lafan_use_heatmap \
	--no_html