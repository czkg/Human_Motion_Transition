set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/train.py \
	--dataroot ../dataset/lafan/dataset \
	--dataset_mode lafan \
	--name vaedmp \
	--model vaedmp \
	--checkpoints_dir ../results \
	--niter 30 \
	--niter_decay 20 \
	--sigma 0.05 \
	--x_dim 66 \
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
	--lafan_mode seq \
	--lafan_minmax ../dataset/lafan/minmax.npy \
	--lafan_window 30 \
	--lafan_offset 10 \
	--lafan_samplerate 5 \
	--lafan_norm \
	--no_html \