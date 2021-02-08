set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/train.py \
	--dataroot ../dataset/lafan/train_set \
	--dataset_mode lafan \
	--name rtncl \
	--model rtncl \
	--checkpoints_dir ../results \
	--niter 70 \
	--niter_decay 30 \
	--sigma 0.05 \
	--num_joints 21 \
	--x_dim 63 \
	--hidden_dim 128 \
	--lr 1e-5 \
	--beta1 0.9 \
	--init_type normal \
	--init_gain 0.8 \
	--batch_size 16 \
	--lafan_mode seq \
	--lafan_window 30 \
	--lafan_offset 10 \
	--lafan_samplerate 5 \
	--no_html \