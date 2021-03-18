set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/test.py \
	--dataroot ../dataset/lafan/train_set_new \
	--dataset_mode lafan \
	--name vaedmp \
	--model vaedmp \
	--checkpoints_dir ../results \
	--sigma 0.05 \
	--dim_heatmap 64 \
	--num_joints 21 \
	--z_dim 32 \
	--u_dim 32 \
	--hidden_dim 128 \
	--noise_dim 32 \
	--transform_dim 64 \
	--init_type kaiming \
	--init_gain 0.8 \
	--batch_size 1 \
	--epoch 100 \
	--lafan_mode seq \
	--lafan_window 30 \
	--lafan_offset 6 \
	--lafan_samplerate 2 \
	--lafan_use_heatmap \
	--output_path ../res/vaedmp