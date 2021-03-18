set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/test_rtn.py \
	--dataroot ../dataset/lafan/test_set_new \
	--dataset_mode lafan \
	--name rtn \
	--model rtn \
	--checkpoints_dir ../results \
	--sigma 0.05 \
	--num_joints 21 \
	--x_dim 32 \
	--hidden_dim 512 \
	--z_dim 128 \
	--epoch 100 \
	--init_type xavier \
	--init_gain 0.8 \
	--batch_size 1 \
	--epoch 100 \
	--lafan_mode seq \
	--lafan_window 30 \
	--lafan_offset 6 \
	--lafan_samplerate 2 \
	--lafan_use_heatmap \
	--output_path ../res/rtn