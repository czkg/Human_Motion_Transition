set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/test.py \
	--dataroot ../res/vaedmp \
	--dataset_mode path \
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
	--output_path ../res/rln