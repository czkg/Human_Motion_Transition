set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/test.py \
	--dataroot ../dataset/cmu/test \
	--dataset_mode path \
	--name vaedmp \
	--model vaedmp \
	--checkpoints_dir ../results \
	--path_length 30 \
	--sigma 0.05 \
	--x_dim 72 \
	--z_dim 16 \
	--u_dim 16 \
	--hidden_dim 128 \
	--noise_dim 16 \
	--transform_dim 64 \
	--init_type xavier \
	--init_gain 0.8 \
	--batch_size 1 \
	--epoch latest \
	--input_path ../dataset/cmu/test \
	--output_path ../res/vaedmp