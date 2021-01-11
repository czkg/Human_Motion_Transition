set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/test.py \
	--dataroot ../dataset/lafan/dataset \
	--dataset_mode lafan \
	--name vaedmp \
	--model vaedmp \
	--checkpoints_dir ../results \
	--path_length 30 \
	--sigma 0.05 \
	--x_dim 63 \
	--z_dim 16 \
	--u_dim 16 \
	--hidden_dim 128 \
	--noise_dim 16 \
	--transform_dim 64 \
	--init_type xavier \
	--init_gain 0.8 \
	--batch_size 1 \
	--epoch latest \
	--input_path ../dataset/lafan/dataset \
	--output_path ../res/vaedmp \
	--lafan_minmax ../dataset/lafan/minmax.npy \
	--lafan_window 30 \
	--lafan_offset 10 \
	--lafan_samplerate 5