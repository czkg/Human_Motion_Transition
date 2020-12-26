set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/test.py \
	--dataset_mode pose2 \
	--name vae2 \
	--model vae2 \
	--checkpoints_dir ../results \
	--sigma 0.05 \
	--x_dim 72 \
	--z_dim 128 \
	--pca_dim 256 \
	--batch_size 1 \
	--epoch 60 \
	--input_path ../dataset/cmu_poses/test \
	--output_path ../res/vae2