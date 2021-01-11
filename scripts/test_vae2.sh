set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/test.py \
	--dataroot ../dataset/Human3.6m/3d_poses/ \
	--dataset_mode h36m \
	--name vae2 \
	--model vae2 \
	--checkpoints_dir ../results \
	--sigma 0.05 \
	--x_dim 48 \
	--z_dim 128 \
	--pca_dim 256 \
	--batch_size 1 \
	--epoch latest \
	--output_path ../res/vae2 \
	--lafan_minmax ../dataset/lafan/minmax.npy