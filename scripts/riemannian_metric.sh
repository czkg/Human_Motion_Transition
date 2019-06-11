set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/compute_geodesic.py \
	--input_path ../dataset/Human3.6m/latent_nth \
	--name vae \
	--model vae \
	--checkpoints_dir ../results \
	--z_dim 512 \
	--pca_dim 2048 \
	--dim_heatmap 64 \
	--init_type normal \
	--init_gain 0.8 \
	--current_s S1 \
	--n_neighbors 10 \
	--z0 ../dataset/Human3.6m/latent_nth/S1/Directions/6.mat \
	--z1 ../dataset/Human3.6m/latent_nth/S1/Directions/1586.mat \
	--gpu_ids -1 \
	--batch_size 512 \
	--store_path ../dataset/Human3.6m/graph \
	--epoch 20