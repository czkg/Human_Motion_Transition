set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/compute_geodesic.py \
	--dataroot ../dataset/Human3.6m/heatmaps_nth \
	--name vae \
	--model vae \
	--checkpoints_dir ../results \
	--z_dim 512 \
	--pca_dim 2048 \
	--dim_heatmap 64 \
	--init_type normal \
	--init_gain 0.8 \
	--current_s S1 \
	--n_neighbors 4 \
	--z0 1 \
	--z1 136 \
	--gpu_ids -1 \
	--batch_size 512 \
	--store_path ../dataset/Human3.6m/graph \
	--epoch 20