set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/test.py \
	--name path_gan \
	--model path_gan \
	--checkpoints_dir ../results \
	--input_latent 512 \
	--output_latent 512 \
	--dim_heatmap 64 \
	--sigma 0.05 \
	--z_dim 512 \
	--pca_dim 2048 \
	--init_type kaiming \
	--init_gain 0.8 \
	--batch_size 32 \
	--epoch 100 \
	--epoch2 latest \
	--z0 ../dataset/Human3.6m/latent_nth/S5/Sitting/216.mat \
	--z1 ../dataset/Human3.6m/latent_nth/S5/Sitting/261.mat \
	--output_path ../res/gan \
	--path_length 10 \
	--a_mode ones \
	--norm instance \
	--where_add all \
	--z_size 32 \
	--num_downs 7 \
	--d_layers 3 \
	--nl lrelu \
	--use_dropout