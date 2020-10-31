set -ex
GPU_ID=0
# command
# lr 1e-5/2e-4
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/train.py \
	--dataroot ../dataset/Human3.6m/latent_path_all \
	--dataset_mode aligned_path \
	--name path_gan \
	--model path_gan \
	--input_latent 512 \
	--output_latent 512 \
	--checkpoints_dir ../results \
	--niter 50 \
	--niter_decay 100 \
	--dim_heatmap 64 \
	--sigma 0.05 \
	--z_dim 512 \
	--pca_dim 2048 \
	--lr 2e-4 \
	--lr_d 1e-6 \
	--beta1 0.9 \
	--init_type kaiming \
	--init_gain 0.8 \
	--batch_size 32 \
	--path_length 10 \
	--key_frames 0 9 \
	--a_mode ones \
	--gan_mode lsgan \
	--norm batch \
	--lambda_BCE 1 \
	--lambda_BCE_decoder 1 \
	--lambda_GAN 1000 \
	--lambda_consistency 1000 \
	--lambda_bone 100 \
	--lambda_keyposes 100\
	--where_add all \
	--z_size 32 \
	--num_downs 7 \
	--d_layers 3 \
	--nl lrelu \
	--epoch2 0 \
	--no_html \
	--use_dropout