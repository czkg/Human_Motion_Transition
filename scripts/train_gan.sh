set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/train.py \
	--dataroot ../dataset/Human3.6m/latent_path_new \
	--dataset_mode aligned_path \
	--name path_gan \
	--model path_gan \
	--input_latent 512 \
	--output_latent 512 \
	--checkpoints_dir ../results \
	--niter 20 \
	--niter_decay 80 \
	--lr 0.0001 \
	--lr_d 0.00001 \
	--beta1 0.9 \
	--init_type kaiming \
	--init_gain 0.8 \
	--batch_size 32 \
	--path_length 10 \
	--key_frames 0 14 \
	--a_mode zeros \
	--gan_mode wgangp \
	--norm instance \
	--lambda_L1 1000 \
	--lambda_GAN 1 \
	--where_add none \
	--z_size 0 \
	--num_downs 7 \
	--d_layers 3 \
	--nl lrelu \
	--no_html \
	--use_dropout