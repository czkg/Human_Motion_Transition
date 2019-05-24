set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/train.py \
	--dataroot ../dataset/Human3.6m/latent_path \
	--dataset_mode aligned_path \
	--name path_gan \
	--model path_gan \
	--input_latent 512 \
	--output_latent 512 \
	--checkpoints_dir ../results \
	--niter 100 \
	--niter_decay 30 \
	--lr 0.0001 \
	--beta1 0.9 \
	--init_type kaiming \
	--init_gain 0.8 \
	--batch_size 32 \
	--path_length 15 \
	--key_frames 0 14 \
	--a_mode linear \
	--gan_mode lsgan \
	--norm instance \
	--lambda_L1 1000 \
	--where_add none \
	--z_size 0 \
	--num_downs 20 \
	--nl lrelu \
	--no_html \
	--use_dropout