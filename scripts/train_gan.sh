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
	--niter 50 \
	--niter_decay 30 \
	--lr 0.0005 \
	--beta1 0.9 \
	--init_type normal \
	--init_gain 0.8 \
	--batch_size 512 \
	--path_length 15 \
	--a_mode zeros \
	--no_html \
	--use_dropout