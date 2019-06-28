set -ex
#command
python3 ../src/vae.py \
	--batchsize 32 \
	--lr 1e-3 \
	--epochs 50 \
	--log_interval 10 \
	--save_interval 200