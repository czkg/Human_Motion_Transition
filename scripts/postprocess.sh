set -ex
# command
python3 ../src/pre_post.py \
	--input /home/cz/cs/PG19/res/vae2 \
	--output /home/cz/cs/PG19/res/vae2_recon \
	--mode post \
	--m_path /home/cz/cs/PG19/dataset/cmu_poses/minmax.npy